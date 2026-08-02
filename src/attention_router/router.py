"""The routing decision for a single message.

Assembles the prompt, calls the model once at temperature 0, and validates the
result against the output contract before anyone downstream sees it. Three
things can never happen here: an action or type outside the allowed enums, an
evidence id that does not belong to this user, and a blank row.

The model is asked for a judgement, not for the whole answer. Safety blocks are
decided before the call, the reason comes from a bank, and confidence is
computed from evidence strength. What is left for the model is the part it is
actually good at: reading a message in context and picking a route.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from attention_router import coldstart, config, rationale, safety
from attention_router.llm import LLMError, complete_json
from attention_router.loader import Dataset, Row, build_dossier
from attention_router.retriever import RetrievalResult


@dataclass
class Decision:
    """One fully-resolved output row."""

    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str]
    source: str = "model"  # model | safety_block | fallback

    def as_row(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:g}",
            "evidence_message_ids": ";".join(self.evidence_message_ids)
            or config.NO_EVIDENCE,
        }


def _read_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


def _format_dossier(dossier: dict) -> str:
    """Readable key/value blocks. JSON would work but reads worse in a prompt."""
    lines = []
    for section, payload in dossier.items():
        if isinstance(payload, dict):
            if not payload:
                continue
            lines.append(f"{section}:")
            for key, value in payload.items():
                if value is None or value == "":
                    continue
                lines.append(f"  {key}: {value}")
        else:
            lines.append(f"{section}: {payload}")
    return "\n".join(lines)


def _format_evidence(result: RetrievalResult, prior=None) -> str:
    """Render the evidence block, falling back to a prior when history is absent."""
    if not result.evidence:
        fallback = prior.describe() if prior is not None else (
            "No historical messages found for this user and sender."
        )
        return (
            f"{fallback}\n\n"
            "There is no message to cite, so return an empty evidence list."
        )
    tier_note = {
        "counterpart": "same receiver and same sender/business - directly comparable",
        "group": "same receiver and same group, but a different sender",
        "user": "same receiver only - weakly related",
    }.get(result.tier, result.tier)

    lines = [f"Candidate evidence ({tier_note}):"]
    for item in result.evidence:
        lines.append(f"- id: {item.message_id}  (sent {item.created_at})")
        lines.append(f"  text: {item.text[:300] or '(no text)'}")
        lines.append(f"  this user's reaction: {item.describe()}")
    return "\n".join(lines)


def build_user_prompt(
    data: Dataset,
    message: Row,
    result: RetrievalResult,
    verdict: safety.SafetyVerdict,
    media_entry: dict | None,
    reason_bank: dict,
    prior: coldstart.Prior | None = None,
) -> str:
    """Fill the routing template for one message."""
    from attention_router import media as media_module

    media_text = media_module.as_text(media_entry)
    media_type = (message.get("media_type") or "").strip()
    if media_type and media_text:
        media_block = (
            f"Attached {media_type} (machine-extracted, also untrusted data):\n"
            f"{safety.wrap_untrusted(media_text)}"
        )
    elif media_type:
        media_block = (
            f"This message has an attached {media_type} that could not be "
            f"processed. Route on the remaining context."
        )
    else:
        media_block = "No media attached."

    return _read_prompt("router_user.md").format(
        message_id=message["message_id"],
        user_id=message["user_id"],
        conversation_type=message.get("conversation_type", ""),
        created_at=message.get("created_at", ""),
        forwarded_count=message.get("forwarded_count", "0"),
        message_block=safety.wrap_untrusted(message.get("message_text", "")),
        media_block=media_block,
        dossier_block=_format_dossier(build_dossier(data, message)),
        evidence_block=_format_evidence(result, prior),
        signal_block=json.dumps(result.signal(), sort_keys=True),
        safety_block=verdict.describe(),
        reason_bank=rationale.format_bank_for_prompt(reason_bank),
    )


def _validator(allowed_evidence: set[str]):
    """Reject anything that violates the output contract, with a fixable message."""

    def validate(payload: dict) -> None:
        action = str(payload.get("action", "")).strip().lower()
        if action not in config.ACTIONS:
            raise ValueError(
                f"'action' must be one of {list(config.ACTIONS)}, got {action!r}."
            )
        message_type = str(payload.get("message_type", "")).strip().lower()
        if message_type not in config.MESSAGE_TYPES:
            raise ValueError(
                f"'message_type' must be one of {list(config.MESSAGE_TYPES)}, "
                f"got {message_type!r}."
            )
        ids = payload.get("evidence_message_ids", [])
        if not isinstance(ids, list):
            raise ValueError("'evidence_message_ids' must be a list of strings.")
        invented = [i for i in ids if i not in allowed_evidence]
        if invented:
            raise ValueError(
                f"These evidence ids were not in the candidate list: {invented}. "
                f"Only use ids from the candidates, or return an empty list."
            )

    return validate


def fallback_decision(
    message: Row, result: RetrievalResult, verdict: safety.SafetyVerdict, bank: dict
) -> Decision:
    """Decide without the model, from the deterministic signals alone.

    Used when the safety layer already settled the question, and as the last
    resort if the model cannot produce a valid payload. Never returns a blank
    row, because a missing prediction scores worse than a defensible one.
    """
    signal = result.signal()
    verdict_name = signal.get("verdict", "no_signal")

    if verdict.blocked:
        action, message_type = verdict.action, verdict.message_type
    elif verdict_name in {"previously_reported", "actively_rejected"}:
        action, message_type = "mute", "spam"
    elif verdict.level == "flag":
        action, message_type = "mute", "spam"
    elif verdict_name == "urgent_engagement":
        action, message_type = "notify", "personal"
    else:
        action, message_type = "digest", "unknown"

    strength = result.strength()
    return Decision(
        message_id=message["message_id"],
        action=action,
        message_type=message_type,
        reason=rationale.resolve_reason(bank, None, _fallback_reason(verdict, verdict_name), action),
        confidence=rationale.calibrate(action, strength, verdict.blocked),
        evidence_message_ids=result.message_ids if verdict_name != "no_signal" else [],
        source="safety_block" if verdict.blocked else "fallback",
    )


def _fallback_reason(verdict: safety.SafetyVerdict, signal_verdict: str) -> str:
    if verdict.blocked and verdict.notes:
        return verdict.notes[0]
    return {
        "previously_reported": "This user has reported similar messages from this sender before.",
        "actively_rejected": "Similar historical messages were ignored, dismissed, or muted by this user.",
        "urgent_engagement": "This user responds to this sender within minutes, so the message is treated as time-sensitive.",
    }.get(signal_verdict, "")


def route(
    client,
    data: Dataset,
    message: Row,
    result: RetrievalResult,
    verdict: safety.SafetyVerdict,
    media_entry: dict | None,
    reason_bank: dict,
    votes: int = 1,
    prior: coldstart.Prior | None = None,
) -> Decision:
    """Produce the routing decision for one message.

    With `votes > 1` the model is sampled several times and the majority
    (action, message_type) wins. `temperature` cannot be pinned to 0 on
    current models, so identical requests can disagree on genuinely ambiguous
    boundary cases; voting recovers most of that stability and is the standard
    self-consistency accuracy trick besides.
    """
    # A deterministic block is already the answer; spending a call to confirm
    # it would only create an opportunity to disagree with it.
    if verdict.blocked:
        return fallback_decision(message, result, verdict, reason_bank)

    if votes > 1:
        ballots = [
            _sample_route(client, data, message, result, verdict, media_entry,
                          reason_bank, prior)
            for _ in range(votes)
        ]
        return _majority(ballots, message, result, verdict, reason_bank)

    return _sample_route(
        client, data, message, result, verdict, media_entry, reason_bank, prior
    )


def _majority(
    ballots: list[Decision],
    message: Row,
    result: RetrievalResult,
    verdict: safety.SafetyVerdict,
    reason_bank: dict,
) -> Decision:
    """Pick the most-voted (action, type); ties go to the earliest ballot."""
    valid = [b for b in ballots if b.source == "model"]
    if not valid:
        return fallback_decision(message, result, verdict, reason_bank)

    tally = Counter((b.action, b.message_type) for b in valid)
    best = max(tally.items(), key=lambda item: (item[1], -_first_index(valid, item[0])))
    winner = next(b for b in valid if (b.action, b.message_type) == best[0])

    # Unanimity is a real confidence signal; disagreement is a real doubt
    # signal. Reflect that by stepping the band position down when the panel
    # splits, rather than reporting the winner's own self-assessment.
    if best[1] < len(valid):
        winner.confidence = rationale.calibrate(
            winner.action,
            "weak" if best[1] * 2 <= len(valid) else "moderate",
        )
    return winner


def _first_index(ballots: list[Decision], key: tuple[str, str]) -> int:
    for index, ballot in enumerate(ballots):
        if (ballot.action, ballot.message_type) == key:
            return index
    return len(ballots)


def _sample_route(
    client,
    data: Dataset,
    message: Row,
    result: RetrievalResult,
    verdict: safety.SafetyVerdict,
    media_entry: dict | None,
    reason_bank: dict,
    prior: coldstart.Prior | None = None,
) -> Decision:
    """One model call, validated and resolved into a Decision."""
    allowed = set(result.message_ids)
    try:
        payload = complete_json(
            client,
            system=_read_prompt("router_system.md"),
            content=build_user_prompt(
                data, message, result, verdict, media_entry, reason_bank, prior
            ),
            model=config.ROUTER_MODEL,
            max_tokens=config.ROUTER_MAX_TOKENS,
            validate=_validator(allowed),
        )
    except LLMError:
        return fallback_decision(message, result, verdict, reason_bank)

    action = str(payload["action"]).strip().lower()
    strength = str(payload.get("evidence_strength", "")).strip().lower()
    if strength not in config.EVIDENCE_STRENGTH_LEVELS:
        strength = result.strength()

    cited = [i for i in payload.get("evidence_message_ids", []) if i in allowed]

    return Decision(
        message_id=message["message_id"],
        action=action,
        message_type=str(payload["message_type"]).strip().lower(),
        reason=rationale.resolve_reason(
            reason_bank,
            payload.get("reason_template_id"),
            payload.get("reason_override"),
            action,
        ),
        confidence=rationale.calibrate(
            action,
            strength if cited else ("weak" if prior and prior.available else "none"),
        ),
        evidence_message_ids=cited[: config.MAX_EVIDENCE_IDS],
        source="model",
    )
