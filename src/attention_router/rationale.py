"""Reason selection and confidence calibration.

Two of the six output columns are better served by structure than by free
generation, and both for the same underlying reason: routing decisions should
be comparable across messages.

`reason` - the router picks an entry from a fixed bank (`reasons.json`) by id.
Two messages routed for the same underlying cause then carry the same
explanation, which is what makes a digest of fifty suppressed messages
readable instead of fifty paraphrases of "this looked like spam". Free text
stays available for anything the bank does not cover, so novel situations are
never forced into a template.

`confidence` - each action owns a band, and position within the band is set by
how much evidence supports the call. A number derived from evidence strength
says something checkable; a number the model picks for itself mostly does not,
since language models are poorly calibrated at self-reported certainty.
"""

from __future__ import annotations

import json

from attention_router import config

FALLBACK_REASONS = {
    "notify": "This message needs the user's attention now.",
    "digest": "This message is useful but does not need to interrupt the user.",
    "mute": "This message is low value or unsafe for this user.",
}


def load_bank(path=None) -> dict[str, dict]:
    """Load the rationale bank as an id -> entry mapping."""
    source = path or config.REASONS_JSON
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in payload["reasons"]}


def format_bank_for_prompt(bank: dict[str, dict], action: str | None = None) -> str:
    """Render the bank as a menu, optionally narrowed to one action."""
    lines = []
    for entry in bank.values():
        if action and action not in entry["actions"]:
            continue
        types = "/".join(entry["message_types"])
        lines.append(f"{entry['id']} [{types}] {entry['text']}")
    return "\n".join(lines)


def resolve_reason(
    bank: dict[str, dict],
    template_id: str | None,
    override: str | None,
    action: str,
) -> str:
    """Pick the final sentence: bank entry, then free text, then a default."""
    entry = bank.get((template_id or "").strip().upper())
    if entry:
        return entry["text"]
    cleaned = " ".join((override or "").split())
    if cleaned:
        # Keep explanations to a single readable sentence.
        if len(cleaned) > 220:
            cleaned = cleaned[:217].rstrip() + "..."
        return cleaned
    return FALLBACK_REASONS.get(action, FALLBACK_REASONS["digest"])


def calibrate(action: str, evidence_strength: str, safety_blocked: bool = False) -> float:
    """Place confidence inside the band its action implies.

    A deterministic safety block is the most certain state the system reaches,
    so it takes the top of the mute band whatever the history says.
    """
    band = config.CONFIDENCE_BANDS.get(action) or config.CONFIDENCE_BANDS["digest"]
    if safety_blocked and action == "mute":
        return band[-1]
    try:
        index = config.EVIDENCE_STRENGTH_LEVELS.index(evidence_strength)
    except ValueError:
        index = 1
    return band[min(index, len(band) - 1)]


def build_bank_from_labels(labelled_csv, destination=None) -> dict:
    """Derive a bank from a labelled corpus instead of using the authored one.

    Optional, and unused by default. It exists for adapting the router to a
    corpus whose own explanations follow a house style worth matching: point it
    at a CSV carrying `reason`, `action`, and `message_type` columns.
    """
    from attention_router.loader import read_csv

    observed: dict[str, dict] = {}
    order: list[str] = []
    for row in read_csv(labelled_csv):
        reason = (row.get("reason") or "").strip()
        if not reason:
            continue
        if reason not in observed:
            observed[reason] = {"actions": set(), "message_types": set(), "count": 0}
            order.append(reason)
        entry = observed[reason]
        entry["actions"].add(row["action"])
        entry["message_types"].add(row["message_type"])
        entry["count"] += 1

    bank = {
        "_about": f"Derived from {getattr(labelled_csv, 'name', labelled_csv)}.",
        "reasons": [
            {
                "id": f"R{index:02d}",
                "text": reason,
                "actions": sorted(observed[reason]["actions"]),
                "message_types": sorted(observed[reason]["message_types"]),
                "observed_count": observed[reason]["count"],
            }
            for index, reason in enumerate(order, start=1)
        ],
    }
    if destination:
        destination.write_text(json.dumps(bank, indent=2) + "\n", encoding="utf-8")
    return bank
