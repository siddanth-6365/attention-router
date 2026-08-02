"""Human-readable trace of how one message was routed.

Every stage of this pipeline is inspectable on its own, but reading them
separately does not tell you why a particular message ended up where it did.
This prints the whole chain in order - what was known about the receiver, what
was retrieved and how it scored, what the safety rules made of it, what the
model decided, and how confidence was arrived at.

It is the fastest way to answer "why was this muted?", which in a notification
system is a question users and operators ask constantly.
"""

from __future__ import annotations

from attention_router import coldstart, config, media, rationale, router, safety
from attention_router.loader import Dataset, Row, build_dossier, load_dataset, read_csv
from attention_router.retriever import EvidenceRetriever

RULE = "─" * 72


def _heading(text: str) -> str:
    return f"\n{RULE}\n{text}\n{RULE}"


def _render_dossier(dossier: dict, indent: str = "  ") -> str:
    lines = []
    for section, payload in dossier.items():
        if isinstance(payload, dict):
            if not payload:
                continue
            lines.append(f"{indent}{section}:")
            for key, value in payload.items():
                if value not in (None, ""):
                    lines.append(f"{indent}  {key}: {value}")
        else:
            lines.append(f"{indent}{section}: {payload}")
    return "\n".join(lines)


def trace(
    message: Row,
    data: Dataset | None = None,
    *,
    client=None,
    show_prompt: bool = False,
) -> str:
    """Build the full decision trace for one message."""
    data = data or load_dataset()
    cache = media.load_cache()
    retriever = EvidenceRetriever(data, media_text=media.text_index(cache))
    bank = rationale.load_bank()

    media_entry = cache.get((message.get("media_id") or "").strip())
    media_text = media.as_text(media_entry)
    result = retriever.retrieve(message)
    verdict = safety.evaluate(data, message, media_text)
    prior = coldstart.compute(data, message, has_direct_history=bool(result.evidence))

    out: list[str] = []
    out.append(_heading(f"MESSAGE  {message['message_id']}  ->  user {message['user_id']}"))
    out.append(f"  conversation : {message.get('conversation_type', '')}")
    counterpart = (message.get("sender_user_id") or message.get("business_id") or "-")
    out.append(f"  from         : {counterpart}")
    out.append(f"  sent         : {message.get('created_at', '')}")
    out.append(f"  forwarded    : {message.get('forwarded_count', '0')}x")
    body = (message.get("message_text") or "").strip()
    out.append(f"  text         : {body[:200] if body else '(no text - media only)'}")
    if media_text:
        out.append(f"  {message.get('media_type', 'media')} content:")
        for line in media_text.splitlines()[:6]:
            out.append(f"      {line[:150]}")

    out.append(_heading("1. RECEIVER CONTEXT"))
    out.append(_render_dossier(build_dossier(data, message)))

    out.append(_heading("2. EVIDENCE RETRIEVAL"))
    if result.evidence:
        tier_label = {
            "counterpart": "same receiver + same sender  (strongest)",
            "group": "same receiver + same group",
            "user": "same receiver, any conversation  (weakest)",
        }.get(result.tier, result.tier)
        out.append(f"  tier: {tier_label}")
        out.append(f"  cited {len(result.evidence)} of the candidates ranked by BM25:")
        for item in result.evidence:
            out.append(f"    [{item.message_id}]  bm25={item.score:<7} sent {item.created_at}")
            out.append(f"        text     : {(item.text or '(none)')[:130]}")
            out.append(f"        reaction : {item.describe()}")
        signal = result.signal()
        out.append(f"  behavioural verdict : {signal['verdict']}")
        out.append(f"  evidence strength   : {result.strength()}")
    else:
        out.append("  no direct history for this receiver and sender")

    out.append(_heading("3. COLD-START PRIOR"))
    if prior.available:
        out.append(f"  source       : {prior.kind}")
        out.append(f"  leaning      : {prior.leaning}  ({prior.observations} observations)")
        out.append(f"  basis        : {prior.basis}")
        out.append("  note         : a prior caps confidence at the bottom of its band")
    elif result.evidence:
        out.append("  not needed - direct history was available")
    else:
        out.append("  none available - nothing is known about this sender")

    out.append(_heading("4. DETERMINISTIC SAFETY PASS"))
    out.append(f"  level: {verdict.level}")
    out.append("  " + verdict.describe().replace("\n", "\n  "))
    if verdict.blocked:
        out.append("  -> the model is NOT called; this decision is already settled")

    out.append(_heading("5. DECISION"))
    if client is None and not verdict.blocked:
        out.append("  (no client supplied - skipping the model call)")
        out.append("  Pass --live to make the real routing call.")
        return "\n".join(out)

    decision = router.route(
        client, data, message, result, verdict, media_entry, bank, prior=prior
    )
    band = config.CONFIDENCE_BANDS.get(decision.action, ())
    position = band.index(decision.confidence) + 1 if decision.confidence in band else "?"
    out.append(f"  action       : {decision.action}")
    out.append(f"  message_type : {decision.message_type}")
    out.append(f"  reason       : {decision.reason}")
    out.append(f"  confidence   : {decision.confidence}  "
               f"(position {position} of {len(band)} in the {decision.action} band {list(band)})")
    out.append(f"  evidence     : {';'.join(decision.evidence_message_ids) or 'none'}")
    out.append(f"  decided by   : {decision.source}")

    if show_prompt:
        out.append(_heading("PROMPT SENT TO THE MODEL"))
        out.append(router.build_user_prompt(
            data, message, result, verdict, media_entry, bank, prior
        ))
    return "\n".join(out)


def find_message(message_id: str, *paths) -> Row | None:
    """Look a message up across the routable and labelled corpora."""
    for path in paths or (config.MESSAGES_CSV, config.LABELLED_CSV):
        if not path.exists():
            continue
        for row in read_csv(path):
            if row["message_id"] == message_id:
                return row
    return None
