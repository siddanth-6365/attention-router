"""Score the router against the labelled samples.

    python code/evaluation/main.py                  # full pipeline
    python code/evaluation/main.py --no-evidence    # ablation
    python code/evaluation/main.py --errors         # show every miss

Calls `route_messages` from the production entry point, so what is measured
here is exactly what is submitted - there is no separate evaluation path that
could drift from the real one.

One caveat stated plainly: `reasons.json` is derived from this same file, so
the reason-similarity numbers are optimistic by construction. Run with
`--free-text-reasons` for the uncontaminated view. The action, message_type,
and evidence numbers are unaffected either way, because no label from this file
is ever consulted at inference time.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from attention_router import config, rationale
from attention_router.cli import load_dotenv, route_messages
from attention_router.llm import USAGE
from attention_router.loader import read_csv
from attention_router.retriever import tokenize

INPUT_COLUMNS = (
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
)


def strip_labels(row: dict) -> dict:
    """Hand the pipeline only the columns it gets at inference time."""
    return {key: row.get(key, "") for key in INPUT_COLUMNS}


def gold_evidence(row: dict) -> set[str]:
    raw = (row.get("evidence_message_ids") or "").strip()
    if not raw or raw.lower() == config.NO_EVIDENCE:
        return set()
    return {part.strip() for part in raw.split(";") if part.strip()}


def prf(true_positives: int, predicted: int, actual: int) -> tuple[float, float, float]:
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def token_f1(predicted: str, actual: str) -> float:
    """Bag-of-tokens overlap, the usual stand-in for judged similarity."""
    left, right = Counter(tokenize(predicted)), Counter(tokenize(actual))
    overlap = sum((left & right).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def confusion_matrix(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> str:
    counts = Counter(pairs)
    width = max(len(label) for label in labels) + 2
    lines = ["gold \\ pred".ljust(14) + "".join(name.rjust(width) for name in labels)]
    for gold in labels:
        row = gold.ljust(14)
        for predicted in labels:
            row += str(counts.get((gold, predicted), 0)).rjust(width)
        lines.append(row)
    return "\n".join(lines)


def report(rows: list[dict], decisions: list, show_errors: bool) -> float:
    predictions = {d.message_id: d for d in decisions}

    action_pairs: list[tuple[str, str]] = []
    type_pairs: list[tuple[str, str]] = []
    evidence_tp = evidence_pred = evidence_gold = 0
    exact_reasons = 0
    reason_scores: list[float] = []
    confidence_errors: list[float] = []
    in_band = 0
    misses: list[str] = []

    for row in rows:
        decision = predictions[row["message_id"]]
        action_pairs.append((row["action"], decision.action))
        type_pairs.append((row["message_type"], decision.message_type))

        predicted_ids = set(decision.evidence_message_ids)
        actual_ids = gold_evidence(row)
        evidence_tp += len(predicted_ids & actual_ids)
        evidence_pred += len(predicted_ids)
        evidence_gold += len(actual_ids)

        exact_reasons += decision.reason.strip() == row["reason"].strip()
        reason_scores.append(token_f1(decision.reason, row["reason"]))

        confidence_errors.append(abs(decision.confidence - float(row["confidence"])))
        in_band += decision.confidence in config.CONFIDENCE_BANDS.get(row["action"], ())

        if (
            decision.action != row["action"]
            or decision.message_type != row["message_type"]
        ):
            misses.append(
                f"  {row['message_id']:<16} gold={row['action']}/{row['message_type']:<15} "
                f"pred={decision.action}/{decision.message_type:<15} src={decision.source}\n"
                f"      text: {(row['message_text'] or '(media only)')[:96]}\n"
                f"      cited: {decision.evidence_message_ids or 'none'} "
                f"(gold {sorted(actual_ids) or 'none'})"
            )

    total = len(rows)
    action_hits = sum(g == p for g, p in action_pairs)
    type_hits = sum(g == p for g, p in type_pairs)
    precision, recall, f1 = prf(evidence_tp, evidence_pred, evidence_gold)

    print(f"\n{'=' * 66}\nSCORED {total} LABELLED MESSAGES\n{'=' * 66}")
    print(f"action accuracy        {action_hits / total:>7.1%}  ({action_hits}/{total})")
    print(f"message_type accuracy  {type_hits / total:>7.1%}  ({type_hits}/{total})")
    print(f"evidence precision     {precision:>7.1%}")
    print(f"evidence recall        {recall:>7.1%}")
    print(f"evidence F1            {f1:>7.1%}")
    print(f"reason exact match     {exact_reasons / total:>7.1%}")
    print(f"reason token F1        {sum(reason_scores) / total:>7.1%}")
    print(f"confidence MAE         {sum(confidence_errors) / total:>7.3f}")
    print(f"confidence in band     {in_band / total:>7.1%}")

    print(f"\n--- action confusion ---\n{confusion_matrix(action_pairs, config.ACTIONS)}")

    print("\n--- message_type per class ---")
    gold_counts = Counter(g for g, _ in type_pairs)
    pred_counts = Counter(p for _, p in type_pairs)
    hit_counts = Counter(g for g, p in type_pairs if g == p)
    for label in sorted(set(gold_counts) | set(pred_counts)):
        p, r, class_f1 = prf(hit_counts[label], pred_counts[label], gold_counts[label])
        print(
            f"  {label:<16} gold={gold_counts[label]:<3} pred={pred_counts[label]:<3} "
            f"P={p:5.1%} R={r:5.1%} F1={class_f1:5.1%}"
        )

    print(f"\ndecision sources: {dict(Counter(d.source for d in decisions))}")

    if misses:
        print(f"\n--- {len(misses)} rows missed on action or type ---")
        for miss in misses if show_errors else misses[:8]:
            print(miss)
        if not show_errors and len(misses) > 8:
            print(f"  ... {len(misses) - 8} more (use --errors)")

    return action_hits / total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.LABELLED_CSV)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    parser.add_argument("--votes", type=int, default=config.ROUTER_VOTES)
    parser.add_argument("--errors", action="store_true", help="print every miss")
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--no-coldstart", action="store_true")
    parser.add_argument("--mask-history", type=float, default=0.0,
                        metavar="FRACTION",
                        help="hide direct history for this fraction of rows, "
                             "simulating first-contact senders")
    parser.add_argument(
        "--free-text-reasons",
        action="store_true",
        help="disable the reason bank to measure uncontaminated reason quality",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    rows = read_csv(args.input)
    if args.limit:
        rows = rows[: args.limit]

    if args.free_text_reasons:
        # Empty the bank so the router must write its own sentence.
        rationale.load_bank = lambda path=None: {}

    ablations = [
        name
        for name, disabled in (
            ("media", args.no_media),
            ("evidence", args.no_evidence),
            ("safety", args.no_safety),
            ("cold-start priors", args.no_coldstart),
            ("reason-bank", args.free_text_reasons),
        )
        if disabled
    ]
    print(f"ablations disabled: {ablations or 'none (full pipeline)'}")

    decisions = route_messages(
        [strip_labels(row) for row in rows],
        use_media=not args.no_media,
        use_evidence=not args.no_evidence,
        use_safety=not args.no_safety,
        use_coldstart=not args.no_coldstart,
        mask_history=args.mask_history,
        workers=args.workers,
        votes=args.votes,
    )
    report(rows, decisions, args.errors)
    print(f"usage:             {USAGE.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
