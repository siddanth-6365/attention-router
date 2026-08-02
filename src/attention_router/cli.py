"""Command line interface for the attention router.

    attention-router route                    route every message -> output.csv
    attention-router route --limit 5          smoke test
    attention-router explain msg_012          trace one decision end to end
    attention-router evaluate                 score against the labelled corpus
    attention-router route --media-only       fill the media cache, then stop

Output rows are written in input order regardless of how the work was
scheduled, so concurrency never changes the file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from attention_router import coldstart, config, media, rationale, router, safety
from attention_router.llm import USAGE, build_client
from attention_router.loader import load_dataset, read_csv
from attention_router.retriever import EvidenceRetriever, RetrievalResult


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from a .env at the repo root, if one exists.

    Does not overwrite variables already in the environment, so an exported
    key always wins over the file.
    """
    env_file = path or (config.PROJECT_ROOT / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _media_id(message: dict) -> str:
    return (message.get("media_id") or "").strip()


def route_messages(
    messages: list[dict],
    *,
    use_media: bool = True,
    use_evidence: bool = True,
    use_safety: bool = True,
    use_coldstart: bool = True,
    mask_history: float = 0.0,
    workers: int = config.MAX_WORKERS,
    refresh_media: bool = False,
    votes: int = config.ROUTER_VOTES,
    verbose: bool = True,
) -> list[router.Decision]:
    """Run the full pipeline over a list of message rows.

    Shared by `main` and by the evaluation harness, so the scored path and the
    submitted path are the same code.
    """
    data = load_dataset()
    reason_bank = rationale.load_bank()

    cache: dict[str, dict] = {}
    if use_media:
        cache = media.ensure_media(
            data,
            media.referenced_media_ids(messages, list(data.history.values())),
            refresh=refresh_media,
            verbose=verbose,
        )

    retriever = EvidenceRetriever(data, media_text=media.text_index(cache))

    # Deterministic masking: hide direct history for a fraction of rows to
    # simulate first-contact senders and measure whether priors recover the
    # accuracy that retrieval would otherwise have supplied.
    masked = set()
    if mask_history > 0:
        ordered = sorted(m["message_id"] for m in messages)
        masked = set(ordered[: int(len(ordered) * mask_history)])

    prepared = []
    for message in messages:
        hidden = message["message_id"] in masked
        result = (
            retriever.retrieve(message)
            if use_evidence and not hidden
            else RetrievalResult()
        )
        verdict = (
            safety.evaluate(data, message, media.as_text(cache.get(_media_id(message))))
            if use_safety
            else safety.SafetyVerdict()
        )
        # A prior only ever fills a gap; direct history always wins.
        prior = (
            coldstart.compute(data, message, has_direct_history=bool(result.evidence))
            if use_coldstart
            else coldstart.Prior()
        )
        prepared.append((message, result, verdict, prior))

    needs_model = [item for item in prepared if not item[2].blocked]
    client = build_client() if needs_model else None

    def decide(item):
        message, result, verdict, prior = item
        return router.route(
            client,
            data,
            message,
            result,
            verdict,
            cache.get(_media_id(message)),
            reason_bank,
            votes=votes,
            prior=prior,
        )

    if verbose:
        blocked = len(prepared) - len(needs_model)
        print(
            f"routing {len(prepared)} messages "
            f"({blocked} settled by the safety layer, "
            f"{len(needs_model)} to the model at {votes} vote(s) each)"
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(decide, prepared))


def write_output(decisions: list[router.Decision], destination: Path) -> None:
    """Write output.csv with the exact required header and column order."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.OUTPUT_HEADER)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.as_row())


def verify_output(destination: Path, messages: list[dict]) -> None:
    """Fail loudly if the submission file breaks the contract."""
    with open(destination, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == config.OUTPUT_HEADER, (
            f"header mismatch: {reader.fieldnames}"
        )
        rows = list(reader)

    expected = [m["message_id"] for m in messages]
    assert [r["message_id"] for r in rows] == expected, (
        "output rows must match messages.csv one-to-one, in order"
    )
    for row in rows:
        assert row["action"] in config.ACTIONS, row
        assert row["message_type"] in config.MESSAGE_TYPES, row
        assert row["reason"].strip(), row
        assert 0.0 <= float(row["confidence"]) <= 1.0, row
        assert row["evidence_message_ids"].strip(), row
    print(f"verified {len(rows)} rows against the output contract")


def route_command(argv: list[str] | None = None) -> int:
    """Route every message in the input corpus and write output.csv."""
    parser = argparse.ArgumentParser(prog="attention-router route")
    parser.add_argument("--input", type=Path, default=config.MESSAGES_CSV)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_CSV)
    parser.add_argument("--limit", type=int, default=None, help="route only the first N")
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    parser.add_argument("--votes", type=int, default=config.ROUTER_VOTES,
                        help="samples per message; majority wins")
    parser.add_argument("--media-only", action="store_true", help="fill the cache and exit")
    parser.add_argument("--refresh-media", action="store_true", help="ignore cached media")
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--no-coldstart", action="store_true",
                        help="disable priors for senders with no history")
    args = parser.parse_args(argv)

    load_dotenv()
    messages = read_csv(args.input)
    if args.limit:
        messages = messages[: args.limit]

    if args.media_only:
        data = load_dataset()
        cache = media.ensure_media(
            data,
            media.referenced_media_ids(messages, list(data.history.values())),
            refresh=args.refresh_media,
        )
        print(f"media cache holds {len(cache)} entries")
        return 0

    decisions = route_messages(
        messages,
        use_media=not args.no_media,
        use_evidence=not args.no_evidence,
        use_safety=not args.no_safety,
        use_coldstart=not args.no_coldstart,
        workers=args.workers,
        refresh_media=args.refresh_media,
        votes=args.votes,
    )

    write_output(decisions, args.output)
    print(f"wrote {len(decisions)} rows to {args.output}")
    if args.limit is None and args.input == config.MESSAGES_CSV:
        verify_output(args.output, messages)

    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    for decision in decisions:
        counts[decision.action] = counts.get(decision.action, 0) + 1
        sources[decision.source] = sources.get(decision.source, 0) + 1
    print("action distribution:", dict(sorted(counts.items())))
    print("decision sources:  ", dict(sorted(sources.items())))
    print("usage:             ", USAGE.summary())

    # A heuristic fallback still satisfies the output contract, so without
    # this the run looks clean while quietly degrading. Rate limiting is the
    # usual cause; lower --workers or --votes and rerun.
    degraded = sources.get("fallback", 0)
    if degraded:
        share = degraded / len(decisions)
        print(
            f"\n!! WARNING: {degraded}/{len(decisions)} rows ({share:.0%}) fell back "
            f"to heuristics because the model call failed.\n"
            f"   These rows are NOT model decisions. Rerun with fewer workers "
            f"before submitting."
        )
        if share > 0.05:
            return 1
    return 0


def explain_command(argv: list[str] | None = None) -> int:
    """Print the full decision trace for one message."""
    from attention_router import explain as explain_module

    parser = argparse.ArgumentParser(prog="attention-router explain")
    parser.add_argument("message_id")
    parser.add_argument("--live", action="store_true",
                        help="make the real model call instead of stopping before it")
    parser.add_argument("--show-prompt", action="store_true",
                        help="also print the exact prompt sent to the model")
    args = parser.parse_args(argv)

    load_dotenv()
    message = explain_module.find_message(args.message_id)
    if message is None:
        print(f"No message '{args.message_id}' in {config.MESSAGES_CSV.parent}.")
        print("Generate a corpus first:  python synth/generate.py")
        return 1

    client = build_client() if args.live else None
    print(explain_module.trace(message, client=client, show_prompt=args.show_prompt))
    return 0


def evaluate_command(argv: list[str] | None = None) -> int:
    """Score the router against a labelled corpus."""
    from attention_router.evaluate import main as evaluate_main

    return evaluate_main(argv)


COMMANDS = {
    "route": route_command,
    "explain": explain_command,
    "evaluate": evaluate_command,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    if argv and argv[0] in {"-h", "--help"}:
        print(__doc__)
        print("Commands:")
        for name, handler in COMMANDS.items():
            print(f"  {name:<10} {(handler.__doc__ or '').strip().splitlines()[0]}")
        print("\nRun `attention-router <command> --help` for per-command options.")
        return 0
    # No subcommand: route, so the common case stays a single word.
    return route_command(argv)


if __name__ == "__main__":
    sys.exit(main())
