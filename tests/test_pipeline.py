"""End-to-end: generate a corpus, route it, check the output contract.

This is the test that would have caught the failure mode that mattered most in
practice — a run that degrades to heuristics while still producing a
schema-valid file and exiting zero.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from attention_router import cli, config, media, rationale, router, safety
from attention_router.loader import load_dataset, read_csv
from tests.conftest import StubClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    """A small generated corpus, built once for the whole session."""
    out = tmp_path_factory.mktemp("corpus")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "synth" / "generate.py"),
         "--out", str(out), "--seed", "3", "--users", "12", "--groups", "4",
         "--route", "12", "--labelled", "8"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return out


@pytest.fixture
def wired(corpus, monkeypatch):
    """Point the package at the generated corpus and stub out the model."""
    monkeypatch.setattr(config, "DATASET_DIR", corpus)
    monkeypatch.setattr(config, "DATA_DIR", corpus)
    monkeypatch.setattr(config, "MESSAGES_CSV", corpus / "messages.csv")
    monkeypatch.setattr(config, "LABELLED_CSV", corpus / "labelled.csv")
    monkeypatch.setattr(config, "MEDIA_CACHE_JSON", corpus / "cache" / "media_cache.json")
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: StubClient())
    return corpus


class TestGeneratedCorpus:
    def test_every_expected_table_is_written(self, corpus):
        for name in ["users", "groups", "group_members", "business_accounts",
                     "user_business_history", "message_history", "message_events",
                     "messages", "labelled", "images", "voice_notes",
                     "daily_notification_summary"]:
            assert (corpus / f"{name}.csv").exists(), name

    def test_generation_is_deterministic(self, tmp_path):
        outputs = []
        for index in range(2):
            out = tmp_path / f"run{index}"
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "synth" / "generate.py"),
                 "--out", str(out), "--seed", "11", "--users", "8", "--groups", "3",
                 "--route", "6", "--labelled", "4"],
                capture_output=True, text=True, check=True)
            outputs.append((out / "messages.csv").read_text())
        assert outputs[0] == outputs[1], "same seed must produce identical output"

    def test_every_history_row_has_a_reaction(self, corpus):
        history = read_csv(corpus / "message_history.csv")
        events = {(r["user_id"], r["message_id"])
                  for r in read_csv(corpus / "message_events.csv")}
        for row in history:
            assert (row["user_id"], row["message_id"]) in events, row["message_id"]

    def test_blank_reaction_time_is_preserved_as_blank(self, corpus):
        """Written as empty, not zero — the loader relies on the distinction."""
        events = read_csv(corpus / "message_events.csv")
        ignored = [r for r in events if r["message_opened"] == "0"]
        assert ignored, "corpus should contain unopened messages"
        assert all(r["reaction_time_minutes"] == "" for r in ignored)

    def test_media_is_written_in_several_real_formats(self, corpus):
        signatures = set()
        for row in read_csv(corpus / "images.csv"):
            head = (corpus / row["file_path"]).read_bytes()[:16]
            signatures.add(media.detect_image_type(head, row["file_path"]))
        assert len(signatures) >= 2, f"expected mixed formats, got {signatures}"
        assert signatures <= media.SUPPORTED_IMAGE_TYPES

    def test_the_identical_text_pair_is_planted(self, corpus):
        """The headline phenomenon must survive generation."""
        rows = read_csv(corpus / "labelled.csv")
        by_text: dict[str, list[dict]] = {}
        for row in rows:
            if row["message_text"]:
                by_text.setdefault(row["message_text"], []).append(row)
        twins = [group for group in by_text.values()
                 if len(group) > 1
                 and len({r["user_id"] for r in group}) > 1
                 and len({r["action"] for r in group}) > 1]
        assert twins, "no identical-text pair with differing actions was generated"

    def test_impersonators_are_separable_from_legitimate_shorteners(self, corpus):
        businesses = read_csv(corpus / "business_accounts.csv")
        mismatched = [b for b in businesses
                      if b["official_domain"]
                      and b["official_domain"] != b["domain_used_by_sender"]]
        assert any(b["verified"] == "0" for b in mismatched), "no impostors generated"
        assert any(b["verified"] == "1" for b in mismatched), \
            "no verified-shortener control: the rule would be untestable"


class TestEndToEnd:
    def test_routes_and_satisfies_the_output_contract(self, wired, tmp_path):
        messages = read_csv(wired / "messages.csv")
        decisions = cli.route_messages(messages, workers=2, verbose=False)
        assert len(decisions) == len(messages)

        out = tmp_path / "output.csv"
        cli.write_output(decisions, out)
        cli.verify_output(out, messages)

        with out.open() as handle:
            rows = list(csv.DictReader(handle))
        assert [r["message_id"] for r in rows] == [m["message_id"] for m in messages]

    def test_output_order_is_independent_of_concurrency(self, wired):
        messages = read_csv(wired / "messages.csv")
        single = [d.message_id for d in cli.route_messages(messages, workers=1, verbose=False)]
        parallel = [d.message_id for d in cli.route_messages(messages, workers=8, verbose=False)]
        assert single == parallel

    def test_safety_blocks_bypass_the_model_entirely(self, wired):
        data = load_dataset(wired)
        messages = read_csv(wired / "messages.csv")
        blocked = [m for m in messages if safety.evaluate(data, m).blocked]
        assert blocked, "corpus should contain messages the safety layer settles"

        decisions = {d.message_id: d for d in cli.route_messages(messages, workers=2, verbose=False)}
        for msg in blocked:
            assert decisions[msg["message_id"]].source == "safety_block"

    def test_a_degraded_run_is_visible_not_silent(self, wired, monkeypatch):
        """Heuristic fallbacks are schema-valid, so they must be reported."""
        monkeypatch.setattr(cli, "build_client", lambda *a, **k: StubClient(fail_times=99))
        messages = read_csv(wired / "messages.csv")
        decisions = cli.route_messages(messages, workers=2, verbose=False)
        assert any(d.source == "fallback" for d in decisions)
        # The contract still passes, which is exactly why source tracking exists.
        assert all(d.action in config.ACTIONS for d in decisions)

    def test_cold_start_priors_reach_the_prompt(self, wired):
        data = load_dataset(wired)
        bank = rationale.load_bank()
        messages = read_csv(wired / "messages.csv")
        from attention_router import coldstart
        from attention_router.retriever import RetrievalResult

        cold = next((m for m in messages if coldstart.is_cold(data, m)), None)
        if cold is None:
            pytest.skip("this corpus happened to contain no cold senders")
        prior = coldstart.compute(data, cold)
        prompt = router.build_user_prompt(
            data, cold, RetrievalResult(), safety.SafetyVerdict(), None, bank, prior)
        assert "no direct history" in prompt.lower() or "no prior information" in prompt.lower()
