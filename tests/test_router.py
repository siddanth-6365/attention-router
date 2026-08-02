"""Output contract enforcement and the fallback path."""

import pytest

from attention_router import config, rationale, router, safety
from attention_router.retriever import EvidenceRetriever, RetrievalResult
from tests.conftest import StubClient, message

LISTING = "Selling a denim jacket, size M. Pickup near Gate 2 this weekend."


@pytest.fixture
def bank():
    return rationale.load_bank()


class TestValidator:
    def test_accepts_a_well_formed_payload(self):
        validate = router._validator({"h_00"})
        validate({"action": "mute", "message_type": "scam",
                  "evidence_message_ids": ["h_00"]})

    def test_normalises_case(self):
        router._validator(set())({"action": "Mute", "message_type": "Scam",
                                  "evidence_message_ids": []})

    @pytest.mark.parametrize("payload", [
        {"action": "NOTIFY_NOW", "message_type": "scam", "evidence_message_ids": []},
        {"action": "mute", "message_type": "phishing", "evidence_message_ids": []},
        {"action": "mute", "message_type": "scam", "evidence_message_ids": "h_00"},
        {"action": "mute", "message_type": "scam", "evidence_message_ids": ["invented"]},
        {"message_type": "scam", "evidence_message_ids": []},
    ])
    def test_rejects_every_contract_violation(self, payload):
        with pytest.raises(ValueError):
            router._validator({"h_00"})(payload)

    def test_the_model_cannot_invent_a_citation(self):
        """Evidence must come from the candidates we supplied, or not at all."""
        with pytest.raises(ValueError, match="not in the candidate list"):
            router._validator({"h_00"})({
                "action": "mute", "message_type": "scam",
                "evidence_message_ids": ["h_00", "fabricated_id"]})


class TestSerialisation:
    def test_row_matches_the_output_contract_exactly(self):
        decision = router.Decision("m1", "notify", "urgent", "because", 0.89, ["h_00"])
        row = decision.as_row()
        assert list(row) == config.OUTPUT_HEADER
        assert row["evidence_message_ids"] == "h_00"

    def test_multiple_citations_are_semicolon_joined(self):
        row = router.Decision("m1", "mute", "spam", "r", 0.83, ["a", "b"]).as_row()
        assert row["evidence_message_ids"] == "a;b"

    def test_absent_evidence_is_the_literal_none(self):
        row = router.Decision("m1", "digest", "unknown", "r", 0.78, []).as_row()
        assert row["evidence_message_ids"] == "none"


class TestSafetyShortCircuit:
    def test_a_blocked_message_never_reaches_the_model(self, tiny_dataset, bank):
        msg = message(message_text=(
            "Your account will be blocked in 2 hours. Confirm your password and OTP now."))
        verdict = safety.evaluate(tiny_dataset, msg)
        assert verdict.blocked

        client = StubClient()
        decision = router.route(client, tiny_dataset, msg, RetrievalResult(),
                                verdict, None, bank)
        assert client.prompts == [], "the model must not be called on a hard block"
        assert decision.source == "safety_block"
        assert (decision.action, decision.message_type) == ("mute", "scam")
        assert decision.confidence == max(config.CONFIDENCE_BANDS["mute"])


class TestModelPath:
    def test_decision_is_built_from_the_model_reply(self, tiny_dataset, bank):
        retriever = EvidenceRetriever(tiny_dataset)
        msg = message(user_id="u_rejecting", sender_user_id="u_seller", message_text=LISTING)
        result = retriever.retrieve(msg)
        client = StubClient(payload={
            "action": "mute", "message_type": "promotion",
            "reason_template_id": "R22", "reason_override": "",
            "evidence_message_ids": result.message_ids,
            "evidence_strength": "strong"})

        decision = router.route(client, tiny_dataset, msg, result,
                                safety.SafetyVerdict(), None, bank)
        assert decision.source == "model"
        assert (decision.action, decision.message_type) == ("mute", "promotion")
        assert decision.reason == bank["R22"]["text"]
        assert decision.confidence in config.CONFIDENCE_BANDS["mute"]

    def test_prompt_fences_untrusted_content(self, tiny_dataset, bank):
        client = StubClient()
        router.route(client, tiny_dataset, message(message_text="hi"), RetrievalResult(),
                     safety.SafetyVerdict(), None, bank)
        assert "UNTRUSTED_MESSAGE_CONTENT" in client.prompts[0]

    def test_braces_in_message_text_do_not_break_templating(self, tiny_dataset, bank):
        client = StubClient()
        msg = message(message_text="Use {this} and {{that}} at 100% capacity")
        router.route(client, tiny_dataset, msg, RetrievalResult(),
                     safety.SafetyVerdict(), None, bank)
        assert "{this}" in client.prompts[0]


class TestFallback:
    def test_model_failure_still_produces_a_valid_row(self, tiny_dataset, bank):
        """A missing prediction scores worse than a defensible one."""
        retriever = EvidenceRetriever(tiny_dataset)
        msg = message(user_id="u_rejecting", sender_user_id="u_seller", message_text=LISTING)
        decision = router.route(StubClient(fail_times=99), tiny_dataset, msg,
                                retriever.retrieve(msg), safety.SafetyVerdict(), None, bank)
        assert decision.source == "fallback"
        assert decision.action in config.ACTIONS
        assert decision.message_type in config.MESSAGE_TYPES
        assert decision.reason.strip()

    def test_fallback_respects_recorded_rejection(self, tiny_dataset, bank):
        retriever = EvidenceRetriever(tiny_dataset)
        msg = message(user_id="u_rejecting", sender_user_id="u_seller", message_text=LISTING)
        decision = router.route(StubClient(fail_times=99), tiny_dataset, msg,
                                retriever.retrieve(msg), safety.SafetyVerdict(), None, bank)
        assert decision.action == "mute"
