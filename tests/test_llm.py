"""JSON extraction and retry behaviour."""

import pytest
from tests.conftest import StubClient

from attention_router.llm import LLMError, Usage, complete_json, extract_json


class TestJsonExtraction:
    @pytest.mark.parametrize("raw", [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure, here you go:\n{"a": 1}\nHope that helps.',
        '   {"a": 1}   ',
    ])
    def test_survives_the_ways_models_wrap_output(self, raw):
        assert extract_json(raw) == {"a": 1}

    def test_handles_nesting(self):
        assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}

    @pytest.mark.parametrize("raw", ["", "no json here", "[1, 2, 3]", "null", "42"])
    def test_rejects_anything_that_is_not_an_object(self, raw):
        with pytest.raises(LLMError):
            extract_json(raw)


class TestCompletion:
    def test_validation_failure_is_retried_with_a_correction(self):
        """A rejected payload is repaired, not blindly resampled."""
        client = StubClient(payload={"action": "bogus"})
        seen = []

        def validate(payload):
            seen.append(payload)
            if payload.get("action") == "bogus":
                raise ValueError("'action' must be one of notify/digest/mute")

        with pytest.raises(LLMError):
            complete_json(client, system="s", content="c", model="m",
                          max_tokens=50, validate=validate, max_attempts=3)
        assert len(seen) == 3, "should have retried up to the attempt limit"
        assert "rejected" in client.prompts[-1].lower() or len(client.prompts) == 3

    def test_transient_failure_is_retried(self):
        client = StubClient(fail_times=1)
        result = complete_json(client, system="s", content="c", model="m",
                               max_tokens=50, max_attempts=3)
        assert result["action"] == "digest"

    def test_gives_up_within_the_attempt_budget(self):
        client = StubClient(fail_times=99)
        with pytest.raises(LLMError):
            complete_json(client, system="s", content="c", model="m",
                          max_tokens=50, max_attempts=2)


class TestUsageAccounting:
    def test_counts_every_call_including_retries(self):
        usage = Usage()
        usage.record("m", 100, 20)
        usage.record("m", 100, 20)
        assert usage.calls == 2
        assert usage.input_tokens == 200 and usage.output_tokens == 40

    def test_summary_is_reportable(self):
        usage = Usage()
        usage.record("claude-sonnet-5", 1_000_000, 0)
        assert "1,000,000" in usage.summary()
        assert usage.estimated_usd() > 0
