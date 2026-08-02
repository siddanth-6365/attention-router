"""Reason selection and confidence calibration."""

import pytest

from attention_router import config, rationale


@pytest.fixture
def bank():
    return rationale.load_bank()


class TestBank:
    def test_every_entry_is_well_formed(self, bank):
        assert bank
        for entry in bank.values():
            assert entry["text"].strip()
            assert entry["actions"], entry["id"]
            assert set(entry["actions"]) <= set(config.ACTIONS), entry["id"]
            assert set(entry["message_types"]) <= set(config.MESSAGE_TYPES), entry["id"]

    def test_every_action_has_options(self, bank):
        for action in config.ACTIONS:
            assert any(action in e["actions"] for e in bank.values()), action

    def test_prompt_menu_can_be_narrowed(self, bank):
        muted = rationale.format_bank_for_prompt(bank, action="mute")
        assert muted
        assert all("mute" in bank[line.split()[0]]["actions"] for line in muted.splitlines())


class TestReasonResolution:
    def test_bank_id_wins(self, bank):
        first = next(iter(bank))
        assert rationale.resolve_reason(bank, first, "ignored", "mute") == bank[first]["text"]

    def test_unknown_id_falls_through_to_free_text(self, bank):
        assert rationale.resolve_reason(bank, "R999", "custom text", "mute") == "custom text"

    def test_blank_everything_falls_back_by_action(self, bank):
        for action in config.ACTIONS:
            assert rationale.resolve_reason(bank, None, "  ", action) == \
                rationale.FALLBACK_REASONS[action]

    def test_overlong_free_text_is_truncated(self, bank):
        resolved = rationale.resolve_reason(bank, None, "x" * 500, "digest")
        assert len(resolved) <= 220 and resolved.endswith("...")


class TestCalibration:
    def test_confidence_rises_with_evidence(self):
        for action in config.ACTIONS:
            values = [rationale.calibrate(action, s)
                      for s in config.EVIDENCE_STRENGTH_LEVELS]
            assert values == sorted(values), action
            assert values[0] < values[-1], action

    def test_result_always_lands_in_the_action_band(self):
        for action in config.ACTIONS:
            for strength in (*config.EVIDENCE_STRENGTH_LEVELS, "nonsense"):
                assert rationale.calibrate(action, strength) in config.CONFIDENCE_BANDS[action]

    def test_safety_block_is_maximally_confident(self):
        assert rationale.calibrate("mute", "none", safety_blocked=True) == \
            max(config.CONFIDENCE_BANDS["mute"])

    def test_bands_are_a_regular_grid_in_range(self):
        for action, band in config.CONFIDENCE_BANDS.items():
            assert list(band) == sorted(band), action
            assert all(0.0 <= v <= 1.0 for v in band), action
            steps = {round(b - a, 3) for a, b in zip(band, band[1:], strict=False)}
            assert len(steps) == 1, f"{action} band is not evenly spaced: {steps}"

    def test_notify_outranks_digest(self):
        """An interrupt should never be reported as less certain than a deferral."""
        assert min(config.CONFIDENCE_BANDS["notify"]) > max(config.CONFIDENCE_BANDS["digest"])
