"""Parsing and context assembly."""

from datetime import datetime

import pytest

from attention_router.loader import (
    build_dossier,
    counterpart_of,
    in_quiet_hours,
    to_bool,
    to_int,
    to_opt_int,
)
from tests.conftest import message


class TestNumericParsing:
    def test_blank_reaction_time_is_none_not_zero(self):
        """A user who never engaged has no reaction time.

        Collapsing blank to 0 would read as "reacted instantly" and invert the
        signal the whole router depends on.
        """
        assert to_opt_int("") is None
        assert to_opt_int("   ") is None
        assert to_int("") == 0

    @pytest.mark.parametrize("raw,expected", [("7", 7), (" 7 ", 7), ("x", 0), (None, 0)])
    def test_int_parsing_is_tolerant(self, raw, expected):
        assert to_int(raw) == expected

    def test_only_literal_one_is_true(self):
        assert to_bool("1")
        for falsey in ("0", "", "true", "True", "yes", None):
            assert not to_bool(falsey)


class TestQuietHours:
    def test_window_wrapping_midnight(self):
        """22:00-07:00 spans midnight; both sides must count as quiet."""
        assert in_quiet_hours("22:00-07:00", datetime(2026, 7, 31, 23, 30))
        assert in_quiet_hours("22:00-07:00", datetime(2026, 7, 31, 3, 0))
        assert not in_quiet_hours("22:00-07:00", datetime(2026, 7, 31, 12, 0))

    def test_same_day_window(self):
        assert in_quiet_hours("09:00-17:00", datetime(2026, 7, 31, 12, 0))
        assert not in_quiet_hours("09:00-17:00", datetime(2026, 7, 31, 20, 0))

    @pytest.mark.parametrize("window", ["", None, "garbage", "25:00-99:00"])
    def test_unparseable_window_is_never_quiet(self, window):
        assert not in_quiet_hours(window, datetime(2026, 7, 31, 3, 0))


class TestCounterpart:
    def test_sender_wins_over_business(self):
        """A row with both set is keyed by the user, and callers rely on that."""
        assert counterpart_of(message(sender_user_id="u_9", business_id="b_1")) == "u_9"

    def test_business_used_when_no_sender(self):
        assert counterpart_of(message(sender_user_id="", business_id="b_1")) == "b_1"

    def test_empty_when_neither(self):
        assert counterpart_of(message(sender_user_id="", business_id="")) == ""


class TestDossier:
    def test_sections_match_conversation_type(self, tiny_dataset):
        """Only the sections that apply are emitted, so the prompt carries no
        empty scaffolding for a conversation type that has no such context."""
        personal = build_dossier(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_seller"))
        assert "sender" in personal
        assert "business" not in personal and "group" not in personal

        business = build_dossier(tiny_dataset, message(
            user_id="u_engaged", conversation_type="business",
            sender_user_id="", business_id="biz_good"))
        assert "business" in business
        assert business["business"]["domain_matches_official"] is True

    def test_identical_message_yields_different_dossiers(self, tiny_dataset):
        """The premise of the system, asserted directly."""
        engaged = build_dossier(tiny_dataset, message(
            user_id="u_engaged", sender_user_id="u_seller"))
        rejecting = build_dossier(tiny_dataset, message(
            user_id="u_rejecting", sender_user_id="u_seller"))
        assert engaged != rejecting
        assert engaged["sender"]["historically_opened"] > 0
        assert rejecting["sender"]["historically_muted_after"] > 0
