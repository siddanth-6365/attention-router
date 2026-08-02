"""Shared fixtures.

Everything here is offline. The suite never makes a network call and never
needs an API key, which is what lets CI run it on every push.

The `tiny_dataset` fixture is the important one: rather than depending on a
generated corpus, most tests build the smallest possible world that exhibits
the behaviour under test. That keeps assertions about *mechanisms* — "opposite
recorded reactions produce opposite verdicts" — instead of assertions about
particular rows, which is what made the original self-checks so brittle.
"""

from __future__ import annotations

import json

import pytest

from attention_router.loader import Dataset


def message(**overrides) -> dict:
    """A message row with every column present, so nothing KeyErrors."""
    row = {
        "message_id": "m_001",
        "user_id": "u_001",
        "conversation_type": "personal",
        "group_id": "",
        "business_id": "",
        "sender_user_id": "u_100",
        "created_at": "2026-07-31 14:00",
        "message_text": "hello",
        "media_type": "",
        "media_id": "",
        "forwarded_count": "0",
    }
    row.update({k: str(v) for k, v in overrides.items()})
    return row


def event(message_id: str, user_id: str, persona: str) -> dict:
    """A reaction row matching one of the engagement personas."""
    profiles = {
        "acts_fast": ("1", "1", "2", "0", "0", "0"),
        "reads_later": ("1", "0", "120", "0", "0", "0"),
        "ignores": ("0", "0", "", "1", "0", "0"),
        "rejects": ("0", "0", "", "1", "1", "0"),
        "reports": ("0", "0", "", "1", "1", "1"),
    }
    opened, replied, reaction, dismissed, muted, reported = profiles[persona]
    return {
        "user_id": user_id,
        "message_id": message_id,
        "message_opened": opened,
        "message_replied": replied,
        "reaction_time_minutes": reaction,
        "notification_dismissed": dismissed,
        "muted_after_message": muted,
        "message_reported": reported,
    }


@pytest.fixture
def tiny_dataset() -> Dataset:
    """Two receivers, one sender, opposite recorded reactions.

    This is the whole thesis of the system in fixture form: identical text
    from an identical sender must route differently for the two users, and
    the only thing that differs is what they did last time.
    """
    data = Dataset()
    data.users = {
        "u_engaged": {
            "user_id": "u_engaged", "do_not_disturb_window": "22:00-07:00",
            "messages_opened_30d": "40", "messages_replied_30d": "10",
            "notifications_dismissed_30d": "5", "messages_reported_30d": "0",
        },
        "u_rejecting": {
            "user_id": "u_rejecting", "do_not_disturb_window": "23:00-06:00",
            "messages_opened_30d": "12", "messages_replied_30d": "1",
            "notifications_dismissed_30d": "30", "messages_reported_30d": "3",
        },
    }
    data.businesses = {
        "biz_good": {
            "business_id": "biz_good", "brand_name": "Northline Bank", "category": "bank",
            "verified": "1", "official_domain": "northlinebank.com",
            "domain_used_by_sender": "northlinebank.com", "account_age_days": "2000",
            "messages_sent_30d": "900", "user_reports_30d": "2",
            "domain_used_by_sender_age_days": "2000",
        },
        "biz_impostor": {
            "business_id": "biz_impostor", "brand_name": "Northline Bank", "category": "bank",
            "verified": "0", "official_domain": "northlinebank.com",
            "domain_used_by_sender": "northline-secure.net", "account_age_days": "24",
            "messages_sent_30d": "3000", "user_reports_30d": "61",
            "domain_used_by_sender_age_days": "24",
        },
        "biz_shortener": {
            # The negative control: a real brand that happens to use a shortener.
            "business_id": "biz_shortener", "brand_name": "Trailhead Travel",
            "category": "travel", "verified": "1",
            "official_domain": "trailheadtravel.com", "domain_used_by_sender": "vl.gl",
            "account_age_days": "4300", "messages_sent_30d": "500",
            "user_reports_30d": "3", "domain_used_by_sender_age_days": "900",
        },
    }

    listing = "Selling a denim jacket, size M. Pickup near Gate 2 this weekend."
    for index, (user, persona) in enumerate(
        [("u_engaged", "reads_later"), ("u_rejecting", "rejects")]
    ):
        for offset in range(2):
            hid = f"h_{index}{offset}"
            data.history[hid] = {
                "message_id": hid, "user_id": user, "conversation_type": "personal",
                "group_id": "", "business_id": "", "sender_user_id": "u_seller",
                "created_at": f"2026-06-1{offset} 10:00", "message_text": listing,
                "media_type": "", "media_id": "", "forwarded_count": "0",
            }
            data.history_by_counterpart.setdefault((user, "u_seller"), []).append(
                data.history[hid]
            )
            data.history_by_user.setdefault(user, []).append(data.history[hid])
            data.events[(user, hid)] = event(hid, user, persona)

    return data


class StubClient:
    """Stands in for the Anthropic/Groq client. Records what it was asked."""

    def __init__(self, payload: dict | None = None, fail_times: int = 0):
        self.payload = payload or {
            "action": "digest",
            "message_type": "personal",
            "reason_template_id": "R11",
            "reason_override": "",
            "evidence_message_ids": [],
            "evidence_strength": "moderate",
        }
        self.fail_times = fail_times
        self.prompts: list[str] = []

    @property
    def messages(self):
        return self

    def create(self, *, model, max_tokens, system, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated transient API failure")
        return _Response(json.dumps(self.payload))


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Usage:
    input_tokens = 100
    output_tokens = 20


class _Response:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.usage = _Usage()


@pytest.fixture
def stub_client():
    return StubClient()
