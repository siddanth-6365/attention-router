"""Priors for senders this user has never heard from.

Retrieval is the core of this system, and it has a cold start: a first-contact
message has no history to look up, which is exactly when the router is least
able to help and most likely to be wrong. Real inboxes are full of these -
a new delivery partner, a new parent in a school group, a business messaging
for the first time.

The fix is not to guess. It is to answer a narrower question the corpus *can*
answer: "we have never seen this sender reach this user, but what do we know
about this sender generally, or about senders like them?"

Three priors, tried in order:

  1. **Cross-recipient abuse reputation** - this sender has been reported by
     other people. Restricted to abuse on purpose: see TRANSFERABLE_LEANINGS
     below for the measurement that forced that restriction.
  2. **Business category baseline** - an unknown business, but this user has a
     track record with other businesses of the same kind. This one is the
     receiver's own behaviour, so it generalises freely.
  3. **Group baseline** - a new sender inside a group this user already has a
     relationship with.

A prior is always weaker than a record, and the system says so: it is labelled
as a prior in the prompt, and it caps `evidence_strength` at `weak`, so
confidence lands at the bottom of its band. Being unsure is the correct
posture when you have never seen someone before.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from attention_router.loader import Dataset, Row, counterpart_of, to_bool, to_opt_int

# Below this many observations a prior is noise dressed as evidence.
MIN_OBSERVATIONS = 3


@dataclass
class Prior:
    """What is known about a sender absent any history with this receiver."""

    kind: str = "none"  # sender_reputation | business_category | group_baseline | none
    leaning: str = "unknown"  # engaged | mixed | ignored | rejected | risky | unknown
    basis: str = ""
    observations: int = 0

    @property
    def available(self) -> bool:
        return self.kind != "none"

    def describe(self) -> str:
        if not self.available:
            return (
                "No prior information about this sender. Route on message content "
                "and the receiver's general profile alone, and stay cautious about "
                "anything asking for money, credentials, or urgent action."
            )
        return (
            f"No direct history between this user and this sender. "
            f"Weaker fallback signal ({self.kind}, {self.observations} observations): "
            f"{self.basis} Treat this as a prior, not as this user's own behaviour."
        )


def _aggregate(data: Dataset, rows: list[tuple[str, str]]) -> tuple[str, int]:
    """Collapse (user_id, message_id) reactions into a single leaning."""
    opened = replied = dismissed = muted = reported = 0
    times: list[int] = []
    seen = 0
    for user_id, message_id in rows:
        event = data.events.get((user_id, message_id))
        if not event:
            continue
        seen += 1
        opened += to_bool(event.get("message_opened"))
        replied += to_bool(event.get("message_replied"))
        dismissed += to_bool(event.get("notification_dismissed"))
        muted += to_bool(event.get("muted_after_message"))
        reported += to_bool(event.get("message_reported"))
        reaction = to_opt_int(event.get("reaction_time_minutes"))
        if reaction is not None:
            times.append(reaction)

    if seen < MIN_OBSERVATIONS:
        return "unknown", seen

    if reported and reported / seen >= 0.2:
        return "risky", seen
    if muted and muted / seen >= 0.4:
        return "rejected", seen
    if dismissed / seen >= 0.6:
        return "ignored", seen
    if replied / seen >= 0.4 or (times and statistics.median(times) <= 15):
        return "engaged", seen
    if opened / seen >= 0.6:
        return "mixed", seen
    return "ignored", seen


_PHRASING = {
    "risky": "other recipients have reported messages from this sender",
    "rejected": "other recipients muted the conversation after messages from this sender",
    "ignored": "other recipients routinely dismiss messages from this sender",
    "engaged": "other recipients read and reply to this sender promptly",
    "mixed": "other recipients usually open messages from this sender but rarely reply",
}

_GROUP_PHRASING = {
    "risky": "this user has reported messages",
    "rejected": "this user has muted conversations after messages",
    "ignored": "this user routinely dismisses messages",
    "engaged": "this user reads and replies promptly",
    "mixed": "this user opens messages but rarely replies",
}

_CATEGORY_PHRASING = {
    "risky": "this user has reported messages from other businesses in this category",
    "rejected": "this user has muted other businesses in this category",
    "ignored": "this user routinely dismisses messages from businesses in this category",
    "engaged": "this user actively engages with other businesses in this category",
    "mixed": "this user opens but rarely replies to businesses in this category",
}


# Which cross-recipient leanings are worth borrowing.
#
# Measured on the corpus, only ~22% of personal senders provoke the same
# reaction from every recipient: whether you want your neighbour's messages is
# a fact about your relationship, not about your neighbour. Carrying a
# stranger's engagement preference over to you is therefore mostly noise, and
# an earlier version of this module that did so measurably *reduced* accuracy.
#
# Abuse is the exception. Someone running a credential-phishing script is
# doing it to everyone, so a reported sender stays reported no matter who
# receives the next message. Only that signal crosses the recipient boundary.
TRANSFERABLE_LEANINGS = frozenset({"risky"})


def sender_reputation(data: Dataset, message: Row) -> Prior:
    """How everyone else reacted to this sender, where that generalises."""
    sender = (message.get("sender_user_id") or "").strip()
    if not sender:
        return Prior()
    receiver = message["user_id"]
    rows = [
        (row["user_id"], row["message_id"])
        for row in data.history.values()
        if row.get("sender_user_id") == sender and row["user_id"] != receiver
    ]
    leaning, seen = _aggregate(data, rows)
    if leaning == "unknown" or leaning not in TRANSFERABLE_LEANINGS:
        return Prior()
    return Prior("sender_reputation", leaning, _PHRASING[leaning] + ".", seen)


def business_category(data: Dataset, message: Row) -> Prior:
    """How this user reacted to other businesses of the same kind."""
    business_id = (message.get("business_id") or "").strip()
    business = data.businesses.get(business_id)
    if not business:
        return Prior()
    category = (business.get("category") or "").strip()
    if not category:
        return Prior()

    peers = {
        other_id
        for other_id, other in data.businesses.items()
        if other.get("category") == category and other_id != business_id
    }
    receiver = message["user_id"]
    rows = [
        (row["user_id"], row["message_id"])
        for row in data.history.values()
        if row["user_id"] == receiver and row.get("business_id") in peers
    ]
    leaning, seen = _aggregate(data, rows)
    if leaning == "unknown":
        return Prior()
    # This one is the receiver's *own* behaviour toward comparable senders,
    # not a stranger's, so every leaning is worth carrying across.
    return Prior("business_category", leaning,
                 f"{_CATEGORY_PHRASING[leaning]} ({category}).", seen)


def group_baseline(data: Dataset, message: Row) -> Prior:
    """How this user treats the group a new sender is posting in."""
    group_id = (message.get("group_id") or "").strip()
    if not group_id:
        return Prior()
    receiver = message["user_id"]
    rows = [
        (receiver, row["message_id"])
        for row in data.history_by_group.get((receiver, group_id), [])
    ]
    leaning, seen = _aggregate(data, rows)
    if leaning == "unknown":
        return Prior()
    label = (data.groups.get(group_id, {}).get("group_name") or group_id)
    return Prior("group_baseline", leaning,
                 f"Across '{label}' as a whole, {_GROUP_PHRASING[leaning]}.", seen)


def compute(data: Dataset, message: Row, has_direct_history: bool = False) -> Prior:
    """Best available prior for a sender with no history with this receiver.

    Returns an empty Prior when direct history exists - a prior should never
    compete with the real thing.
    """
    if has_direct_history:
        return Prior()
    for source in (sender_reputation, business_category, group_baseline):
        prior = source(data, message)
        if prior.available:
            return prior
    return Prior()


def is_cold(data: Dataset, message: Row) -> bool:
    """True when this receiver has no history with this message's sender."""
    counterpart = counterpart_of(message)
    if not counterpart:
        return True
    return not data.history_by_counterpart.get((message["user_id"], counterpart))
