"""Corpus loading, indexing, and per-message context assembly.

Reads a directory of CSVs with the standard library only and exposes them as
lookup indexes. `build_dossier` joins the receiving user, the conversation, the
sender relationship, and the notification load into the personalisation packet
the router reasons over.

Only `message_history.csv` and `message_events.csv` are required - together
they are the behavioural record everything else is built on. The remaining
tables enrich the decision and are read when present, so you can point this at
your own data without first synthesising columns you do not have. See
docs/DATA_SCHEMA.md for the full contract.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time

from attention_router import config

Row = dict[str, str]


# --- Parsing helpers ---------------------------------------------------------


def to_int(value: str | None, default: int = 0) -> int:
    """Tolerant int parse: blank cells are expected throughout."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_opt_int(value: str | None) -> int | None:
    """Like `to_int` but preserves 'not recorded' as None.

    Used for `reaction_time_minutes`, where a blank means the user never
    engaged at all - which is not the same as reacting in 0 minutes.
    Collapsing it to 0 would invert the signal the router depends on.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def to_bool(value: str | None) -> bool:
    return str(value).strip() == "1"


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse the accepted timestamp formats ('%Y-%m-%d %H:%M' and date-only)."""
    text = (value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_quiet_window(window: str | None) -> tuple[time, time] | None:
    """Parse a 'HH:MM-HH:MM' do-not-disturb window."""
    text = (window or "").strip()
    if "-" not in text:
        return None
    start_text, _, end_text = text.partition("-")
    try:
        start = datetime.strptime(start_text.strip(), "%H:%M").time()
        end = datetime.strptime(end_text.strip(), "%H:%M").time()
    except ValueError:
        return None
    return start, end


def in_quiet_hours(window: str | None, moment: datetime | None) -> bool:
    """True if `moment` falls inside the user's DND window (wraps midnight)."""
    parsed = parse_quiet_window(window)
    if parsed is None or moment is None:
        return False
    start, end = parsed
    now = moment.time()
    if start <= end:
        return start <= now < end
    return now >= start or now < end  # window crosses midnight


def read_csv(path) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def counterpart_of(message: Row) -> str:
    """The other party in the conversation: a user for personal/group, else a business."""
    return (message.get("sender_user_id") or "").strip() or (
        message.get("business_id") or ""
    ).strip()


# --- Dataset -----------------------------------------------------------------


@dataclass
class Dataset:
    """All context tables, indexed for O(1) lookup during routing."""

    users: dict[str, Row] = field(default_factory=dict)
    groups: dict[str, Row] = field(default_factory=dict)
    memberships: dict[tuple[str, str], Row] = field(default_factory=dict)
    businesses: dict[str, Row] = field(default_factory=dict)
    business_relations: dict[tuple[str, str], Row] = field(default_factory=dict)

    history: dict[str, Row] = field(default_factory=dict)
    history_by_counterpart: dict[tuple[str, str], list[Row]] = field(default_factory=dict)
    history_by_group: dict[tuple[str, str], list[Row]] = field(default_factory=dict)
    history_by_user: dict[str, list[Row]] = field(default_factory=dict)

    events: dict[tuple[str, str], Row] = field(default_factory=dict)
    media_paths: dict[str, str] = field(default_factory=dict)
    notification_load: dict[str, list[Row]] = field(default_factory=dict)

    def group_name(self, group_id: str) -> str:
        return (self.groups.get(group_id) or {}).get("group_name", "")


def read_optional(path) -> list[Row]:
    """Read a table that may legitimately not exist.

    Only three tables are load-bearing: the messages being routed, the history
    they are compared against, and the reactions to that history. Everything
    else enriches the decision. Treating the rest as optional is what lets you
    point this at your own data without first inventing a business-accounts
    table you do not have.
    """
    return read_csv(path) if path.exists() else []


# Tables the router cannot do its job without.
REQUIRED_TABLES = ("message_history.csv", "message_events.csv")


def load_dataset(dataset_dir=None, *, strict: bool = True) -> Dataset:
    """Read the context tables and build the lookup indexes.

    Missing optional tables degrade specific signals rather than failing:
    without `groups.csv` there is no group context, without
    `business_accounts.csv` the impersonation rule cannot fire. Pass
    `strict=False` to tolerate missing history too, which is only useful when
    routing purely on content.
    """
    base = dataset_dir or config.DATASET_DIR
    data = Dataset()

    if strict:
        missing = [name for name in REQUIRED_TABLES if not (base / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"{base} is missing {', '.join(missing)}. These carry the behavioural "
                f"history the router personalises on. Generate a corpus with "
                f"`python synth/generate.py`, point ATTENTION_ROUTER_DATA at your own, "
                f"or pass strict=False to route on content alone."
            )

    data.users = {r["user_id"]: r for r in read_optional(base / "users.csv")}
    data.groups = {r["group_id"]: r for r in read_optional(base / "groups.csv")}
    data.memberships = {
        (r["group_id"], r["user_id"]): r for r in read_optional(base / "group_members.csv")
    }
    data.businesses = {
        r["business_id"]: r for r in read_optional(base / "business_accounts.csv")
    }
    data.business_relations = {
        (r["user_id"], r["business_id"]): r
        for r in read_optional(base / "user_business_history.csv")
    }

    by_counterpart: dict[tuple[str, str], list[Row]] = defaultdict(list)
    by_group: dict[tuple[str, str], list[Row]] = defaultdict(list)
    by_user: dict[str, list[Row]] = defaultdict(list)
    for row in read_optional(base / "message_history.csv"):
        data.history[row["message_id"]] = row
        user_id = row["user_id"]
        by_user[user_id].append(row)
        counterpart = counterpart_of(row)
        if counterpart:
            by_counterpart[(user_id, counterpart)].append(row)
        group_id = (row.get("group_id") or "").strip()
        if group_id:
            by_group[(user_id, group_id)].append(row)
    data.history_by_counterpart = dict(by_counterpart)
    data.history_by_group = dict(by_group)
    data.history_by_user = dict(by_user)

    data.events = {
        (r["user_id"], r["message_id"]): r for r in read_optional(base / "message_events.csv")
    }

    for row in read_optional(base / "images.csv"):
        data.media_paths[row["image_id"]] = row["file_path"]
    for row in read_optional(base / "voice_notes.csv"):
        data.media_paths[row["voice_note_id"]] = row["file_path"]

    load: dict[str, list[Row]] = defaultdict(list)
    for row in read_optional(base / "daily_notification_summary.csv"):
        load[row["user_id"]].append(row)
    data.notification_load = dict(load)

    return data


# --- Dossier -----------------------------------------------------------------
# Each section is emitted only when it applies, so the prompt never carries
# empty scaffolding for a conversation type that has no such context.


def _receiver_profile(data: Dataset, message: Row) -> dict:
    user = data.users.get(message["user_id"], {})
    opened = to_int(user.get("messages_opened_30d"))
    dismissed = to_int(user.get("notifications_dismissed_30d"))
    total = opened + dismissed
    sent_at = parse_timestamp(message.get("created_at"))
    return {
        "user_id": message["user_id"],
        "messages_opened_30d": opened,
        "messages_replied_30d": to_int(user.get("messages_replied_30d")),
        "notifications_dismissed_30d": dismissed,
        "messages_reported_30d": to_int(user.get("messages_reported_30d")),
        "dismissal_rate": round(dismissed / total, 2) if total else None,
        "do_not_disturb_window": user.get("do_not_disturb_window", ""),
        "message_arrives_in_quiet_hours": in_quiet_hours(
            user.get("do_not_disturb_window"), sent_at
        ),
    }


def _notification_load(data: Dataset, user_id: str) -> dict:
    days = data.notification_load.get(user_id, [])
    if not days:
        return {}
    recent = sorted(days, key=lambda r: r["date"])[-14:]
    sent = sum(to_int(r["notifications_sent"]) for r in recent)
    dismissed = sum(to_int(r["notifications_dismissed"]) for r in recent)
    return {
        "avg_notifications_per_day_14d": round(sent / len(recent), 1),
        "dismissal_rate_14d": round(dismissed / sent, 2) if sent else None,
    }


def _group_context(data: Dataset, message: Row) -> dict:
    group_id = (message.get("group_id") or "").strip()
    group = data.groups.get(group_id)
    if not group:
        return {}
    membership = data.memberships.get((group_id, message["user_id"]), {})
    sender_membership = data.memberships.get(
        (group_id, (message.get("sender_user_id") or "").strip()), {}
    )
    read = to_int(membership.get("messages_read_30d"))
    dismissed = to_int(membership.get("notifications_dismissed_30d"))
    return {
        "group_name": group.get("group_name", ""),
        "group_type": group.get("group_type", ""),
        "member_count": to_int(group.get("member_count")),
        "messages_30d": to_int(group.get("messages_30d")),
        "receiver_role": membership.get("role", "not_a_listed_member"),
        "receiver_muted_this_group": to_bool(membership.get("group_muted_by_user")),
        "receiver_messages_read_30d": read,
        "receiver_replies_sent_30d": to_int(membership.get("replies_sent_30d")),
        "receiver_notifications_dismissed_30d": dismissed,
        "receiver_engagement_in_group": (
            round(read / (read + dismissed), 2) if (read + dismissed) else None
        ),
        "sender_role_in_group": sender_membership.get("role", "not_a_listed_member"),
    }


def _business_context(data: Dataset, message: Row) -> dict:
    business_id = (message.get("business_id") or "").strip()
    business = data.businesses.get(business_id)
    if not business:
        return {}
    official = (business.get("official_domain") or "").strip()
    used = (business.get("domain_used_by_sender") or "").strip()
    relation = data.business_relations.get((message["user_id"], business_id))

    context = {
        "business_id": business_id,
        "brand_name": business.get("brand_name", ""),
        "category": business.get("category", ""),
        "verified": to_bool(business.get("verified")),
        "account_age_days": to_int(business.get("account_age_days")),
        "user_reports_30d": to_int(business.get("user_reports_30d")),
        "official_domain": official or "(none on record)",
        "domain_used_by_sender": used or "(none on record)",
        "domain_matches_official": bool(official and used and official == used),
        "sender_domain_age_days": to_int(business.get("domain_used_by_sender_age_days")),
    }

    if relation is None:
        context["user_relationship"] = "no prior relationship with this business"
        return context

    opened = to_int(relation.get("messages_opened_30d"))
    dismissed = to_int(relation.get("messages_dismissed_30d"))
    context.update(
        {
            "why_user_knows_account": relation.get("why_user_knows_account", ""),
            "last_activity_at": relation.get("last_activity_at", ""),
            "allows_promotions": to_bool(relation.get("allows_promotions")),
            "promotions_opted_out_at": relation.get("promotions_opted_out_at", "")
            or "(never opted out)",
            "activity_count_180d": to_int(relation.get("activity_count_180d")),
            "messages_opened_30d": opened,
            "messages_dismissed_30d": dismissed,
            "messages_replied_30d": to_int(relation.get("messages_replied_30d")),
            "open_rate_30d": (
                round(opened / (opened + dismissed), 2) if (opened + dismissed) else None
            ),
        }
    )
    return context


def _sender_context(data: Dataset, message: Row) -> dict:
    sender_id = (message.get("sender_user_id") or "").strip()
    if not sender_id:
        return {}
    prior = data.history_by_counterpart.get((message["user_id"], sender_id), [])
    context = {
        "sender_user_id": sender_id,
        "prior_messages_from_sender": len(prior),
        "first_contact": not prior,
    }
    if not prior:
        return context

    opened = replied = dismissed = muted = reported = 0
    for row in prior:
        event = data.events.get((message["user_id"], row["message_id"]))
        if not event:
            continue
        opened += to_bool(event.get("message_opened"))
        replied += to_bool(event.get("message_replied"))
        dismissed += to_bool(event.get("notification_dismissed"))
        muted += to_bool(event.get("muted_after_message"))
        reported += to_bool(event.get("message_reported"))
    context.update(
        {
            "historically_opened": opened,
            "historically_replied": replied,
            "historically_dismissed": dismissed,
            "historically_muted_after": muted,
            "historically_reported": reported,
        }
    )
    return context


def build_dossier(data: Dataset, message: Row) -> dict:
    """Assemble the personalisation packet for one incoming message."""
    dossier = {
        "receiver": _receiver_profile(data, message),
        "notification_load": _notification_load(data, message["user_id"]),
        "conversation_type": message.get("conversation_type", ""),
        "forwarded_count": to_int(message.get("forwarded_count")),
    }
    for key, section in (
        ("group", _group_context(data, message)),
        ("business", _business_context(data, message)),
        ("sender", _sender_context(data, message)),
    ):
        if section:
            dossier[key] = section
    return dossier
