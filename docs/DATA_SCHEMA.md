# Using your own data

The router reads a directory of CSVs. Point it anywhere:

```bash
export ATTENTION_ROUTER_DATA=/path/to/your/corpus
attention-router route
```

Two tables are required. Everything else is read when present and skipped when
absent — a missing table degrades a specific signal rather than failing the
run, so you can start with what you have and add the rest later.

---

## Required

### `message_history.csv` — what was sent before

| Column | Meaning |
|---|---|
| `message_id` | unique id |
| `user_id` | the **recipient** |
| `sender_user_id` | who sent it (person-to-person) |
| `business_id` | who sent it (organisation) — use instead of `sender_user_id` |
| `conversation_type` | `personal` \| `group` \| `business` |
| `group_id` | group the message was posted in, if any |
| `created_at` | `YYYY-MM-DD HH:MM` |
| `message_text` | body; empty for voice notes |
| `media_type` | `` \| `image` \| `voice` |
| `media_id` | joins to `images.csv` / `voice_notes.csv` |
| `forwarded_count` | integer |

### `message_events.csv` — what the recipient did about it

This is the table that makes routing personal. Without it the system is just a
text classifier.

| Column | Meaning |
|---|---|
| `user_id` + `message_id` | joins to history |
| `message_opened` | `1` or `0` |
| `message_replied` | `1` or `0` |
| `reaction_time_minutes` | integer, or **blank** |
| `notification_dismissed` | `1` or `0` |
| `muted_after_message` | `1` or `0` |
| `message_reported` | `1` or `0` |

> **Leave `reaction_time_minutes` blank when there was no engagement.** Blank
> means "never reacted"; `0` means "reacted instantly". Writing `0` for a
> message nobody opened inverts the strongest signal the router has.

### The messages to route

`messages.csv` uses the same columns as `message_history.csv`. This is the
input, not context.

---

## Optional

| Table | What you lose without it |
|---|---|
| `users.csv` | quiet hours, per-user engagement rates |
| `groups.csv` | group type and size context |
| `group_members.csv` | the recipient's role, mute state, group activity |
| `business_accounts.csv` | **the brand-impersonation rule cannot fire** |
| `user_business_history.csv` | marketing-consent and opt-out signals |
| `images.csv`, `voice_notes.csv` | media understanding |
| `daily_notification_summary.csv` | notification-load context |

### `business_accounts.csv`

Worth supplying if you have organisational senders, because it drives the
highest-precision safety rule.

| Column | Meaning |
|---|---|
| `business_id`, `brand_name`, `category` | identity |
| `verified` | `1` or `0` |
| `official_domain` | the domain the brand actually owns |
| `domain_used_by_sender` | the domain this account sends from |
| `account_age_days`, `user_reports_30d` | trust signals |

The impersonation rule fires on `verified == 0` **and** `official_domain !=
domain_used_by_sender`. It is a conjunction on purpose: legitimate brands do
send through link shorteners, and a single-signal rule flags them.

### `users.csv`

`do_not_disturb_window` must be `HH:MM-HH:MM`. Windows crossing midnight
(`22:00-07:00`) are handled.

---

## Conventions

- **Booleans are `1` / `0`.** `true`, `yes`, `TRUE` all read as false.
- **Timestamps** are `YYYY-MM-DD HH:MM`, or `YYYY-MM-DD` for dates. ISO-8601
  with `T` or a timezone offset will not parse.
- **`conversation_type` must agree** with which of `group_id`, `business_id`,
  `sender_user_id` is populated, or the matching context section is skipped.
- If both `sender_user_id` and `business_id` are set, the **person wins**.
- Media `file_path` is relative to the corpus directory.

---

## Adapting the taxonomy

The three actions and eleven message types are defaults, not fixed. Both live
in `src/attention_router/config.py`:

```python
ACTIONS = ("notify", "digest", "mute")
MESSAGE_TYPES = ("personal", "urgent", "event", ...)
```

Change them and update two things to match:

1. `src/attention_router/prompts/router_system.md` — the definitions the model
   reads, including the "telling adjacent types apart" section, which is where
   most classification accuracy actually comes from.
2. `src/attention_router/reasons.json` — the rationale bank. Each entry lists
   the actions and types it applies to.

Confidence bands are also in `config.py`. They exist so a decision's stated
confidence reflects how much evidence supports it; if your downstream consumer
wants raw probabilities instead, replace `rationale.calibrate`.

---

## Evaluating on your own labels

Add five columns to any corpus file and point the evaluator at it:

`action`, `message_type`, `reason`, `confidence`, `evidence_message_ids`
(semicolon-separated, or `none`).

```bash
attention-router evaluate --input /path/to/labelled.csv
attention-router evaluate --input /path/to/labelled.csv --no-evidence   # ablation
```

You get action accuracy with a confusion matrix, per-class precision/recall,
evidence precision/recall, reason similarity, confidence calibration, and a
per-error dump. The evaluator calls the same entry point the CLI does, so what
you measure is what you ship.

---

## Domain-specific safety rules

`src/attention_router/safety.py` holds regex families for credential requests,
account pressure, payment lures, and link shorteners. They are tuned for
consumer messaging — if you are routing support tickets or internal comms,
that vocabulary is the first thing to replace.

The structural distinction is worth keeping whatever the domain: rules that
are effectively never wrong **block** and skip the model entirely, everything
else **flags** and passes a constraint into the prompt. Adding a blocking rule
that is merely usually right is how a safety layer starts causing the failures
it was meant to prevent.
