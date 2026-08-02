"""Evidence retrieval: find the historical messages that explain this user's
likely reaction, and summarise how they actually reacted.

This is the component that makes routing personalised. Two users can receive
byte-identical text from the same sender and need opposite actions; the only
thing that separates them is what they did with similar messages before. So we
retrieve prior messages from the *same conversation partner* and join each one
to its `message_events` row.

Retrieval is tiered by how much a candidate can tell us about this exact
relationship:

    1. same receiver + same counterpart (sender or business)  - strongest
    2. same receiver + same group                             - context only
    3. same receiver, any conversation                        - weakest

Within a tier, candidates are ranked by BM25 over message text. Document
frequencies are computed over the full 412-row history rather than the
candidate pool, because pools are frequently 1-2 documents and pool-local IDF
degenerates (a term present in every document scores zero or negative).
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

from attention_router import config
from attention_router.loader import Dataset, Row, counterpart_of, to_bool, to_opt_int

# Words carrying no discriminative weight in this corpus of short chat messages.
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have i if in is it its of on or
    that the this to was were will with you your we us our not no so please pls
    can could would should just now today do does did been being there here""".split()
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall((text or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


@dataclass
class Evidence:
    """One retrieved historical message plus how the user reacted to it."""

    message_id: str
    text: str
    created_at: str
    tier: str
    score: float
    opened: bool = False
    replied: bool = False
    dismissed: bool = False
    muted_after: bool = False
    reported: bool = False
    reaction_time_minutes: int | None = None
    has_event: bool = False

    def describe(self) -> str:
        """One line for the prompt: what was sent, and what the user did."""
        if not self.has_event:
            return f"[{self.message_id}] no recorded reaction"
        reactions = []
        if self.opened:
            reactions.append("opened")
        if self.replied:
            reactions.append("replied")
        if self.dismissed:
            reactions.append("dismissed the notification")
        if self.muted_after:
            reactions.append("MUTED the conversation afterwards")
        if self.reported:
            reactions.append("REPORTED it")
        if not reactions:
            reactions.append("ignored it entirely")
        if self.reaction_time_minutes is not None:
            reactions.append(f"reacted in {self.reaction_time_minutes} min")
        return f"[{self.message_id}] {', '.join(reactions)}"


@dataclass
class RetrievalResult:
    evidence: list[Evidence] = field(default_factory=list)
    tier: str = "none"

    @property
    def message_ids(self) -> list[str]:
        return [e.message_id for e in self.evidence]

    def signal(self) -> dict:
        """Aggregate the reactions into the behavioural signal the router uses."""
        scored = [e for e in self.evidence if e.has_event]
        if not scored:
            return {"verdict": "no_signal", "evidence_count": len(self.evidence)}

        opened = sum(e.opened for e in scored)
        replied = sum(e.replied for e in scored)
        dismissed = sum(e.dismissed for e in scored)
        muted = sum(e.muted_after for e in scored)
        reported = sum(e.reported for e in scored)
        times = [
            e.reaction_time_minutes
            for e in scored
            if e.reaction_time_minutes is not None
        ]
        median_reaction = statistics.median(times) if times else None

        if reported:
            verdict = "previously_reported"
        elif muted:
            verdict = "actively_rejected"
        elif dismissed and not opened:
            verdict = "consistently_dismissed"
        elif replied and median_reaction is not None and median_reaction <= 15:
            verdict = "urgent_engagement"
        elif replied:
            verdict = "engaged"
        elif opened and median_reaction is not None and median_reaction <= 60:
            verdict = "reads_promptly"
        elif opened:
            verdict = "reads_eventually"
        else:
            verdict = "ignored"

        return {
            "verdict": verdict,
            "evidence_count": len(scored),
            "opened": opened,
            "replied": replied,
            "dismissed": dismissed,
            "muted_after": muted,
            "reported": reported,
            "median_reaction_time_minutes": median_reaction,
        }

    def strength(self) -> str:
        """How much the evidence should move confidence within its band."""
        if not self.evidence:
            return "none"
        signal = self.signal()
        if signal["verdict"] == "no_signal":
            return "weak"
        if self.tier != "counterpart":
            return "weak" if self.tier == "user" else "moderate"
        decisive = {
            "previously_reported",
            "actively_rejected",
            "urgent_engagement",
            "consistently_dismissed",
        }
        if signal["verdict"] in decisive:
            return "strong"
        return "moderate"


class EvidenceRetriever:
    """BM25 retrieval over message history, tiered by relationship closeness."""

    def __init__(self, data: Dataset, media_text: dict[str, str] | None = None):
        self.data = data
        self.media_text = media_text or {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_freq: Counter[str] = Counter()

        for message_id, row in data.history.items():
            tokens = tokenize(self._text_of(row))
            self._doc_tokens[message_id] = tokens
            self._doc_freq.update(set(tokens))

        lengths = [len(t) for t in self._doc_tokens.values()]
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0
        self._num_docs = max(len(self._doc_tokens), 1)

    def _text_of(self, row: Row) -> str:
        """Message text plus any transcribed/described media attached to it."""
        text = (row.get("message_text") or "").strip()
        media_id = (row.get("media_id") or "").strip()
        extra = self.media_text.get(media_id, "") if media_id else ""
        return f"{text}\n{extra}".strip()

    def _idf(self, term: str) -> float:
        # Smoothed Robertson/Sparck-Jones IDF; always positive.
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (self._num_docs - df + 0.5) / (df + 0.5))

    def _bm25(self, query_tokens: list[str], message_id: str) -> float:
        doc = self._doc_tokens.get(message_id, [])
        if not doc or not query_tokens:
            return 0.0
        counts = Counter(doc)
        norm = config.BM25_K1 * (
            1 - config.BM25_B + config.BM25_B * len(doc) / self._avg_len
        )
        score = 0.0
        for term in set(query_tokens):
            tf = counts.get(term, 0)
            if tf:
                score += self._idf(term) * (tf * (config.BM25_K1 + 1)) / (tf + norm)
        return score

    def _candidate_tiers(self, message: Row) -> list[tuple[str, list[Row]]]:
        user_id = message["user_id"]
        counterpart = counterpart_of(message)
        group_id = (message.get("group_id") or "").strip()

        tiers: list[tuple[str, list[Row]]] = []
        if counterpart:
            rows = self.data.history_by_counterpart.get((user_id, counterpart), [])
            if rows:
                tiers.append(("counterpart", rows))
        if group_id:
            rows = self.data.history_by_group.get((user_id, group_id), [])
            if rows:
                tiers.append(("group", rows))
        rows = self.data.history_by_user.get(user_id, [])
        if rows:
            tiers.append(("user", rows))
        return tiers

    def _to_evidence(self, message: Row, row: Row, tier: str, score: float) -> Evidence:
        event = self.data.events.get((message["user_id"], row["message_id"]))
        evidence = Evidence(
            message_id=row["message_id"],
            text=self._text_of(row),
            created_at=row.get("created_at", ""),
            tier=tier,
            score=round(score, 3),
        )
        if event:
            evidence.has_event = True
            evidence.opened = to_bool(event.get("message_opened"))
            evidence.replied = to_bool(event.get("message_replied"))
            evidence.dismissed = to_bool(event.get("notification_dismissed"))
            evidence.muted_after = to_bool(event.get("muted_after_message"))
            evidence.reported = to_bool(event.get("message_reported"))
            evidence.reaction_time_minutes = to_opt_int(
                event.get("reaction_time_minutes")
            )
        return evidence

    def retrieve(self, message: Row, limit: int | None = None) -> RetrievalResult:
        """Top-k evidence for one incoming message, best tier first."""
        limit = limit or config.MAX_EVIDENCE_IDS
        query = tokenize(self._text_of(message))
        exclude = {message.get("message_id", "")}

        for tier, rows in self._candidate_tiers(message):
            candidates = [r for r in rows if r["message_id"] not in exclude]
            if not candidates:
                continue
            # Deterministic ordering: BM25 desc, then recency desc, then id asc.
            # Done as two stable passes rather than one clever composite key.
            by_recency = sorted(
                candidates, key=lambda r: (r.get("created_at", ""), r["message_id"]),
                reverse=True,
            )
            ranked = sorted(
                by_recency, key=lambda r: -self._bm25(query, r["message_id"])
            )
            evidence = [
                self._to_evidence(message, row, tier, self._bm25(query, row["message_id"]))
                for row in ranked[:limit]
            ]
            return RetrievalResult(
                evidence=_trim_to_pattern(evidence), tier=tier
            )

        return RetrievalResult()


def _trim_to_pattern(evidence: list[Evidence]) -> list[Evidence]:
    """Cite a second message only when repetition is itself the evidence.

    The labelled data cites two IDs exactly when the point being made is "this
    keeps happening and the user keeps rejecting it", and one ID otherwise.
    Two rows that agree on a negative reaction demonstrate a pattern; two rows
    that disagree just add noise.
    """
    if len(evidence) < 2:
        return evidence
    first, second = evidence[0], evidence[1]
    rejected = [e.dismissed or e.muted_after or e.reported for e in (first, second)]
    if all(rejected):
        return [first, second]
    return [first]
