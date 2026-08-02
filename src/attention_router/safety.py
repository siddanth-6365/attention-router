"""Deterministic safety layer.

Two jobs, both done before the model is asked anything:

1. Decide risk with rules where rules are more reliable than a model - brand
   impersonation is a table lookup, not a judgement call.
2. Neutralise message text that tries to give the router instructions.

Only rules with essentially perfect precision `block` (short-circuit the LLM
entirely). Everything else `flag`s, passing a constraint into the prompt and
letting the model weigh it against the rest of the context. That split matters:
the corpus contains legitimate brand safety advisories whose text is
"we never ask for OTP, card PIN, or payment details" - keyword matching alone
would route a genuine warning about scams as a scam.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from attention_router.loader import Dataset, Row, to_bool, to_int

# --- Patterns ----------------------------------------------------------------

# Attempts to steer the router itself. The corpus deliberately contains these.
INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+"
    r"(?:instruction|rule|routing|prompt|direction)",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above|your)\s+\w+",
    r"mark\s+this\s+(?:message\s+)?as\s+(?:notify|urgent|important|high)",
    r"(?:you\s+are|act\s+as)\s+(?:now\s+)?an?\s+\w+\s+(?:assistant|model|agent)",
    r"system\s+(?:prompt|message|override)",
    r"do\s+not\s+(?:mute|filter|block)\s+this",
)

# An actual *request* for a secret, not a mention of one.
CREDENTIAL_REQUEST_PATTERNS = (
    r"(?:reply|respond|send|share|give|tell|provide|forward)\s+(?:me\s+|us\s+|with\s+)?"
    r"(?:the\s+|your\s+|a\s+)?(?:\d+\s+digit\s+)?(?:otp|pin|cvv|password|code|passcode)",
    r"(?:confirm|verify|enter|submit|update)\s+(?:your\s+|the\s+)?"
    r"(?:otp|pin|cvv|password|card\s+number|account\s+number|login\s+code|"
    r"bank\s+details|upi\s+pin)",
    r"(?:otp|verification\s+code|login\s+code)\s+(?:abhi\s+)?(?:batao|bhejo|share)",
    r"\b\d\s*digit\s+(?:login\s+|verification\s+|otp\s+)?code\b",
    r"(?:confirm|verify)\s+password\s+and\s+otp",
)

# A brand telling users it will never ask - the inverse of a phishing attempt.
CREDENTIAL_DISCLAIMER_PATTERNS = (
    r"never\s+ask",
    r"(?:do|does|will)\s+not\s+ask",
    r"beware\s+of",
    r"report\s+suspicious",
)

# Manufactured urgency about losing access, which phishing relies on.
# The optional adverb slot matters: "may be temporarily blocked" is the same
# threat as "will be blocked" and must not escape on a single word.
ACCOUNT_PRESSURE_PATTERNS = (
    r"(?:will|may|might|could|going\s+to)\s+(?:be\s+)?(?:\w+\s+)?"
    r"(?:blocked|suspended|deactivated|closed|frozen|restricted)",
    r"(?:account|access|profile|wallet|membership)\s+(?:will\s+)?(?:\w+\s+)?expire",
    r"within\s+\d+\s+(?:hour|minute|day)s?\s+(?:or|to\s+avoid|to\s+keep)",
    r"before\s+(?:access|your\s+account)\s+is\s+blocked",
    r"kyc\s+(?:is\s+)?(?:incomplete|pending|expired)",
    r"to\s+keep\s+(?:your\s+)?(?:account|access|payments?)\s+active",
)

# "Verify at some-domain.in" - authentication pushed to an arbitrary host
# rather than the app. Paired with a credential topic this is phishing's
# signature move.
OFFSITE_VERIFY_PATTERNS = (
    r"(?:verify|log\s?in|login|confirm|authenticate|update)\s+"
    r"(?:your\s+\w+\s+|now\s+|here\s+)?(?:at|on|via|through)\s+"
    r"[a-z0-9-]+(?:\.[a-z0-9-]+)+",
    r"(?:open|tap|click)\s+(?:the\s+)?link\s+and\s+(?:confirm|verify|enter)",
)

# Nouns that make a message credential-adjacent even without an imperative.
CREDENTIAL_TOPIC_PATTERNS = (
    r"\b(?:otp|one[\s-]?time[\s-]?password|cvv|upi\s+pin|card\s+pin|"
    r"login\s+code|verification\s+code|password|kyc)\b",
)

PAYMENT_LURE_PATTERNS = (
    r"(?:pay|transfer|deposit)\s+(?:a\s+)?(?:small\s+)?(?:fee|amount|charge)",
    r"(?:refund|cashback|reward|prize|lottery|lucky\s+draw)\s+(?:of\s+)?"
    r"(?:rs\.?|inr|\$|₹)?\s*[\d,]+",
    r"you\s+(?:have\s+)?won\b",
    r"claim\s+your\s+(?:prize|reward|refund|cashback|gift)",
    r"(?:scan|open)\s+(?:this\s+|the\s+)?qr\s+to\s+(?:pay|receive|claim)",
)

# Link shorteners: legitimate brands sometimes use them, so this only ever
# contributes to a flag, never to a block on its own.
SHORTENER_PATTERNS = (
    r"\b(?:bit\.ly|tinyurl|shorturl\.at|vl\.gl|weurl\.co|wame\.pro|t\.co|"
    r"rb\.gy|cutt\.ly|is\.gd)\b",
)


def _matches(patterns: tuple[str, ...], text: str) -> str | None:
    """Return the first matching pattern's matched text, if any."""
    for pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            return found.group(0)
    return None


# --- Verdict -----------------------------------------------------------------


@dataclass
class SafetyVerdict:
    """Outcome of the deterministic pass over one message."""

    level: str = "clear"  # clear | flag | block
    action: str | None = None
    message_type: str | None = None
    rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    injection_detected: bool = False

    @property
    def blocked(self) -> bool:
        return self.level == "block"

    def _record(self, rule: str, note: str) -> None:
        self.rules.append(rule)
        self.notes.append(note)

    def block(self, rule: str, note: str, message_type: str) -> None:
        self._record(rule, note)
        self.level = "block"
        self.action = "mute"
        self.message_type = message_type

    def flag(self, rule: str, note: str) -> None:
        self._record(rule, note)
        if self.level != "block":
            self.level = "flag"

    def describe(self) -> str:
        if not self.rules:
            return "No deterministic safety rule fired."
        header = (
            "HARD BLOCK - the routing decision is already fixed to "
            f"{self.action}/{self.message_type}."
            if self.blocked
            else "RISK FLAGS - weigh these against the rest of the context."
        )
        return header + "\n" + "\n".join(f"- {note}" for note in self.notes)


# --- Evaluation --------------------------------------------------------------


def _business_risk(data: Dataset, message: Row, verdict: SafetyVerdict) -> None:
    """Impersonation and low-trust-sender checks driven by the business table."""
    business_id = (message.get("business_id") or "").strip()
    business = data.businesses.get(business_id)
    if not business:
        return

    official = (business.get("official_domain") or "").strip().lower()
    used = (business.get("domain_used_by_sender") or "").strip().lower()
    verified = to_bool(business.get("verified"))
    reports = to_int(business.get("user_reports_30d"))
    age_days = to_int(business.get("account_age_days"))
    brand = business.get("brand_name", "") or "this brand"

    # Brand impersonation: an unverified account sending from a domain that is
    # not the brand's own. In this corpus that isolates a clean cluster of
    # lookalike domains (swiggy-refund.in, hdfcbank-kyc.in, chase-secure-alert
    # .com) on accounts 20-35 days old carrying 10-77 reports. Verified brands
    # that route through a shortener are deliberately excluded.
    if official and used and official != used and not verified:
        verdict.block(
            "brand_impersonation",
            f"Sender claims to be {brand} but messages from '{used}' instead of "
            f"the official '{official}', is unverified, is {age_days} days old, "
            f"and has {reports} user reports in 30 days.",
            "scam",
        )
        return

    # These two describe a low-trust bulk sender, not a fraud attempt. Say so
    # explicitly: without it, "unverified" plus "reported" reads as evidence of
    # a scam, when the usual shape is unwanted marketing.
    if not verified and reports >= 15:
        verdict.flag(
            "unverified_reported_sender",
            f"Unverified business account with {reports} user reports in 30 days "
            f"and an account age of {age_days} days. This marks the sender as "
            f"low-trust and its bulk messaging as unwanted; on its own it is a "
            f"spam signal, not evidence of fraud.",
        )

    if not official and _matches(SHORTENER_PATTERNS, used):
        verdict.flag(
            "shortener_only_sender",
            f"Sender has no official domain on record and uses the link "
            f"shortener '{used}', which is typical of bulk marketing. Treat as "
            f"a spam signal unless the content itself attempts fraud.",
        )


def _content_risk(text: str, verdict: SafetyVerdict) -> None:
    """Risk signals carried by the message text itself."""
    disclaimer = _matches(CREDENTIAL_DISCLAIMER_PATTERNS, text)
    credential = _matches(CREDENTIAL_REQUEST_PATTERNS, text)
    pressure = _matches(ACCOUNT_PRESSURE_PATTERNS, text)
    payment = _matches(PAYMENT_LURE_PATTERNS, text)

    if credential and disclaimer:
        # "we never ask for OTP" is a warning about phishing, not phishing.
        verdict.flag(
            "credential_topic_with_disclaimer",
            "Message discusses credentials but explicitly states the brand "
            "never asks for them, which reads as a genuine safety advisory.",
        )
    elif credential:
        verdict.flag(
            "credential_request",
            f"Message asks the user to hand over a secret (matched: "
            f"'{credential.strip()}').",
        )

    if pressure:
        verdict.flag(
            "account_pressure",
            f"Message manufactures urgency about losing access (matched: "
            f"'{pressure.strip()}').",
        )
    if payment:
        verdict.flag(
            "payment_lure",
            f"Message dangles a payment, refund, or prize (matched: "
            f"'{payment.strip()}').",
        )
    if _matches(SHORTENER_PATTERNS, text):
        verdict.flag("shortened_link", "Message body contains a shortened link.")

    offsite = _matches(OFFSITE_VERIFY_PATTERNS, text)
    if offsite and not disclaimer:
        verdict.flag(
            "offsite_verification",
            f"Message pushes authentication to an arbitrary web address rather "
            f"than the official app (matched: '{offsite.strip()}').",
        )


def evaluate(data: Dataset, message: Row, media_text: str = "") -> SafetyVerdict:
    """Run every deterministic rule over one message."""
    verdict = SafetyVerdict()
    text = f"{message.get('message_text') or ''}\n{media_text}".strip()

    injection = _matches(INJECTION_PATTERNS, text)
    if injection:
        verdict.injection_detected = True
        verdict.block(
            "prompt_injection",
            f"Message tries to instruct the router itself (matched: "
            f"'{injection.strip()}'). Content that attacks the routing system "
            f"is treated as hostile regardless of what it claims to be.",
            "scam",
        )

    _business_risk(data, message, verdict)
    if not verdict.blocked:
        _content_risk(text, verdict)

    # Phishing has two recognisable shapes. Both pair a credential angle with
    # manufactured urgency, which no legitimate message in this corpus does,
    # so they are decided here rather than handed to the model.
    if not verdict.blocked and not _matches(CREDENTIAL_DISCLAIMER_PATTERNS, text):
        pressure = _matches(ACCOUNT_PRESSURE_PATTERNS, text)
        if pressure and _matches(CREDENTIAL_REQUEST_PATTERNS, text):
            verdict.block(
                "credential_phishing",
                "Message both demands a secret and threatens loss of access, "
                "the defining shape of a phishing attempt.",
                "scam",
            )
        elif (
            pressure
            and _matches(OFFSITE_VERIFY_PATTERNS, text)
            and _matches(CREDENTIAL_TOPIC_PATTERNS, text)
        ):
            verdict.block(
                "offsite_credential_phishing",
                "Message raises a credential problem, sends the user to an "
                "arbitrary web address to resolve it, and threatens loss of "
                "access if they do not.",
                "scam",
            )

    return verdict


def wrap_untrusted(text: str) -> str:
    """Fence message content so the model reads it as data, never as orders."""
    body = (text or "").strip() or "(no text content)"
    return (
        "<<<UNTRUSTED_MESSAGE_CONTENT\n"
        f"{body}\n"
        ">>>END_UNTRUSTED_MESSAGE_CONTENT"
    )
