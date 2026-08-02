"""Central configuration: paths, enums, model settings, calibration bands.

Every path is derived from this file's location so the pipeline runs from any
working directory on any machine. Nothing here is machine-specific.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# Package assets resolve from this file. The corpus resolves from the working
# tree, or from ATTENTION_ROUTER_DATA when the data lives elsewhere - so the
# package stays importable no matter where it is installed.

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

PROMPTS_DIR = PACKAGE_DIR / "prompts"
REASONS_JSON = PACKAGE_DIR / "reasons.json"

DATA_DIR = Path(os.getenv("ATTENTION_ROUTER_DATA", PROJECT_ROOT / "data")).resolve()
CACHE_DIR = DATA_DIR / "cache"

MESSAGES_CSV = DATA_DIR / "messages.csv"
LABELLED_CSV = DATA_DIR / "labelled.csv"
OUTPUT_CSV = DATA_DIR / "output.csv"
MEDIA_CACHE_JSON = CACHE_DIR / "media_cache.json"

# `file_path` in images.csv / voice_notes.csv is relative to DATA_DIR.
DATASET_DIR = DATA_DIR  # retained: media paths resolve against it

# --- Output contract ---------------------------------------------------------
# One row per input message, these columns, in this order.

OUTPUT_HEADER = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

ACTIONS = ("notify", "digest", "mute")

MESSAGE_TYPES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

NO_EVIDENCE = "none"

# --- Confidence calibration --------------------------------------------------
# The labelled samples place confidence on a 0.02 grid whose band is determined
# by the action. We reproduce that structure rather than letting the model
# invent a float: position within the band is driven by evidence strength.

CONFIDENCE_BANDS: dict[str, tuple[float, ...]] = {
    "notify": (0.85, 0.87, 0.89, 0.91),
    "mute": (0.81, 0.83, 0.85, 0.87),
    "digest": (0.78, 0.80, 0.82, 0.84),
}

EVIDENCE_STRENGTH_LEVELS = ("none", "weak", "moderate", "strong")

# --- Models ------------------------------------------------------------------

# The router is pluggable across providers so the same pipeline can be scored
# on several models (see the model-comparison loop in code/README.md). Groq's hosted open
# models are roughly an order of magnitude cheaper per token than a frontier
# model for what is, at bottom, an eleven-way classification over evidence we
# have already assembled for it.
ROUTER_PROVIDER = os.getenv("ROUTER_PROVIDER", "anthropic")  # anthropic | groq

ROUTER_MODELS = {
    "groq": os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b"),
    "anthropic": os.getenv("ANTHROPIC_ROUTER_MODEL", "claude-sonnet-5"),
}
ROUTER_MODEL = ROUTER_MODELS[ROUTER_PROVIDER]

GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Published list prices in USD per million tokens, for the run-cost estimate
# printed at the end of a run. Approximate and for orientation only - bill
# against your provider dashboard, not this table.
PRICE_PER_MTOK = {
    "claude-sonnet-5": (2.00, 10.00),   # introductory rate through 2026-08-31
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "qwen/qwen3.6-27b": (0.29, 0.59),
}


def price_per_mtok(model: str) -> tuple[float, float]:
    """(input, output) USD per million tokens; zeros when the model is unlisted."""
    return PRICE_PER_MTOK.get(model, (0.0, 0.0))

# Vision always runs on Anthropic: it needs real multimodal understanding, and
# it is a one-time cost because the results are committed to the media cache.
# A populated cache means this model is never called again.
VISION_MODEL = os.getenv("VISION_MODEL", "claude-sonnet-5")

# Routing is a bounded judgement over evidence we have already assembled, not
# open-ended reasoning, so thinking is off and effort is capped. Vision is a
# straight extraction task and gets the same treatment.
ROUTER_THINKING = {"type": "disabled"}
ROUTER_EFFORT = os.getenv("ROUTER_EFFORT", "medium")

ROUTER_MAX_TOKENS = 700
# Some posters in this corpus are dense multi-paragraph documents; 900 tokens
# truncated their JSON mid-transcription. The prompt also caps how much body
# text to transcribe, so this ceiling is headroom rather than a target.
VISION_MAX_TOKENS = 2500
MAX_ROUTER_ATTEMPTS = 3  # 1 initial + 2 repair retries

# Samples per message; the majority (action, message_type) wins. Measured on
# the labelled set, 3 votes cost 3x and did not improve accuracy over 1, so
# the default is 1. Raise it only if you measure a gain on your own eval.
ROUTER_VOTES = int(os.getenv("ROUTER_VOTES", "1"))

# NOTE ON DETERMINISM: `temperature` is rejected outright by Sonnet 5 and
# Opus 5, so it cannot be pinned to 0. Everything the pipeline controls is
# still deterministic - retrieval ordering, safety rules, reason selection,
# confidence calibration, and the committed media cache - but the routing
# call itself is not bit-reproducible. Sampling at temperature 0 never
# guaranteed identical output on earlier models either.

# Concurrency for the per-message routing calls. Groq's hosted tiers rate
# limit far more aggressively than the Anthropic API, and a 429 storm is what
# silently pushed a whole run onto the heuristic fallback, so the default is
# provider-aware. Override with ROUTER_WORKERS.
_DEFAULT_WORKERS = {"groq": 3, "anthropic": 8}
MAX_WORKERS = int(
    os.getenv("ROUTER_WORKERS", str(_DEFAULT_WORKERS.get(ROUTER_PROVIDER, 4)))
)

# --- Retrieval ---------------------------------------------------------------
# Ground truth cites one or two historical IDs, never more.
MAX_EVIDENCE_IDS = 2

# BM25 parameters (Robertson/Sparck-Jones defaults).
BM25_K1 = 1.5
BM25_B = 0.75

# --- Secrets -----------------------------------------------------------------
# Read from the environment only; never hardcoded, never logged.

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ASR_API_KEY_ENVS = ("GROQ_API_KEY", "OPENAI_API_KEY")

ASR_ENDPOINTS = {
    "GROQ_API_KEY": "https://api.groq.com/openai/v1/audio/transcriptions",
    "OPENAI_API_KEY": "https://api.openai.com/v1/audio/transcriptions",
}
ASR_MODELS = {
    "GROQ_API_KEY": "whisper-large-v3",
    "OPENAI_API_KEY": "whisper-1",
}
