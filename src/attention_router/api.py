"""HTTP service around the router.

    uvicorn attention_router.api:app --reload

The batch CLI remains the primary interface — routing a whole corpus is the
common case. This exists because the interesting question for a real
deployment is per-message latency on a live stream, and because a decision
service is easier to integrate with than a CSV.

Both front ends call the same `route_messages`, so there is one routing
implementation and no chance of the two drifting apart.

The corpus and its indexes load once at startup rather than per request: the
CSVs are the reference data the router reasons over, and re-reading them on
every call would dominate the latency.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from attention_router import __version__, coldstart, config, media, rationale, safety
from attention_router.cli import load_dotenv, route_messages
from attention_router.explain import find_message, trace
from attention_router.llm import USAGE, build_client
from attention_router.loader import load_dataset
from attention_router.retriever import EvidenceRetriever

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the corpus once, not per request."""
    load_dotenv()
    STATE["data"] = load_dataset()
    STATE["cache"] = media.load_cache()
    STATE["bank"] = rationale.load_bank()
    STATE["retriever"] = EvidenceRetriever(
        STATE["data"], media_text=media.text_index(STATE["cache"])
    )
    yield
    STATE.clear()


app = FastAPI(
    title="Attention Router",
    version=__version__,
    summary="Decides which messages interrupt a user, which wait, and which are suppressed.",
    lifespan=lifespan,
)


class MessageIn(BaseModel):
    """One incoming message. Mirrors a row of the corpus."""

    message_id: str = Field(examples=["msg_001"])
    user_id: str = Field(description="the receiving user", examples=["u_032"])
    conversation_type: Literal["personal", "group", "business"] = "personal"
    group_id: str = ""
    business_id: str = ""
    sender_user_id: str = ""
    created_at: str = Field(default="", examples=["2026-07-31 14:26"])
    message_text: str = ""
    media_type: Literal["", "image", "voice"] = ""
    media_id: str = ""
    forwarded_count: int = 0


class DecisionOut(BaseModel):
    message_id: str
    action: Literal["notify", "digest", "mute"]
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: list[str]
    decided_by: str = Field(description="model, safety_block, or fallback")


@app.get("/health", summary="Liveness and corpus size")
def health() -> dict:
    data = STATE.get("data")
    return {
        "status": "ok" if data else "loading",
        "version": __version__,
        "router_model": config.ROUTER_MODEL,
        "provider": config.ROUTER_PROVIDER,
        "corpus": {
            "users": len(data.users) if data else 0,
            "history": len(data.history) if data else 0,
            "media_cached": len(STATE.get("cache", {})),
        },
        "usage_this_process": USAGE.summary(),
    }


@app.post("/route", response_model=DecisionOut, summary="Route one message")
def route_one(message: MessageIn) -> DecisionOut:
    payload = {k: str(v) for k, v in message.model_dump().items()}
    if payload["user_id"] not in STATE["data"].users:
        raise HTTPException(404, f"unknown user_id '{payload['user_id']}'")

    decision = route_messages([payload], workers=1, verbose=False)[0]
    return DecisionOut(
        message_id=decision.message_id,
        action=decision.action,
        message_type=decision.message_type,
        reason=decision.reason,
        confidence=decision.confidence,
        evidence_message_ids=decision.evidence_message_ids,
        decided_by=decision.source,
    )


@app.post("/preview", summary="Retrieval and safety only, no model call")
def preview(message: MessageIn) -> dict:
    """What the router knows before it asks the model.

    Free and instant — useful for debugging a decision, and for checking
    whether the evidence layer has anything to work with on a given sender.
    """
    payload = {k: str(v) for k, v in message.model_dump().items()}
    data = STATE["data"]
    result = STATE["retriever"].retrieve(payload)
    media_text = media.as_text(STATE["cache"].get(payload["media_id"]))
    verdict = safety.evaluate(data, payload, media_text)
    prior = coldstart.compute(data, payload, has_direct_history=bool(result.evidence))
    return {
        "retrieval": {
            "tier": result.tier,
            "evidence_message_ids": result.message_ids,
            "signal": result.signal(),
            "strength": result.strength(),
        },
        "cold_start_prior": {
            "kind": prior.kind,
            "leaning": prior.leaning,
            "basis": prior.basis,
            "observations": prior.observations,
        },
        "safety": {
            "level": verdict.level,
            "rules_fired": verdict.rules,
            "would_block": verdict.blocked,
            "forced_decision": (
                {"action": verdict.action, "message_type": verdict.message_type}
                if verdict.blocked else None
            ),
        },
    }


@app.get("/explain/{message_id}", summary="Decision trace for a corpus message")
def explain(message_id: str, live: bool = False, show_prompt: bool = False) -> dict:
    message = find_message(message_id)
    if message is None:
        raise HTTPException(404, f"no message '{message_id}' in the loaded corpus")
    client = build_client() if live else None
    return {
        "message_id": message_id,
        "trace": trace(message, STATE["data"], client=client, show_prompt=show_prompt),
    }
