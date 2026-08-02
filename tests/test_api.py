"""HTTP surface. Uses the generated corpus and never calls a model."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from attention_router import config  # noqa: E402
from attention_router.api import app  # noqa: E402
from attention_router.loader import read_csv  # noqa: E402


@pytest.fixture(scope="module")
def client():
    if not config.MESSAGES_CSV.exists():
        pytest.skip("no corpus generated; run python synth/generate.py")
    with TestClient(app) as test_client:
        yield test_client


def a_user() -> str:
    return read_csv(config.MESSAGES_CSV)[0]["user_id"]


class TestHealth:
    def test_reports_loaded_corpus(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["corpus"]["history"] > 0


class TestPreview:
    """Retrieval and safety without spending a model call."""

    def test_returns_all_three_layers(self, client):
        body = client.post("/preview", json={
            "message_id": "probe", "user_id": a_user(),
            "conversation_type": "personal", "sender_user_id": "u_001",
            "message_text": "Are we still on for Sunday?"}).json()
        assert set(body) == {"retrieval", "cold_start_prior", "safety"}
        assert body["retrieval"]["tier"] in {"counterpart", "group", "user", "none"}

    def test_flags_a_phishing_message(self, client):
        body = client.post("/preview", json={
            "message_id": "probe", "user_id": a_user(),
            "conversation_type": "personal", "sender_user_id": "u_001",
            "message_text": ("Your account will be blocked in 2 hours. Confirm "
                             "your password and OTP now to keep access active.")}).json()
        assert body["safety"]["would_block"] is True
        assert body["safety"]["forced_decision"]["action"] == "mute"

    def test_leaves_an_advisory_alone(self, client):
        body = client.post("/preview", json={
            "message_id": "probe", "user_id": a_user(),
            "conversation_type": "personal", "sender_user_id": "u_001",
            "message_text": ("Safety advisory: we never ask for your OTP or card "
                             "PIN over a call.")}).json()
        assert body["safety"]["would_block"] is False


class TestValidation:
    def test_unknown_user_is_rejected(self, client):
        response = client.post("/route", json={
            "message_id": "x", "user_id": "u_does_not_exist",
            "conversation_type": "personal", "message_text": "hi"})
        assert response.status_code == 404

    def test_bad_enum_is_rejected_before_routing(self, client):
        response = client.post("/preview", json={
            "message_id": "x", "user_id": a_user(),
            "conversation_type": "telepathy", "message_text": "hi"})
        assert response.status_code == 422

    def test_missing_message_returns_404(self, client):
        assert client.get("/explain/not_a_real_id").status_code == 404


class TestExplain:
    def test_returns_a_trace_for_a_corpus_message(self, client):
        message_id = read_csv(config.MESSAGES_CSV)[0]["message_id"]
        body = client.get(f"/explain/{message_id}").json()
        assert "EVIDENCE RETRIEVAL" in body["trace"]
        assert "SAFETY" in body["trace"]
