"""End-to-end API tests. Skipped automatically until `python -m fmd.train` has run."""
import pytest

from fmd import config

pytestmark = pytest.mark.skipif(not (config.MODELS_DIR / "messages_vector_store.joblib").exists(),
                                reason="train the models first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.app import app

    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_scam_is_flagged_with_evidence(client):
    r = client.post("/v1/analyze", json={
        "text": "URGENT: You have won a $2000 cash prize! Call 09061701461 or text CLAIM to 80082 now",
        "llm_provider": "offline"}).json()
    assert r["task"] == "messages" and r["label"] == "deceptive"
    assert r["evidence"] and r["assistant"]["explanation"]


def test_normal_message_skips_llm(client):
    r = client.post("/v1/analyze", json={"text": "Can you pick up milk on the way home? Thanks", "task": "messages"}).json()
    assert r["label"] == "legitimate" and "assistant" not in r


def test_claim_route_returns_score(client):
    r = client.post("/v1/analyze", json={"text": "Did the Treasury announce it will stop printing $20 bills?",
                                         "task": "claims", "llm_provider": "offline"}).json()
    assert 0 <= r["score"] <= 1 and r["model"]


def test_validation_rejects_empty(client):
    assert client.post("/v1/analyze", json={"text": ""}).status_code == 422


def test_model_info_and_monitoring(client):
    info = client.get("/v1/model-info").json()
    assert set(info) == set(config.TASKS)
    mon = client.get("/v1/monitoring").json()
    assert mon["messages"]["n_logged"] >= 1
