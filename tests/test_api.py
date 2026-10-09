from fastapi.testclient import TestClient

from clinical_rag.api import create_app


def test_health_and_ask(pipeline) -> None:
    with TestClient(create_app(pipeline)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["retriever"] == "bm25"

        resp = client.post("/ask", json={"question": "What fall risk score is high risk?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["refused"] is False
        assert body["citations"][0]["doc_id"] == "falls-protocol"
        assert body["retrieval"][0]["rank"] == 1
        assert body["latency_ms"] >= 0


def test_ask_refusal_and_validation(pipeline) -> None:
    with TestClient(create_app(pipeline)) as client:
        resp = client.post("/ask", json={"question": "What is the cafeteria menu today?"})
        assert resp.status_code == 200
        assert resp.json()["refused"] is True
        assert resp.json()["citations"] == []

        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={"question": "valid?", "top_k": 99}).status_code == 422
