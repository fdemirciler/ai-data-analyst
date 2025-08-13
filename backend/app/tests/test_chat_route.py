from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.session import session_store
from backend.data_processing_profiling.pipeline import run_processing_pipeline


def test_chat_route_basic():
    client = TestClient(app)
    root = Path(__file__).resolve().parents[3]
    sample = root / "sample_PL.csv"
    assert sample.exists(), "sample_PL.csv not found"
    result = run_processing_pipeline(str(sample), mode="schema_only")
    payload = result["payload"]
    session = session_store.create(payload=payload, data_path=sample)
    resp = client.post(
        "/api/chat",
        json={"sessionId": session.session_id, "message": "What is the dataset shape?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sessionId"] == session.session_id
    assert "completed" in data["progress"]
    assert "Dataset shape" in data["content"]
