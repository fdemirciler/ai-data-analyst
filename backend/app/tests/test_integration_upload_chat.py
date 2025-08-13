from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app import config as cfg


def test_upload_then_chat(monkeypatch):
    # Ensure LLM disabled for deterministic output
    monkeypatch.setattr(cfg.settings, "enable_llm", False)
    client = TestClient(app)
    root = Path(__file__).resolve().parents[3]
    sample = root / "sample_PL.csv"
    with sample.open("rb") as f:
        resp = client.post("/api/upload", files={"file": (sample.name, f, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    session_id = data["sessionId"]
    assert session_id

    chat_resp = client.post(
        "/api/chat",
        json={"sessionId": session_id, "message": "Give me a quick summary."},
    )
    assert chat_resp.status_code == 200
    chat = chat_resp.json()
    assert chat["sessionId"] == session_id
    assert "Dataset shape" in chat["content"]
    assert "completed" in chat["progress"]
    assert isinstance(chat.get("artifactIndex"), int)

    # Retrieve artifacts list
    art_resp = client.get(f"/api/sessions/{session_id}/artifacts")
    assert art_resp.status_code == 200
    arts = art_resp.json()
    assert arts["sessionId"] == session_id
    assert len(arts["artifacts"]) >= 1
    first = arts["artifacts"][chat["artifactIndex"]]
    assert "code" in first
