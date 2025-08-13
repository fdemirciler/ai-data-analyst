from __future__ import annotations

from backend.app.agent.llm import generate_analysis_code


def test_llm_fallback_without_key(monkeypatch):
    from backend.app import config as cfg

    monkeypatch.setattr(cfg.settings, "gemini_api_key", None)
    monkeypatch.setattr(cfg.settings, "enable_llm", True)
    code = generate_analysis_code(
        "Plan X", {"dataset": {"rows": 10, "columns": 2}, "columns": {}}, "Question?"
    )
    assert "Plan length" in code or "stub" in code.lower()
