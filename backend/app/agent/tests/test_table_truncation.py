import re
import json
import types
import pytest

from app.agent.output_parser import parse_exec_output
from app.agent import llm as llm_module
from app.agent import orchestrator as orch_module
from app.config import settings


def _make_json_logs(tables):
    payload = {
        "version": "1.0",
        "tables": tables,
        "stats": [],
        "messages": [],
        "errors": [],
    }
    begin = settings.json_marker_begin
    end = settings.json_marker_end
    return f"noise before...\n{begin}\n{json.dumps(payload)}\n{end}\nnoise after..."


def test_parse_exec_output_json_first_preserves_full_data():
    # JSON table with more rows/cols than typical UI caps to ensure no trimming for JSON
    cols = [f"C{i}" for i in range(6)]
    rows = [[f"R{r}C{c}" for c in range(6)] for r in range(10)]

    logs = _make_json_logs([
        {"title": "Test JSON Table", "columns": cols, "rows": rows, "n_rows": 10, "n_cols": 6}
    ]) + "\n<table><tr><td>should be ignored due to JSON-first</td></tr></table>"

    res = parse_exec_output(
        logs,
        max_tables=5,
        max_rows=2,  # even with tiny caps, JSON must keep full data when no_limit_for_json is True
        max_cols=2,
    )
    assert isinstance(res, dict)
    tables = res.get("tables")
    assert isinstance(tables, list) and len(tables) == 1
    t = tables[0]
    assert t.get("source") == "json"
    # Full data preserved
    assert t.get("columns_full") == cols
    assert t.get("rows_full") == rows
    # UI view equals full when no_limit_for_json=True
    assert t.get("columns") == cols
    assert t.get("rows") == rows
    assert t.get("truncated") is False
    # Excerpts present
    ex = res.get("excerpts", {})
    assert "head" in ex and "tail" in ex


def test_stream_summary_chunks_prefers_full_json_and_observability(monkeypatch):
    # Ensure LLM path is enabled regardless of env
    monkeypatch.setattr(settings, "enable_llm", True, raising=False)
    # Prepare structured exec_result with a JSON table
    cols = [f"C{i}" for i in range(8)]
    rows = [[f"R{r}C{c}" for c in range(8)] for r in range(20)]
    table = {
        "title": "LLM Prompt JSON Table",
        "source": "json",
        "columns_full": cols,
        "rows_full": rows,
        "n_rows": len(rows),
        "n_cols": len(cols),
        "truncated": False,
        "html": "",
    }

    exec_result = {
        "status": "ok",
        "logs": "dummy",
        "json_raw": "",  # structured provided directly
        "structured": {"tables": [table], "excerpts": {}},
    }

    # Tighten prompt caps to validate usage
    monkeypatch.setattr(settings, "summary_prompt_max_rows", 3, raising=False)
    monkeypatch.setattr(settings, "summary_prompt_max_cols", 4, raising=False)

    # Stub LLM client
    class _StubResp:
        text = "OK"

    class _StubClient:
        def generate_content(self, prompt: str):
            # prompt should contain HTML table built from full data with prompt caps
            assert "<table" in prompt
            return _StubResp()

    monkeypatch.setattr(llm_module, "_get_llm_client", lambda provider: _StubClient())

    chunks = list(
        llm_module.stream_summary_chunks(
            session_payload={},
            question="q",
            exec_result=exec_result,
            generated_code="# code",
        )
    )
    # We don't assert specific content of chunks; ensure observability was attached
    obs = exec_result.get("observability", {})
    assert "summary" in obs
    assert "tables" in obs
    assert len(obs["tables"]) == 1
    m = obs["tables"][0]
    assert m.get("source") == "json"
    assert m.get("used_full_html") is True
    assert m.get("used_raw_html") is False
    assert m.get("used_prompt_caps") == {"rows": 3, "cols": 4}


def test_orchestrator_preface_uses_ui_caps_and_badges(monkeypatch):
    # Construct a JSON table larger than UI caps
    cols = [f"C{i}" for i in range(5)]
    rows = [[f"R{r}C{c}" for c in range(5)] for r in range(7)]
    table = {
        "title": "UI Preface Table",
        "source": "json",
        "columns_full": cols,
        "rows_full": rows,
        "n_rows": len(rows),
        "n_cols": len(cols),
        "truncated": False,
        "html": "",
    }

    exec_result = {
        "status": "ok",
        "logs": "x",  # triggers LLM summary stage
        "structured": {"tables": [table], "excerpts": {}},
    }

    # UI caps: expect to see only 2 rows and 2 cols rendered in preface
    monkeypatch.setattr(settings, "summary_ui_max_rows", 2, raising=False)
    monkeypatch.setattr(settings, "summary_ui_max_cols", 2, raising=False)

    # Stub dependencies
    monkeypatch.setattr(orch_module, "generate_analysis_code", lambda plan, session_payload, question: "# code")
    monkeypatch.setattr(orch_module, "validate_code", lambda code: [])
    monkeypatch.setattr(orch_module, "execute_analysis_script", lambda code, dataset_path: exec_result)

    def _stub_stream_summary_chunks(session_payload, question, exec_result, code, **kw):
        yield "LLM ANSWER"

    # Patch the imported symbol used inside orchestrator
    import app.agent.llm as llm_in_orch
    monkeypatch.setattr(llm_in_orch, "stream_summary_chunks", _stub_stream_summary_chunks)

    res = orch_module.run_agent_response({}, "q", dataset_path=None)
    answer = res["answer"]

    # Title and metadata badge present
    assert "<strong>Title: UI Preface Table</strong>" in answer
    assert "Source: json" in answer and "Rows: 7" in answer and "Cols: 5" in answer

    # Count rows in the first table's tbody -> should be UI max rows (2)
    m = re.search(r"<table[\s\S]*?<tbody>([\s\S]*?)</tbody>", answer, flags=re.I)
    assert m, "Preface table not found"
    tbody = m.group(1)
    tr_count = len(re.findall(r"<tr[\s\S]*?>", tbody, flags=re.I))
    assert tr_count == 2


def test_parse_exec_output_without_json_falls_back_to_html():
    html = """
    <h3>Some Table</h3>
    <table>
      <thead><tr><th>A</th><th>B</th></tr></thead>
      <tbody>
        <tr><td>1</td><td>2</td></tr>
        <tr><td>3</td><td>4</td></tr>
      </tbody>
    </table>
    """
    res = parse_exec_output(html, max_tables=5, max_rows=5, max_cols=5)
    assert isinstance(res, dict)
    tables = res.get("tables")
    assert isinstance(tables, list) and len(tables) >= 1
    assert tables[0].get("source") == "html"


def test_strict_json_only_blocks_html_fallback(monkeypatch):
    # Enable strict JSON-only: without JSON block, we should not parse HTML tables
    monkeypatch.setattr(settings, "enable_json_first", True, raising=False)
    monkeypatch.setattr(settings, "strict_json_only", True, raising=False)

    html = """
    <h3>Title</h3>
    <table>
      <thead><tr><th>A</th><th>B</th></tr></thead>
      <tbody>
        <tr><td>1</td><td>2</td></tr>
      </tbody>
    </table>
    """
    res = parse_exec_output(html, max_tables=5, max_rows=5, max_cols=5)
    assert isinstance(res, dict)
    assert res.get("tables") == []
    ex = res.get("excerpts", {})
    assert "head" in ex and "tail" in ex


def test_strict_json_only_with_json_still_parses_json(monkeypatch):
    monkeypatch.setattr(settings, "enable_json_first", True, raising=False)
    monkeypatch.setattr(settings, "strict_json_only", True, raising=False)

    cols = ["A", "B"]
    rows = [["1", "2"], ["3", "4"]]
    logs = _make_json_logs([
        {"title": "J", "columns": cols, "rows": rows, "n_rows": 2, "n_cols": 2}
    ])

    res = parse_exec_output(logs, max_tables=5, max_rows=1, max_cols=1)
    tables = res.get("tables")
    assert isinstance(tables, list) and len(tables) == 1
    assert tables[0].get("source") == "json"
