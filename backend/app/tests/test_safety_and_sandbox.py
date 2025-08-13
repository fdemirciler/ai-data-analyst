from __future__ import annotations

from backend.app.agent.safety import validate_code
from backend.app.agent.sandbox import execute_analysis_script


def test_validate_code_blocks_import():
    violations = validate_code("import os\nprint('hi')")
    assert any(v.startswith("import_not_allowed") for v in violations)


def test_sandbox_disabled_returns_skip(tmp_path):
    result = execute_analysis_script("print('hi')", dataset_path=tmp_path)
    assert result.get("status") in {"skipped", "error"}
    if result.get("status") == "skipped":
        assert result.get("reason") == "sandbox_disabled"
