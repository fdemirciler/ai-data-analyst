from __future__ import annotations

import ast
from typing import List
from ..logging_utils import log_safety_check

DENYLIST_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "requests",
    "httpx",
}

DENYLIST_CALLS = {"system", "popen", "exec", "eval", "remove", "unlink", "rmtree"}


def validate_code(code: str) -> List[str]:
    """Return list of violation messages if unsafe patterns detected."""
    violations: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        violations = [f"syntax_error: {e}"]
        log_safety_check(code, violations)
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DENYLIST_IMPORTS:
                    violations.append(f"import_not_allowed:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DENYLIST_IMPORTS:
                violations.append(f"import_not_allowed:{node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DENYLIST_CALLS:
                violations.append(f"call_not_allowed:{node.func.id}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in DENYLIST_CALLS
            ):
                violations.append(f"attr_call_not_allowed:{node.func.attr}")

    log_safety_check(code, violations)
    return violations
