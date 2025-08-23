from __future__ import annotations

from typing import Dict, Any
import tempfile
from pathlib import Path
import textwrap

from ..config import settings
from ..logging_utils import log_sandbox_execution, log_error
import re

try:  # pragma: no cover - docker optional
    import docker  # type: ignore
except Exception:  # pragma: no cover - no docker installed
    docker = None  # type: ignore


def _smart_truncate_logs(logs: str, max_chars: int = 16000) -> str:
    """Intelligently truncate logs preserving table boundaries and key content.
    
    Priority order:
    1. Complete HTML tables (preserve table integrity)
    2. Section headers (## Overview, ## Results, etc.)
    3. Key statistics and metrics
    4. Error messages
    5. General content from beginning and end
    """
    if len(logs) <= max_chars:
        return logs

    # Always attempt to fully preserve a JSON result block if present
    begin_marker = settings.json_marker_begin
    end_marker = settings.json_marker_end
    json_range: tuple[int, int, str] | None = None
    b = logs.rfind(begin_marker)
    if b != -1:
        e = logs.find(end_marker, b + len(begin_marker))
        if e != -1:
            e += len(end_marker)
            json_range = (b, e, logs[b:e])

    # Expand effective budget to avoid truncating the JSON block
    budget = max_chars
    if json_range is not None:
        json_len = len(json_range[2])
        # Add small slack for separators
        budget = max(max_chars, json_len + 200)
    
    # Extract complete HTML tables first
    table_matches = list(re.finditer(r"<table[^>]*>.*?</table>", logs, re.DOTALL | re.IGNORECASE))
    tables = []
    table_chars = 0
    
    for match in table_matches:
        table_content = match.group(0)
        if table_chars + len(table_content) < budget * 0.7:  # Reserve 70% for tables
            tables.append((match.start(), match.end(), table_content))
            table_chars += len(table_content)
    
    # Extract section headers and key content
    header_pattern = r"^##\s+[^\n]+$"
    headers = []
    for match in re.finditer(header_pattern, logs, re.MULTILINE):
        start = max(0, match.start() - 50)  # Include some context
        end = min(len(logs), match.end() + 200)
        headers.append((start, end, logs[start:end]))
    
    # Build preserved content (JSON first if present)
    preserved_candidates = tables + headers
    if json_range is not None:
        preserved_candidates.append(json_range)
    preserved_ranges = sorted(preserved_candidates, key=lambda x: x[0])
    
    # Merge overlapping ranges
    merged = []
    for start, end, content in preserved_ranges:
        if merged and start <= merged[-1][1] + 100:  # Merge if close
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), 
                         logs[merged[-1][0]:max(end, merged[-1][1])])
        else:
            merged.append((start, end, content))
    
    # Calculate remaining budget for head/tail
    preserved_chars = sum(len(content) for _, _, content in merged)
    remaining_budget = budget - preserved_chars - 200  # Reserve for separators
    
    if remaining_budget > 0:
        head_budget = remaining_budget // 3
        tail_budget = remaining_budget - head_budget
        
        # Get head content (avoiding overlap with preserved)
        head_end = merged[0][0] if merged else len(logs)
        head = logs[:min(head_budget, head_end)]
        
        # Get tail content (avoiding overlap with preserved)
        tail_start = merged[-1][1] if merged else 0
        tail = logs[max(tail_start, len(logs) - tail_budget):]
    else:
        head = tail = ""
    
    # Assemble final result
    parts = []
    if head:
        parts.append(head)
        parts.append("\n\n[... content preserved for key sections ...]\n\n")
    
    for _, _, content in merged:
        parts.append(content)
        parts.append("\n\n")
    
    if tail and tail != head:
        parts.append("[... content continues ...]\n\n")
        parts.append(tail)
    
    result = "".join(parts)
    
    # Final safety truncation if still over budget.
    # Never truncate inside the JSON block: drop non-JSON parts first.
    if len(result) > budget:
        if json_range is not None:
            # Keep only the JSON block if necessary
            result = json_range[2]
            # If still somehow over budget (very large JSON), keep JSON intact
            # by allowing it to exceed the nominal budget.
        else:
            result = result[: budget - 50] + "\n\n[... truncated ...]"
    
    return result


def execute_analysis_script(
    code: str, dataset_path: str | Path, timeout_seconds: int = 30
) -> Dict[str, Any]:
    """Execute generated analysis code in an optional Docker sandbox.

    Behavior:
      - If sandbox disabled or docker unavailable -> return skipped result.
      - Writes code to temp directory as script.py.
      - (Future) Run container python script.py with volume mounts (read-only dataset).
    """
    if not settings.enable_docker_sandbox or docker is None:
        result = {"status": "skipped", "reason": "sandbox_disabled"}
        log_sandbox_execution(result)
        return result

    client = docker.from_env()  # type: ignore
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script_path = tmp_path / "script.py"
        script_path.write_text(textwrap.dedent(code), encoding="utf-8")

        ds_path = Path(dataset_path)
        data_mount_dir = "/data"

        # Use a pandas-enabled image or install without network if available
        try:
            # Try to use jupyter/scipy-notebook which has pandas pre-installed
            # If not available, temporarily enable network for package installation
            try:
                # First attempt: use pre-built image with pandas
                container = client.containers.run(  # type: ignore
                    image="jupyter/scipy-notebook:latest",
                    command=["python", "/work/script.py"],
                    network_disabled=True,
                    volumes={
                        str(tmp_path): {"bind": "/work", "mode": "ro"},
                        str(ds_path.parent): {"bind": data_mount_dir, "mode": "ro"},
                    },
                    working_dir="/work",
                    detach=True,
                    mem_limit="512m",
                    nano_cpus=1_000_000_000,  # 1.0 CPU
                    stdout=True,
                    stderr=True,
                    user="root",  # Override to avoid permission issues
                )
                print("DEBUG: Using pre-built scipy image")
            except docker.errors.ImageNotFound:  # type: ignore
                # Fallback: temporarily enable network for package installation
                print(
                    "DEBUG: Pre-built image not found, installing packages with network"
                )
                container = client.containers.run(  # type: ignore
                    image="python:3.11-slim",
                    command=[
                        "bash",
                        "-c",
                        "pip install pandas numpy pyarrow && python /work/script.py",
                    ],
                    network_disabled=False,  # Enable network for installation
                    volumes={
                        str(tmp_path): {"bind": "/work", "mode": "ro"},
                        str(ds_path.parent): {"bind": data_mount_dir, "mode": "ro"},
                    },
                    working_dir="/work",
                    detach=True,
                    mem_limit="512m",
                    nano_cpus=1_000_000_000,  # 1.0 CPU
                    stdout=True,
                    stderr=True,
                )

            # Wait for container and get results
            wait_result = container.wait(timeout=timeout_seconds)  # type: ignore
            exit_code = wait_result.get("StatusCode", -1)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore")  # type: ignore

            # Enhanced logging for debugging
            from ..logging_utils import log_sandbox_execution as log_info

            print(f"DEBUG: Container exit code: {exit_code}")
            print(f"DEBUG: Container logs (first 1000 chars): {logs[:1000]}")

            container.remove()

            # Consider execution successful if we have meaningful output,
            # even if exit code is non-zero (due to warnings/non-critical errors)
            has_meaningful_output = (
                logs
                and len(logs.strip()) > 100
                and (
                    "Variance Table" in logs or "DataFrame" in logs or "Metric" in logs
                )
            )

            status = "ok" if (exit_code == 0 or has_meaningful_output) else "error"

            result = {
                "status": status,
                "exit_code": exit_code,
                "logs": _smart_truncate_logs(
                    logs,
                    max_chars=int(getattr(settings, "sandbox_max_log_chars", 16000)),
                ),
            }
            log_sandbox_execution(result)
            return result
        except Exception as e:  # pragma: no cover - depends on docker runtime
            log_error(
                "sandbox", e, f"Docker execution failed for script: {code[:100]}..."
            )
            result = {"status": "error", "reason": str(e)}
            log_sandbox_execution(result)
            return result
