from __future__ import annotations

from typing import Dict, Any, List, Tuple
import math
import re

from .safety import validate_code
from .sandbox import execute_analysis_script
from .llm import generate_analysis_code
from .response_sanitizer import sanitize_answer_html_tables
from ..config import settings
from .output_parser import _build_html_table
from ..logging_utils import (
    log_orchestrator_stage,
    log_final_decision,
    log_config_state,
    log_error,
)


def _generate_code_stub(plan: str) -> str:  # legacy fallback
    return generate_analysis_code(plan, {}, "")


def run_agent_response(
    session_payload: Dict[str, Any], user_message: str, dataset_path: str | None
) -> Dict[str, Any]:
    """Run the staged agent pipeline and return rich artifact.

    Returns dict with keys:
      answer: str final assistant text
      progress: list[str] stage markers
      code: str generated analysis code (may be stub)
      violations: list[str] safety violations (empty if none)
      exec: execution result dict (status, logs, etc.)
    """
    log_config_state()

    progress: List[str] = []

    # Planning stage
    log_orchestrator_stage("planning")
    progress.append("planning")
    plan = f"Answer the question: {user_message} using dataset metadata only."

    # Code generation stage
    log_orchestrator_stage("code_gen", f"Plan: {plan[:100]}...")
    progress.append("code_gen")
    code = generate_analysis_code(plan, session_payload, user_message)
    log_orchestrator_stage("code_gen_result", f"Generated {len(code)} chars of code")

    # Safety check stage
    log_orchestrator_stage("safety_check")
    progress.append("safety_check")
    violations = validate_code(code)

    if violations:
        log_orchestrator_stage("safety_blocked", f"Violations: {violations}")
        progress.append("safety_blocked")
        heuristic = run_basic_analysis(session_payload, user_message)
        answer = (
            heuristic + "\n\n(Script blocked by safety: " + "; ".join(violations) + ")"
        )
        progress.append("completed")
        log_final_decision("safety_blocked_heuristic", len(answer))
        return {
            "answer": answer,
            "progress": progress,
            "code": code,
            "violations": violations,
            "exec": {"status": "blocked"},
        }

    log_orchestrator_stage("safety_passed")

    # Sandbox execution stage
    log_orchestrator_stage("sandbox_exec", f"Dataset path: {dataset_path}")
    progress.append("sandbox_exec")
    exec_result = execute_analysis_script(
        code, dataset_path=dataset_path or "<in-memory>"
    )
    log_orchestrator_stage(
        "sandbox_result",
        f"Status: {exec_result.get('status')}, Logs: {len(exec_result.get('logs', ''))}",
    )

    # Summarization stage
    log_orchestrator_stage("summarize")
    progress.append("summarize")

    # If execution was successful and produced output, use LLM to summarize results
    # Otherwise fall back to heuristic analysis
    if exec_result.get("status") == "ok" and exec_result.get("logs"):
        log_orchestrator_stage(
            "llm_summary_attempt", "Execution successful, trying LLM summary"
        )
        try:
            from .llm import stream_summary_chunks

            # Collect LLM chunks into a single response
            llm_chunks = []
            max_answer_chars = int(getattr(settings, "summary_max_answer_chars", 0) or 0)
            for chunk in stream_summary_chunks(
                session_payload, user_message, exec_result, code
            ):
                llm_chunks.append(chunk)
                if max_answer_chars > 0 and len("".join(llm_chunks)) > max_answer_chars:
                    break

            if llm_chunks:
                answer = "".join(llm_chunks)
                # Prepend tables (prefer raw HTML when available) so they render in the UI
                structured = exec_result.get("structured") or {}
                tables = (structured.get("tables") if isinstance(structured, dict) else None) or []
                if tables:
                    mt = int(getattr(settings, "summary_max_tables", 5))
                    mr = int(getattr(settings, "summary_max_rows", 50))
                    mc = int(getattr(settings, "summary_max_cols", 15))
                    html_parts: List[str] = []

                    def _html_row_count(s: str) -> int:
                        if not isinstance(s, str) or not s.strip():
                            return 0
                        m = re.search(r"<tbody[^>]*>(.*?)</tbody>", s, re.I | re.S)
                        segment = m.group(1) if m else s
                        return len(re.findall(r"<tr[^>]*>", segment, re.I))

                    for t in tables[:mt]:
                        title = (t.get("title") or "").strip()
                        if title:
                            html_parts.append(f"<p><strong>Title: {title}</strong></p>")
                        raw_html = t.get("html")
                        cols = t.get("columns") or []
                        rows = t.get("rows") or []
                        # Prefer raw html unless it appears truncated vs reported rows
                        try:
                            n_rows = int(t.get("n_rows") or (len(rows) if isinstance(rows, list) else 0))
                        except Exception:
                            n_rows = len(rows) if isinstance(rows, list) else 0

                        if isinstance(raw_html, str) and raw_html.strip():
                            html_rows = _html_row_count(raw_html)
                            target_rows = n_rows or (len(rows) if isinstance(rows, list) else 0)
                            if target_rows and html_rows and html_rows < target_rows and rows:
                                # Rebuild from FULL rows/cols (untrimmed) to avoid truncated HTML
                                html_parts.append(_build_html_table(cols, rows))
                            else:
                                html_parts.append(raw_html.strip())
                        else:
                            # Fallback to a trimmed rebuild if no raw HTML is available
                            tcols = cols[:mc] if cols else cols
                            trows = [r[:mc] for r in rows[:mr]]
                            html_parts.append(_build_html_table(tcols, trows))
                    preface = "\n\n".join(html_parts)
                    if preface:
                        answer = preface + "\n\n" + answer
                log_final_decision("llm_summary", len(answer))
            else:
                # Empty LLM response, fall back to exec logs + heuristic
                log_orchestrator_stage("llm_empty", "LLM returned empty chunks")
                logs = exec_result.get("logs", "").strip()
                if logs:
                    answer = f"Analysis Results:\n{logs}\n\n" + run_basic_analysis(
                        session_payload, user_message
                    )
                    log_final_decision("exec_logs_plus_heuristic", len(answer))
                else:
                    answer = (
                        run_basic_analysis(session_payload, user_message)
                        + f"\n\n(Exec status: {exec_result.get('status', 'n/a')})"
                    )
                    log_final_decision("heuristic_only", len(answer))

        except Exception as e:
            # LLM failed, but we have execution output - use it directly
            log_error("orchestrator", e, "LLM summary failed")
            logs = exec_result.get("logs", "").strip()
            if logs:
                answer = f"Analysis Results:\n{logs}\n\n" + run_basic_analysis(
                    session_payload, user_message
                )
                log_final_decision("exec_logs_fallback", len(answer))
            else:
                answer = (
                    run_basic_analysis(session_payload, user_message)
                    + f"\n\n(Exec status: {exec_result.get('status', 'n/a')}, LLM error: {str(e)})"
                )
                log_final_decision("heuristic_with_error", len(answer))
    else:
        # Execution skipped or failed - use heuristic analysis
        log_orchestrator_stage(
            "fallback_heuristic", f"Exec status: {exec_result.get('status')}"
        )
        heuristic = run_basic_analysis(session_payload, user_message)
        answer = heuristic + f"\n\n(Exec status: {exec_result.get('status', 'n/a')})"
        log_final_decision("heuristic_fallback", len(answer))

    # Response sanitization (HTML-only tables enforcement)
    try:
        if getattr(settings, "enable_response_sanitizer", False):
            answer = sanitize_answer_html_tables(answer)
    except Exception:
        # Do not block on sanitizer errors
        pass

    progress.append("completed")
    log_orchestrator_stage("completed", f"Final answer length: {len(answer)}")

    return {
        "answer": answer,
        "progress": progress,
        "code": code,
        "violations": violations,
        "exec": exec_result,
    }


def run_basic_analysis(session_payload: Dict[str, Any], user_message: str) -> str:
    """Heuristic answer until full LangGraph + LLM pipeline is implemented.

    Provides:
      - Echo of user intent
      - Basic dataset shape
      - Quick numeric summaries for up to 3 numeric columns
    """
    dataset = session_payload.get("dataset", {})
    columns_meta = session_payload.get("columns", {})
    column_names = dataset.get("column_names", [])
    rows = dataset.get("rows", "?")

    # Identify numeric columns from meta stats if available
    numeric_candidates = []
    for name, meta in columns_meta.items():
        stats = meta.get("stats", {})
        if any(k in stats for k in ("min", "max", "mean")):
            numeric_candidates.append(name)
    numeric_candidates = numeric_candidates[:3]

    parts = [
        f"Your question: {user_message}",
        f"Dataset shape: {rows} rows x {len(column_names)} columns.",
    ]
    if numeric_candidates:
        parts.append("Quick numeric summaries:")
        for col in numeric_candidates:
            stats = columns_meta[col].get("stats", {})
            summary_bits = []
            for k in ("min", "max", "mean"):
                if k in stats:
                    summary_bits.append(f"{k}={stats[k]}")
            if summary_bits:
                parts.append(f" - {col}: " + ", ".join(summary_bits))
    else:
        parts.append("No numeric columns detected for quick summary.")

    return "\n".join(parts)
