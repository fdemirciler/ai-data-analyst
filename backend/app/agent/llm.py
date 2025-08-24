from __future__ import annotations

from typing import Dict, Any, Iterable
import re
from ..config import settings
from ..logging_utils import log_llm_generation, log_error
from .output_parser import parse_exec_output, _build_html_table

_NO_KEY_MSG = "# LLM disabled or missing API key; using stub code\n"


def _get_llm_client(provider: str):
    if provider == "google":
        import google.generativeai as genai

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")
        genai.configure(api_key=settings.gemini_api_key)
        # Apply temperature via generation_config
        return genai.GenerativeModel(
            settings.gemini_model,
            generation_config={"temperature": float(settings.llm_temperature)},
        )
    elif provider == "together":
        from together import Together

        if not settings.together_api_key:
            raise ValueError("TOGETHER_API_KEY is not set in the environment.")
        return Together(api_key=settings.together_api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_analysis_code(
    plan: str, session_payload: Dict[str, Any], question: str
) -> str:
    """Generate Python analysis code using the configured LLM provider."""

    if not settings.enable_llm:
        log_llm_generation("LLM disabled", 0, False)
        return _NO_KEY_MSG + _fallback_stub(plan)

    try:
        client = _get_llm_client(settings.llm_provider)

        schema_summary = _build_schema_summary(session_payload)
        prompt = (
            "You are an expert data analyst. Your goal is to understand the user's intent and generate Python code that fully answers their question using the available dataset.\n\n"
            "CRITICAL ANALYSIS PRINCIPLES:\n"
            "1. UNDERSTAND USER INTENT: Carefully analyze what the user is asking for\n"
            "2. LEVERAGE AVAILABLE DATA: The dataset contains actual data that can answer the question\n"
            "3. BE COMPREHENSIVE: If user asks for a 'list', show the actual values, not just metadata\n"
            "4. EXTRACT AND DISPLAY: Don't just count or describe - show the actual data when requested\n\n"
            "TECHNICAL REQUIREMENTS:\n"
            "- Load data using: df = pd.read_parquet('/data/data.parquet')\n"
            "- Use only pandas, numpy, and basic Python - no external packages\n"
            "- DO NOT write any files or make network calls\n"
            "- Print clear, formatted results that directly answer the question\n"
            "- Do NOT sample or use head()/tail() for final outputs; include ALL relevant rows requested\n"
            "- For lists/values: Display the actual data, not just counts or descriptions\n"
            "- For comparisons: Calculate differences and percentage changes\n"
            "- Format numbers appropriately (commas, decimals)\n"
            "- Include column headers and meaningful labels in output\n\n"
            "OUTPUT FORMAT (JSON-FIRST):\n"
            f"- At the end, print a single JSON result block delimited by '{settings.json_marker_begin}' and '{settings.json_marker_end}'.\n"
            "- The JSON must be a dict: {version:'1.0', tables:[{title, columns, rows, n_rows, n_cols, html}], stats:[{name,value}], messages:[], errors:[]}\n"
            "- Ensure FULL table rendering: set pandas to avoid truncation (e.g., pd.set_option('display.max_rows', None); pd.set_option('display.max_columns', None)).\n"
            "- Include HTML tables using pandas DataFrame.to_html(index=False, max_rows=None, max_cols=None) and set them under tables[*].html.\n"
            "- Keep other printed noise to a minimum; the JSON block is the source of truth.\n\n"
            "HUMAN-READABLE PRINTS (optional):\n"
            "- You may print section headings (e.g., '## Results') and brief bullets before the JSON block if helpful.\n"
            "- Prefer HTML tables if you print any tables outside JSON.\n\n"
            "NAMING CONVENTIONS:\n"
            "- When aggregating across year-like columns (e.g., 2022, 2023, ...), name the dimension column 'Year' (not 'Metric').\n"
            "- When listing metric names from a label column, use 'Metric' as the label column.\n\n"
            "EXAMPLES OF GOOD RESPONSES:\n"
            "- If asked 'show me metrics': Print the actual list of metric names from the data\n"
            "- If asked 'compare periods': Show actual values and calculations\n"
            "- If asked 'what are the values': Display the actual data values, not metadata\n\n"
            "CRITICAL CODING PATTERNS:\n"
            "- Column selection: df[['Metric'] + list(year_columns)] NOT df[['Metric' + str(year_columns)]]\n"
            "- Always use list() when concatenating column lists to avoid string concatenation\n"
            "- Test column existence before selection: assert all(col in df.columns for col in selected_cols)\n\n"
            f"User Question: {question}\n"
            f"Analysis Plan: {plan}\n\n"
            f"Dataset Schema:\n{schema_summary}\n\n"
            "Generate a complete Python script that loads the data and produces a comprehensive analysis that fully answers the user's question with actual data.\n"
            "At the end of the script, construct a 'result' dict and print the JSON block exactly like this (adapt to your variables):\n"
            f"print('{settings.json_marker_begin}')\n"
            "print(json.dumps(result, ensure_ascii=False))\n"
            f"print('{settings.json_marker_end}')\n"
        )

        log_llm_generation(f"LLM ({settings.llm_provider}) - Prompt preview", 0, False)

        text = None
        if settings.llm_provider == "google":
            resp = client.generate_content(prompt)
            text = getattr(resp, "text", None) or (
                resp.candidates[0].content.parts[0].text
                if getattr(resp, "candidates", None)
                else ""
            )
        elif settings.llm_provider == "together":
            response = client.chat.completions.create(
                model=settings.together_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(settings.llm_temperature),
            )
            text = response.choices[0].message.content

        log_llm_generation(
            f"LLM ({settings.llm_provider}) raw response: {text}", len(text or ""), True
        )
        code = _extract_code_block(text) or _fallback_stub(plan)

        log_llm_generation(
            f"LLM ({settings.llm_provider}) response received", len(code), True
        )
        return code
    except Exception as e:
        log_error(
            "llm", e, f"Code generation failed for provider {settings.llm_provider}"
        )
        return _NO_KEY_MSG + _fallback_stub(plan) + f"# LLM error: {e}\n"


def _detect_period_labels_from_logs(logs: str) -> list[str]:
    """Heuristically detect year-like period labels (e.g., 2024, 2025) from logs.

    Looks for 4-digit years in the logs. Returns unique labels preserving order of appearance.
    """
    if not logs:
        return []
    seen = set()
    labels: list[str] = []
    for m in re.finditer(r"\b(19|20)\d{2}\b", logs):
        y = m.group(0)
        if y not in seen:
            seen.add(y)
            labels.append(y)
    return labels[:6]


def _fallback_stub(plan: str) -> str:
    return (
        "# auto-generated analysis script (stub)\n"
        f"PLAN = {plan!r}\n"
        "import pandas as pd\n"
        "print('Plan length:', len(PLAN))\n"
    )


def _build_schema_summary(payload: Dict[str, Any]) -> str:
    ds = payload.get("dataset", {})
    cols = payload.get("columns", {})
    lines = [f"Rows: {ds.get('rows')} Columns: {ds.get('columns')}"]
    for name, meta in list(cols.items())[:10]:
        t = (
            meta.get("type")
            or meta.get("detected_type")
            or meta.get("stats", {}).get("dtype")
        )
        lines.append(f"- {name}: {t}")
    return "\n".join(lines)


def _extract_code_block(text: str | None) -> str | None:
    if not text:
        return None

    # Find the start of the Python code block
    start_marker = "```python"
    start_index = text.find(start_marker)

    if start_index == -1:
        # If no python block is found, look for any code block
        start_marker = "```"
        start_index = text.find(start_marker)
        if start_index == -1:
            # If no code blocks are found, assume the whole text is code
            return text.strip()

    # Find the end of the code block
    end_marker = "```"
    end_index = text.find(end_marker, start_index + len(start_marker))

    if end_index == -1:
        # If no end marker is found, take the rest of the string
        return text[start_index + len(start_marker) :].strip()

    # Extract the code, which is between the markers
    code = text[start_index + len(start_marker) : end_index].strip()
    return code


def stream_summary_chunks(
    session_payload: Dict[str, Any],
    question: str,
    exec_result: Dict[str, Any] | None,
    generated_code: str | None,
    *,
    model_name: str | None = None,
) -> Iterable[str]:
    """Yield summary text chunks using the configured LLM provider; raise on failure.

    Consumers should catch exceptions and fall back as needed.
    """
    if not settings.enable_llm:
        raise RuntimeError("LLM disabled")

    try:
        client = _get_llm_client(settings.llm_provider)

        logs_full = (exec_result or {}).get("logs", "") or ""
        json_raw = (exec_result or {}).get("json_raw") or ""
        logs_for_parser = json_raw or logs_full
        status = (exec_result or {}).get("status", "n/a")
        code = generated_code or ""

        # Parse or reuse structured output from execution logs
        structured = (exec_result or {}).get("structured") or parse_exec_output(
            logs_for_parser,
            max_tables=int(settings.summary_max_tables),
            max_rows=int(getattr(settings, "summary_ui_max_rows", settings.summary_max_rows)),
            max_cols=int(getattr(settings, "summary_ui_max_cols", settings.summary_max_cols)),
        )
        # Persist back into exec_result for transparency
        try:
            if isinstance(exec_result, dict):
                exec_result["structured"] = structured
        except Exception:
            pass

        # Helper to build HTML tables string with limits, preferring raw HTML unless it appears truncated
        def build_tables_html(tables: list[dict], mt: int) -> tuple[str, list[dict]]:
            html_parts: list[str] = []
            metrics: list[dict] = []

            def _html_row_count(s: str) -> int:
                if not isinstance(s, str) or not s.strip():
                    return 0
                m = re.search(r"<tbody[^>]*>(.*?)</tbody>", s, re.I | re.S)
                segment = m.group(1) if m else s
                return len(re.findall(r"<tr[^>]*>", segment, re.I))

            pmr = int(getattr(settings, "summary_prompt_max_rows", settings.summary_max_rows))
            pmc = int(getattr(settings, "summary_prompt_max_cols", settings.summary_max_cols))

            for t in (tables or [])[: mt if mt > 0 else 0]:
                title = (t.get("title") or "").strip()
                if title:
                    html_parts.append(f"Title: {title}\n")
                source = t.get("source")
                # JSON-first: always rebuild from full_* and apply prompt caps
                if source == "json" and (t.get("columns_full") is not None or t.get("rows_full") is not None):
                    cols_full = t.get("columns_full") or []
                    rows_full = t.get("rows_full") or []
                    tcols = cols_full[:pmc] if cols_full else cols_full
                    trows = [r[:pmc] for r in rows_full[:pmr]]
                    html_parts.append(_build_html_table(tcols, trows))
                    # Metrics
                    metrics.append({
                        "source": source or "",
                        "n_rows": int(t.get("n_rows") or len(rows_full)),
                        "n_cols": int(t.get("n_cols") or (len(cols_full) if cols_full else max((len(r) for r in rows_full), default=0))),
                        "truncated": bool(t.get("truncated")),
                        "used_full_html": True,
                        "used_raw_html": False,
                        "used_prompt_caps": {"rows": pmr, "cols": pmc},
                    })
                else:
                    raw_html = t.get("html")
                    cols_full = t.get("columns_full") or t.get("columns") or []
                    rows_full = t.get("rows_full") or t.get("rows") or []
                    # prefer raw html when clearly complete; otherwise rebuild from available data with prompt caps
                    if isinstance(raw_html, str) and raw_html.strip():
                        # Estimate completeness
                        try:
                            n_rows = int(t.get("n_rows") or (len(rows_full) if isinstance(rows_full, list) else 0))
                        except Exception:
                            n_rows = len(rows_full) if isinstance(rows_full, list) else 0
                        html_rows = _html_row_count(raw_html)
                        target_rows = n_rows or (len(rows_full) if isinstance(rows_full, list) else 0)
                        if target_rows and html_rows and html_rows < target_rows and rows_full:
                            tcols = cols_full[:pmc] if cols_full else cols_full
                            trows = [r[:pmc] for r in rows_full[:pmr]]
                            html_parts.append(_build_html_table(tcols, trows))
                            metrics.append({
                                "source": source or "",
                                "n_rows": int(t.get("n_rows") or len(rows_full)),
                                "n_cols": int(t.get("n_cols") or (len(cols_full) if cols_full else max((len(r) for r in rows_full), default=0))),
                                "truncated": bool(t.get("truncated")),
                                "used_full_html": True,
                                "used_raw_html": False,
                                "used_prompt_caps": {"rows": pmr, "cols": pmc},
                            })
                        else:
                            html_parts.append(raw_html.strip())
                            metrics.append({
                                "source": source or "",
                                "n_rows": int(t.get("n_rows") or len(rows_full)),
                                "n_cols": int(t.get("n_cols") or (len(cols_full) if cols_full else max((len(r) for r in rows_full), default=0))),
                                "truncated": bool(t.get("truncated")),
                                "used_full_html": False,
                                "used_raw_html": True,
                                "used_prompt_caps": None,
                            })
                    else:
                        tcols = cols_full[:pmc] if cols_full else cols_full
                        trows = [r[:pmc] for r in rows_full[:pmr]]
                        html_parts.append(_build_html_table(tcols, trows))
                        metrics.append({
                            "source": source or "",
                            "n_rows": int(t.get("n_rows") or len(rows_full)),
                            "n_cols": int(t.get("n_cols") or (len(cols_full) if cols_full else max((len(r) for r in rows_full), default=0))),
                            "truncated": bool(t.get("truncated")),
                            "used_full_html": True,
                            "used_raw_html": False,
                            "used_prompt_caps": {"rows": pmr, "cols": pmc},
                        })
            return "\n\n".join(html_parts), metrics

        # Collect excerpts for fallback only
        excerpts = structured.get("excerpts", {}) if structured else {}

        # Initial limits from settings (only number of tables matters for the prompt)
        max_tables = int(settings.summary_max_tables)
        budget = int(settings.summary_prompt_char_budget)

        # Function to assemble the prompt with current limits (simplified)
        def assemble_prompt(mt: int) -> str:
            tables_html, _ = build_tables_html(structured.get("tables", []), mt)
            prompt_parts = [
                "You are a senior data analyst. Provide a comprehensive analysis that directly answers the user's question.\n\n",
                "CRITICAL REQUIREMENTS:\n",
                "- Integrate the provided HTML tables naturally within your analysis narrative\n",
                "- Reference specific data points from the tables in your explanations\n",
                "- Use clear section headings (## Overview, ## Key Findings, etc.)\n",
                "- Be factual and specific - cite actual values from the data\n",
                "- Include HTML tables where they support your analysis\n\n",
                f"User Question: {question}\n\n",
                "Available Data Tables:\n",
                (tables_html + "\n\n") if tables_html else "(No structured data available)\n\n",
                "Provide your analysis below, embedding tables where relevant:\n\n",
            ]
            # Fallback excerpts if no tables
            if not tables_html:
                head = (excerpts or {}).get("head") or ""
                mid = (excerpts or {}).get("mid") or ""
                tail = (excerpts or {}).get("tail") or ""
                prompt_parts += [
                    "## Raw Output Excerpts (fallback)\n",
                    ("BEGINNING:\n" + head + "\n\n") if head else "",
                    ("MIDDLE:\n" + mid + "\n\n") if mid else "",
                    ("END:\n" + tail + "\n\n") if tail else "",
                ]
            prompt_parts.append(
                "Provide a concise, accurate answer that directly addresses the question using the data above."
            )
            return "".join(prompt_parts)

        # Assemble and adaptively downscale based on table content size only
        initial_limits = {"max_tables": max_tables}
        tables_html, _ = build_tables_html(structured.get("tables", []), max_tables)
        table_budget = int(budget * 0.7)  # Reserve 70% of budget for table content
        downscaled = False
        
        # Check if tables exceed budget, not entire prompt
        if len(tables_html) > table_budget:
            # Reduce only the number of tables
            table_steps = [max_tables, 3, 2, 1]
            t_idx = 0
            while t_idx < len(table_steps) and table_steps[t_idx] > max_tables:
                t_idx += 1
            for t_try in range(t_idx, len(table_steps)):
                test_html, _ = build_tables_html(structured.get("tables", []), table_steps[t_try])
                if len(test_html) <= table_budget:
                    max_tables = table_steps[t_try]
                    downscaled = True
                    break
        
        # Final assemble with the best limits found
        prompt = assemble_prompt(max_tables)

        # Log the prompt for debugging
        log_llm_generation(
            f"Summary LLM ({settings.llm_provider}) prompt preview", len(prompt), True
        )
        print(f"DEBUG SUMMARY PROMPT: {prompt[:500]}...")

        # Observability: parse path and prompt metrics
        try:
            tables_for_path = (structured or {}).get("tables", []) if isinstance(structured, dict) else []
            sources = {t.get("source") for t in tables_for_path if isinstance(t, dict)}
            if "json" in sources:
                parse_path = "json"
            elif "html" in sources:
                parse_path = "html"
            elif "markdown" in sources:
                parse_path = "markdown"
            elif "ascii" in sources:
                parse_path = "ascii"
            elif "text_fallback" in sources:
                parse_path = "text_fallback"
            else:
                parse_path = "none"
            # Capture per-table metrics used in prompt
            used_html, table_metrics = build_tables_html(structured.get("tables", []), max_tables)
            obs = {
                "parse_path": parse_path,
                "prompt_chars": len(prompt),
                "budget": budget,
                "initial_limits": initial_limits,
                "final_limits": {"max_tables": max_tables},
                "downscaled": downscaled,
                "table_count": len(tables_for_path),
            }
            # Attach to exec_result for downstream visibility
            if isinstance(exec_result, dict):
                exec_result.setdefault("observability", {})["summary"] = obs
                exec_result["observability"]["tables"] = table_metrics
            # Also log a compact line
            log_llm_generation(
                f"OBS parse_path={parse_path} prompt={len(prompt)}/{budget} downscaled={downscaled} limits={obs['final_limits']}",
                len(prompt),
                True,
            )
            # Log per-table compacts
            for i, m in enumerate(table_metrics):
                log_llm_generation(
                    f"TABLE[{i}] src={m.get('source')} n={m.get('n_rows')}x{m.get('n_cols')} trunc={m.get('truncated')} full={m.get('used_full_html')} raw={m.get('used_raw_html')}",
                    0,
                    True,
                )
        except Exception:
            pass

        text = None
        if settings.llm_provider == "google":
            resp = client.generate_content(prompt)
            text = getattr(resp, "text", None) or (
                resp.candidates[0].content.parts[0].text
                if getattr(resp, "candidates", None)
                else ""
            )
        elif settings.llm_provider == "together":
            response = client.chat.completions.create(
                model=settings.together_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(settings.llm_temperature),
            )
            text = response.choices[0].message.content

        # Log the response for debugging
        print(f"DEBUG SUMMARY RESPONSE: {text[:500]}..." if text else "No response")

        # Return smaller chunks for better streaming experience
        for part in _chunk_text(text or "", max_len=50):
            if part:
                yield part
        return
    except Exception as e:  # pragma: no cover - robustness around SDK/network
        raise RuntimeError(f"LLM error: {e}")


def _chunk_text(text: str, max_len: int = 600) -> list[str]:
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    i = 0
    while i < len(text):
        parts.append(text[i : i + max_len])
        i += max_len
    return parts
