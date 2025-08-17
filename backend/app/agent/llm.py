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
            "- For lists/values: Display the actual data, not just counts or descriptions\n"
            "- For comparisons: Calculate differences and percentage changes\n"
            "- Format numbers appropriately (commas, decimals)\n"
            "- Include column headers and meaningful labels in output\n\n"
            "OUTPUT FORMAT:\n"
            "- Print section headings to structure output (e.g., '## Overview', '## Results', '## Key Insights').\n"
            "- Use short bullet points for lists; emphasize key values with **bold**.\n"
            "- When printing tables, use HTML via pandas DataFrame.to_html(index=False); avoid Markdown/ASCII tables.\n"
            "- Format numeric columns with proper separators: df.style.format({'col': '{:,.2f}'}).to_html() for decimals, '{:,}' for integers.\n"
            "- Print a 'Title: <...>' line immediately before any related table when appropriate.\n\n"
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
            "Generate a complete Python script that loads the data and produces a comprehensive analysis that fully answers the user's question with actual data:"
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

        schema_summary = _build_schema_summary(session_payload)
        logs_full = (exec_result or {}).get("logs", "") or ""
        status = (exec_result or {}).get("status", "n/a")
        code = generated_code or ""

        # Parse or reuse structured output from execution logs
        structured = (exec_result or {}).get("structured") or parse_exec_output(
            logs_full,
            max_tables=int(settings.summary_max_tables),
            max_rows=int(settings.summary_max_rows),
            max_cols=int(settings.summary_max_cols),
        )
        # Persist back into exec_result for transparency
        try:
            if isinstance(exec_result, dict):
                exec_result["structured"] = structured
        except Exception:
            pass

        # Helper to build HTML tables string with limits
        def build_tables_html(tables: list[dict], mt: int, mr: int, mc: int) -> str:
            html_parts: list[str] = []
            for t in (tables or [])[: mt if mt > 0 else 0]:
                cols = t.get("columns") or []
                rows = t.get("rows") or []
                tcols = cols[:mc] if cols else cols
                trows = [r[:mc] for r in rows[:mr]]
                title = (t.get("title") or "").strip()
                if title:
                    html_parts.append(f"Title: {title}\n")
                html_parts.append(_build_html_table(tcols, trows))
            return "\n\n".join(html_parts)

        # Collect stats and messages
        stats_items = structured.get("stats", []) if structured else []
        messages = structured.get("messages", []) if structured else []
        excerpts = structured.get("excerpts", {}) if structured else {}

        # Initial limits from settings
        max_tables = int(settings.summary_max_tables)
        max_rows = int(settings.summary_max_rows)
        max_cols = int(settings.summary_max_cols)
        err_limit = int(settings.summary_include_errors_limit)
        budget = int(settings.summary_prompt_char_budget)

        detected_periods = _detect_period_labels_from_logs(logs_full)

        # Build column semantics from schema metadata
        cols_meta = session_payload.get("columns", {}) or {}
        def _dtype_of(m: dict) -> str:
            return str(
                m.get("type")
                or m.get("detected_type")
                or (m.get("stats", {}) or {}).get("dtype")
                or ""
            ).lower()

        text_like_cols = [
            name
            for name, meta in cols_meta.items()
            if any(k in _dtype_of(meta) for k in ("object", "string", "category", "text", "str"))
        ]
        numeric_like_cols = [
            name
            for name, meta in cols_meta.items()
            if any(k in _dtype_of(meta) for k in ("int", "float", "number", "numeric"))
        ]
        year_like_cols = [
            name for name in cols_meta.keys() if re.fullmatch(r"(19|20)\d{2}", str(name))
        ]
        metric_col_name = next((n for n in cols_meta.keys() if str(n).lower() == "metric"), None)

        # Function to assemble the prompt with current limits
        def assemble_prompt(mt: int, mr: int, mc: int) -> str:
            tables_html = build_tables_html(structured.get("tables", []), mt, mr, mc)
            # Build compact stats bullets
            stats_lines = "\n".join(
                f"- {s.get('name')}: {s.get('value')}" for s in (stats_items or [])
            )
            # Recent error/warning lines
            errs = messages[-err_limit:] if err_limit > 0 else []
            errs_text = "\n".join(f"- {m}" for m in errs)

            prompt_parts = [
                "You are a senior data analyst providing insights based on script execution results.\n\n",
                
                "ANALYSIS APPROACH:\n",
                "1. PRIORITIZE EXECUTION OUTPUT: Use the provided structured results as your primary source.\n",
                "2. PROVIDE CONTEXT: Use the dataset schema to add interpretation.\n",
                "3. ANSWER THE QUESTION: Be direct and use the actual values shown.\n\n",
                "STRICT DATA HANDLING:\n",
                "- Use exact headers/period labels from the provided tables; do NOT invent years or columns.\n",
                "- Keep NaN as NaN; do not fabricate values.\n\n",
                
                "RESPONSE FORMAT:\n",
                "- **Primary Goal**: Your main purpose is to interpret the data provided below and answer the user's question directly.\n"
                "- **Structure**: Organize your response with clear markdown headings (e.g., `## Overview`, `## Key Findings`).\n"
                "- **Tone**: Write in a clear, professional, and analytical tone. Use concise bullet points and bold formatting (`**text**`) to highlight key metrics, trends, and insights.\n"
                "- **Data Interpretation**: Do not simply repeat the data from the tables. Explain what the numbers *mean*. For example, instead of saying 'Sales were $100', say '**Sales increased by 25%** ($80 to $100), driven by strong performance in the new product line.'\n"
                "- **Table Handling**: The table MUST be in HTML format (using `<table>`, `<tr>`, `<th>`, `<td>`).\n\n",
               
               f"User Question: {question}\n",
                f"Dataset Context: {schema_summary}\n\n",
               
                "COLUMN SEMANTICS (from dataset schema):\n",
                (
                    (
                        f"- Label column for metrics: '{metric_col_name}'\n"
                        if metric_col_name
                        else ""
                    )
                    + (
                        f"- Text-like columns: {', '.join(map(str, text_like_cols[:12]))}\n"
                        if text_like_cols
                        else ""
                    )
                    + (
                        f"- Year-like columns: {', '.join(map(str, year_like_cols[:12]))}\n"
                        if year_like_cols
                        else ""
                    )
                    + (
                        f"- Numeric columns: {', '.join(map(str, numeric_like_cols[:12]))}\n"
                        if numeric_like_cols
                        else ""
                    )
                    + (
                        "- For questions about 'metrics', prefer tables where the label column lists metric names (not years).\n"
                    )
                    + (
                        "- If a table header says 'Metric' but values are 4-digit years, interpret that column as 'Year' for orientation (keep values as-is).\n\n"
                    )
                ),
                "## Key Findings\n",
                "- Summarize the primary results from the structured output.\n\n",
                "## Structured Execution Output\n",
                (tables_html + "\n\n") if tables_html else "(No tables detected)\n\n",
            ]
            if stats_lines:
                prompt_parts += [
                    "## Key Statistics\n",
                    stats_lines + "\n\n",
                ]
            if errs_text:
                prompt_parts += [
                    "## Errors/Warnings (recent)\n",
                    errs_text + "\n\n",
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
            if detected_periods:
                prompt_parts.append(
                    f"DETECTED PERIOD LABELS (from output): {', '.join(detected_periods)}\n\n"
                )
            prompt_parts.append(
                f"GENERATED CODE (context only, do not re-run):\n```python\n{code[:1200]}\n```\n\n"
            )
            prompt_parts.append(
                "Provide a clear answer that directly addresses the user's question, using the data shown above."
            )
            return "".join(prompt_parts)

        # Assemble and adaptively downscale if over budget
        prompt = assemble_prompt(max_tables, max_rows, max_cols)
        if len(prompt) > budget:
            # Reduction steps for rows, then cols, then tables
            row_steps = [50, 30, 15, 10, 5]
            col_steps = [15, 10, 8, 6, 5]
            table_steps = [5, 3, 2, 1]
            # Find current indices
            r_idx = 0
            c_idx = 0
            t_idx = 0
            # Move to current/equal step
            while r_idx < len(row_steps) and row_steps[r_idx] > max_rows:
                r_idx += 1
            while c_idx < len(col_steps) and col_steps[c_idx] > max_cols:
                c_idx += 1
            while t_idx < len(table_steps) and table_steps[t_idx] > max_tables:
                t_idx += 1
            # Try reducing progressively
            for r_try in range(r_idx, len(row_steps)):
                prompt = assemble_prompt(max_tables, row_steps[r_try], max_cols)
                if len(prompt) <= budget:
                    max_rows = row_steps[r_try]
                    break
            if len(prompt) > budget:
                for c_try in range(c_idx, len(col_steps)):
                    prompt = assemble_prompt(max_tables, max_rows, col_steps[c_try])
                    if len(prompt) <= budget:
                        max_cols = col_steps[c_try]
                        break
            if len(prompt) > budget:
                for t_try in range(t_idx, len(table_steps)):
                    prompt = assemble_prompt(table_steps[t_try], max_rows, max_cols)
                    if len(prompt) <= budget:
                        max_tables = table_steps[t_try]
                        break
            # Final assemble with the best limits found
            prompt = assemble_prompt(max_tables, max_rows, max_cols)

        # Log the prompt for debugging
        log_llm_generation(
            f"Summary LLM ({settings.llm_provider}) prompt preview", len(prompt), True
        )
        print(f"DEBUG SUMMARY PROMPT: {prompt[:500]}...")

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
