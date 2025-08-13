from __future__ import annotations

from typing import Dict, Any, Iterable
from ..config import settings
from ..logging_utils import log_llm_generation, log_error

_NO_KEY_MSG = "# LLM disabled or missing API key; using stub code\n"


def _get_llm_client(provider: str):
    if provider == "google":
        import google.generativeai as genai

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")
        genai.configure(api_key=settings.gemini_api_key)
        return genai.GenerativeModel(settings.gemini_model)
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
            "You are a data analyst. Generate a Python script that answers the user's question using the provided dataset.\n"
            "IMPORTANT RULES:\n"
            "- Load data using: df = pd.read_parquet('/data/data.parquet')\n"
            "- Use only pandas, numpy, and basic Python - no external packages\n"
            "- DO NOT write any files or make network calls\n"
            "- Print clear, formatted results that directly answer the question\n"
            "- For variance analysis, calculate differences and percentage changes\n"
            "- Format numbers appropriately (commas, decimals)\n"
            "- Include column headers and meaningful labels in output\n\n"
            f"User Question: {question}\n"
            f"Analysis Plan: {plan}\n\n"
            f"Dataset Schema:\n{schema_summary}\n\n"
            "Generate a complete Python script that loads the data and produces the analysis:"
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
        logs = (exec_result or {}).get("logs", "")
        status = (exec_result or {}).get("status", "n/a")
        code = generated_code or ""

        # Enhanced prompt that prioritizes script output analysis and includes table formatting
        prompt = (
            "You are a senior data analyst. You must analyze ONLY the script execution output below.\n\n"
            "ABSOLUTE REQUIREMENTS:\n"
            "1. IGNORE any general dataset information or schema\n"
            "2. ONLY analyze the specific data shown in the EXECUTION OUTPUT section\n"
            "3. The execution output shows the EXACT periods being compared - use ONLY those periods\n"
            "4. NEVER mention periods or data not explicitly shown in the execution results\n"
            "5. Start your analysis by stating which periods are being compared (as shown in the output)\n\n"
            "TABLE FORMATTING REQUIREMENTS:\n"
            "- When the output contains tabular data, create an HTML table using this exact format:\n"
            "  <table class='analysis-table'>\n"
            "  <thead>\n"
            "  <tr><th>Column 1</th><th>Column 2</th><th>Column 3</th></tr>\n"
            "  </thead>\n"
            "  <tbody>\n"
            "  <tr><td>Value 1</td><td>Value 2</td><td>Value 3</td></tr>\n"
            "  </tbody>\n"
            "  </table>\n"
            "- Always include tables when the execution output shows structured data\n"
            "- Use proper number formatting with commas and appropriate decimals\n"
            "- Include table headers that match the data columns\n\n"
            f"User Question: {question}\n\n"
            f"EXECUTION OUTPUT (your ONLY data source):\n{logs[-3000:]}\n\n"
            "Begin your analysis by identifying the periods shown in the execution output, then analyze only those periods using the exact numbers provided. Include HTML tables for any tabular data."
        )

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
