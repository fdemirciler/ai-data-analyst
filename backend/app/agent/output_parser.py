from __future__ import annotations

"""
Lightweight parser for extracting structured analysis results from execution logs.

Outputs a dictionary with keys:
- tables: list of table dicts with keys:
    - html: HTML table string (class='analysis-table')
    - columns: list[str] (when derivable)
    - rows: list[list[str]] (when derivable)
    - n_rows: int (original, before trimming)
    - n_cols: int (original, before trimming)
    - source: 'html' | 'markdown' | 'ascii'
    - title: Optional[str]
- stats: list[dict] items like {"name": str, "value": str}
- messages: list[str] (warnings/errors/tracebacks lines)
- excerpts: {head: str, mid: str | None, tail: str}

Heuristics only; designed to be robust to noisy logs and avoid external deps.
"""

import re
from html import escape
from typing import Any, Dict, List, Tuple
import json
from ..config import settings


def _trim_table(columns: List[str], rows: List[List[str]], max_rows: int, max_cols: int) -> Tuple[List[str], List[List[str]]]:
    cols = columns[:max_cols] if columns else columns
    trimmed_rows = [r[:max_cols] for r in rows[:max_rows]]
    return cols, trimmed_rows


def _build_html_table(columns: List[str], rows: List[List[str]]) -> str:
    # Build a minimal, consistent HTML table for LLM consumption
    thead = ""  # include header only if we have column names
    if columns:
        ths = "".join(f"<th>{escape(str(c))}</th>" for c in columns)
        thead = f"<thead>\n<tr>{ths}</tr>\n</thead>\n"
    tbody_rows = []
    for r in rows:
        tds = "".join(f"<td>{escape(str(v))}</td>" for v in r)
        tbody_rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>\n" + "\n".join(tbody_rows) + "\n</tbody>\n"
    return "<table class='analysis-table'>\n" + thead + tbody + "</table>"


def _extract_cells_from_row_html(row_html: str) -> List[str]:
    # Find th/td content; strip inner tags
    cells = re.findall(r"<(?:t[hd])[^>]*>(.*?)</(?:t[hd])>", row_html, flags=re.I | re.S)
    cleaned: List[str] = []
    for c in cells:
        # Remove any tags inside
        c = re.sub(r"<[^>]+>", " ", c)
        c = re.sub(r"\s+", " ", c).strip()
        cleaned.append(c)
    return cleaned


def _parse_html_table_to_data(html: str) -> Tuple[List[str], List[List[str]]]:
    # Try to get header from thead; else first tr
    thead = re.search(r"<thead[^>]*>(.*?)</thead>", html, flags=re.I | re.S)
    columns: List[str] = []
    body_html = html
    if thead:
        columns = _extract_cells_from_row_html(thead.group(1))
        # Remove thead to avoid duplicating first row when scanning tbody
        body_html = html.replace(thead.group(0), "")
    # Extract rows
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", body_html, flags=re.I | re.S)
    rows: List[List[str]] = []
    for tr in trs:
        cells = _extract_cells_from_row_html(tr)
        if not cells:
            continue
        if not columns and re.search(r"<th", tr, flags=re.I):
            # Treat th row as header if columns empty
            columns = cells
            continue
        rows.append(cells)
    return columns, rows


def _detect_markdown_tables(lines: List[str]) -> List[Tuple[int, int]]:
    # Return list of (start_index, end_index_exclusive) for each MD table block
    blocks: List[Tuple[int, int]] = []
    i = 0
    sep_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
    while i < len(lines) - 1:
        if "|" in lines[i] and sep_re.match(lines[i + 1] or ""):
            # Collect until a blank or no pipe
            j = i + 2
            while j < len(lines) and ("|" in lines[j]) and lines[j].strip():
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def _parse_markdown_table(block_lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    if not block_lines:
        return [], []
    # First line header, second line separators, remaining rows
    header = [c.strip() for c in block_lines[0].strip().strip("|").split("|")]
    rows: List[List[str]] = []
    for ln in block_lines[2:]:
        if not ln.strip():
            continue
        parts = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(parts)
    # Normalize row lengths
    max_len = max([len(header)] + [len(r) for r in rows]) if rows else len(header)
    header = header + [""] * (max_len - len(header))
    norm_rows = [r + [""] * (max_len - len(r)) for r in rows]
    return header, norm_rows


def _detect_ascii_tables(lines: List[str]) -> List[Tuple[int, int]]:
    # Detect blocks where columns are separated by 2+ spaces and shape is consistent
    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < len(lines) - 2:
        header = lines[i]
        if not header.strip():
            i += 1
            continue
        header_cols = re.split(r"\s{2,}", header.strip())
        if len(header_cols) < 2:
            i += 1
            continue
        # Next non-empty line must produce a similar number of cols
        j = i + 1
        # Skip separator lines of dashes
        if re.match(r"^[-=\s]+$", lines[j].strip()):
            j += 1
        if j >= len(lines):
            break
        row_cols = re.split(r"\s{2,}", lines[j].strip())
        if len(row_cols) != len(header_cols):
            i += 1
            continue
        # Collect rows while consistent
        k = j
        while k < len(lines):
            line = lines[k]
            if not line.strip():
                break
            cols = re.split(r"\s{2,}", line.strip())
            if len(cols) != len(header_cols):
                break
            k += 1
        # Only consider if we have at least 2 data rows
        if k - j >= 2:
            blocks.append((i, k))
            i = k
        else:
            i += 1
    return blocks


def _parse_ascii_table(block_lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    header_line = block_lines[0]
    header = re.split(r"\s{2,}", header_line.strip())
    rows: List[List[str]] = []
    # Skip potential separator line of dashes under header
    data_lines = block_lines[1:]
    if data_lines and re.match(r"^[-=\s]+$", data_lines[0].strip()):
        data_lines = data_lines[1:]
    for ln in data_lines:
        if not ln.strip():
            continue
        parts = re.split(r"\s{2,}", ln.strip())
        rows.append(parts)
    # Normalize row lengths
    max_len = max([len(header)] + [len(r) for r in rows]) if rows else len(header)
    header = header + [""] * (max_len - len(header))
    norm_rows = [r + [""] * (max_len - len(r)) for r in rows]
    return header, norm_rows


def _extract_tabular_data_from_text(logs: str) -> List[Tuple[List[str], List[List[str]]]]:
    """Fallback: Extract tabular data from plain text when HTML parsing fails.
    
    Looks for patterns like:
    - Space-separated columns with consistent alignment
    - Pipe-separated values
    - Colon-separated key-value pairs
    """
    lines = logs.splitlines()
    tables = []
    
    # Look for space-separated tabular data
    i = 0
    while i < len(lines) - 2:
        line = lines[i].strip()
        if not line or len(line) < 10:
            i += 1
            continue
            
        # Check if this looks like a header row
        words = line.split()
        if len(words) >= 2 and all(len(w) > 1 for w in words[:3]):
            # Look ahead for data rows with similar structure
            data_rows = []
            j = i + 1
            
            # Skip separator lines
            while j < len(lines) and re.match(r'^[-=\s]+$', lines[j].strip()):
                j += 1
                
            # Collect data rows
            while j < len(lines) and j < i + 20:  # Limit search
                data_line = lines[j].strip()
                if not data_line:
                    break
                data_words = data_line.split()
                if len(data_words) == len(words):  # Same column count
                    data_rows.append(data_words)
                    j += 1
                else:
                    break
            
            if len(data_rows) >= 2:  # Found a table
                tables.append((words, data_rows))
                i = j
            else:
                i += 1
        else:
            i += 1
    
    return tables


def parse_json_block(
    logs: str,
    *,
    max_tables: int = 5,
    max_rows: int = 50,
    max_cols: int = 15,
) -> Dict[str, Any] | None:
    """Parse a structured JSON result block delimited by configured markers.

    Returns a normalized dict with keys tables, stats, messages if found; else None.
    """
    begin_marker = settings.json_marker_begin
    end_marker = settings.json_marker_end

    b = logs.rfind(begin_marker)
    if b == -1:
        return None
    e = logs.find(end_marker, b + len(begin_marker))
    if e == -1:
        return None
    content = logs[b + len(begin_marker):e].strip()
    # Strip optional code fences
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        data = json.loads(content)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    # Extract core sections
    raw_tables = data.get("tables") or []
    raw_stats = data.get("stats") or data.get("statistics") or []
    raw_messages = data.get("messages") or []
    raw_errors = data.get("errors") or []

    tables: List[Dict[str, Any]] = []
    for t in raw_tables:
        if not isinstance(t, dict):
            continue
        title = t.get("title")
        cols = t.get("columns") or []
        rows = t.get("rows") or []
        # Compute original sizes when not provided
        try:
            n_rows = int(t.get("n_rows", len(rows)))
        except Exception:
            n_rows = len(rows)
        try:
            n_cols = int(t.get("n_cols", max((len(r) for r in rows), default=len(cols))))
        except Exception:
            n_cols = max((len(r) for r in rows), default=len(cols))

        # Preserve full rows/cols from JSON without trimming for fidelity
        full_cols = list(map(str, cols))
        full_rows = [[str(v) for v in r] for r in rows]

        # Build normalized HTML as fallback (using full data)
        html_norm = _build_html_table(full_cols, full_rows)

        # Prefer raw HTML from JSON if present; fallback to normalized full HTML
        raw_html = t.get("html")
        html_out = raw_html if isinstance(raw_html, str) and raw_html.strip() else html_norm

        tables.append({
            "html": html_out,
            "columns": full_cols,
            "rows": full_rows,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "source": "json",
            "title": title,
        })
        if len(tables) >= max_tables:
            break

    # Sanitize stats
    stats: List[Dict[str, str]] = []
    for s in raw_stats:
        if isinstance(s, dict) and "name" in s and "value" in s:
            stats.append({"name": str(s["name"]), "value": str(s["value"])})
        elif isinstance(s, (list, tuple)) and len(s) == 2:
            stats.append({"name": str(s[0]), "value": str(s[1])})

    # Merge messages and errors (errors prefixed)
    messages: List[str] = []
    for m in raw_messages:
        try:
            messages.append(str(m))
        except Exception:
            continue
    for err in raw_errors:
        try:
            messages.append(f"ERROR: {str(err)}")
        except Exception:
            continue

    return {"tables": tables, "stats": stats, "messages": messages}


def parse_exec_output(logs: str, *, max_tables: int = 5, max_rows: int = 50, max_cols: int = 15) -> Dict[str, Any]:
    """Parse execution logs to extract structured outputs.

    Returns a dict with tables, stats, messages, and excerpts. Tables are normalized to HTML
    and also include underlying columns/rows when derivable so callers can downscale.
    
    Enhanced with fallback mechanisms for partial/malformed tables.
    """
    # JSON-first path
    if settings.enable_json_first:
        json_res = parse_json_block(
            logs, max_tables=max_tables, max_rows=max_rows, max_cols=max_cols
        )
        if json_res is not None:
            head = logs[:800]
            tail = logs[-800:] if len(logs) > 800 else ""
            mid = None
            for kw in ("Metric", "Variance Table", "DataFrame", "describe", "GroupBy", "Total"):
                idx = logs.find(kw)
                if idx != -1:
                    start = max(0, idx - 600)
                    end = min(len(logs), idx + 600)
                    mid = logs[start:end]
                    break
            json_res["excerpts"] = {"head": head, "mid": mid, "tail": tail}
            return json_res

    tables: List[Dict[str, Any]] = []
    stats: List[Dict[str, str]] = []
    messages: List[str] = []

    # 1) HTML tables (with enhanced error handling)
    html_table_pattern = r"<table[\s\S]*?</table>"
    for m in re.finditer(html_table_pattern, logs, flags=re.I):
        html_tbl = m.group(0)
        # Try to capture a preceding non-empty line as a title/caption
        title: str | None = None
        try:
            start_idx = m.start()
            prev_block = logs[:start_idx].splitlines()
            for ln in reversed(prev_block[-5:]):  # scan up to 5 lines back
                if not ln.strip():
                    continue
                # Remove any HTML tags within the line and trim trailing colon
                tline = re.sub(r"<[^>]+>", " ", ln)
                tline = re.sub(r"\s+", " ", tline).strip().rstrip(":")
                if tline:
                    title = tline
                    break
        except Exception:
            title = None
            
        try:
            cols, rows = _parse_html_table_to_data(html_tbl)
            if not cols and not rows:  # Empty table, skip
                continue
        except Exception:
            # Try to recover from malformed HTML table
            try:
                # Extract table content between tags, parse as text
                table_content = re.sub(r"<[^>]+>", " ", html_tbl)
                table_content = re.sub(r"\s+", " ", table_content).strip()
                if len(table_content) > 20:  # Has meaningful content
                    # Try to parse as space-separated data
                    text_tables = _extract_tabular_data_from_text(table_content)
                    if text_tables:
                        cols, rows = text_tables[0]
                    else:
                        continue
                else:
                    continue
            except Exception:
                continue
                
        n_rows = len(rows)
        n_cols = max((len(r) for r in rows), default=len(cols))
        # Trim and build normalized HTML
        tcols, trows = _trim_table(cols, rows, max_rows, max_cols)
        html_norm = _build_html_table(tcols, trows)
        tables.append({
            "html": html_tbl,
            "columns": tcols,
            "rows": trows,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "source": "html",
            "title": title,
        })
        if len(tables) >= max_tables:
            break

    # 2) Markdown tables
    lines = logs.splitlines()
    for (s, e) in _detect_markdown_tables(lines):
        header, rows = _parse_markdown_table(lines[s:e])
        # Title is the nearest preceding non-empty line that isn't a table row
        title: str | None = None
        if s > 0:
            k = s - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            if k >= 0:
                prev = lines[k].strip()
                if prev and ("|" not in prev or prev.count("|") <= 1):
                    title = prev.rstrip(":")
        if header or rows:
            n_rows = len(rows)
            n_cols = max((len(r) for r in rows), default=len(header))
            tcols, trows = _trim_table(header, rows, max_rows, max_cols)
            html_norm = _build_html_table(tcols, trows)
            tables.append({
                "html": html_norm,
                "columns": tcols,
                "rows": trows,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "source": "markdown",
                "title": title,
            })
            if len(tables) >= max_tables:
                break

    # 3) ASCII tables (pandas-like)
    if len(tables) < max_tables:
        for (s, e) in _detect_ascii_tables(lines):
            header, rows = _parse_ascii_table(lines[s:e])
            # Title is the nearest preceding non-empty line
            title: str | None = None
            if s > 0:
                k = s - 1
                while k >= 0 and not lines[k].strip():
                    k -= 1
                if k >= 0:
                    title = lines[k].strip().rstrip(":") or None
            if header or rows:
                n_rows = len(rows)
                n_cols = max((len(r) for r in rows), default=len(header))
                tcols, trows = _trim_table(header, rows, max_rows, max_cols)
                html_norm = _build_html_table(tcols, trows)
                tables.append({
                    "html": html_norm,
                    "columns": tcols,
                    "rows": trows,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "source": "ascii",
                    "title": title,
                })
                if len(tables) >= max_tables:
                    break

    # 4) Key stats (simple regex scan)
    for m in re.finditer(r"\b(mean|min|max|median|std|sum|count)\b\s*[:=]\s*([^\s,\|]+)", logs, flags=re.I):
        name = m.group(1)
        value = m.group(2)
        stats.append({"name": name.lower(), "value": value})
        if len(stats) >= 200:
            break

    # 5) Messages: capture notable warnings/errors (last N preserved by caller)
    for ln in lines[-2000:]:  # scan last 2k lines for recent messages
        if re.search(r"traceback|error|exception|warning", ln, flags=re.I):
            messages.append(ln.strip())

    # 6) Fallback: Extract tables from plain text if no structured tables found
    if len(tables) < max_tables:
        text_tables = _extract_tabular_data_from_text(logs)
        for cols, rows in text_tables[:max_tables - len(tables)]:
            if cols and rows:
                n_rows = len(rows)
                n_cols = max(len(cols), max((len(r) for r in rows), default=0))
                tcols, trows = _trim_table(cols, rows, max_rows, max_cols)
                html_norm = _build_html_table(tcols, trows)
                tables.append({
                    "html": html_norm,
                    "columns": tcols,
                    "rows": trows,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "source": "text_fallback",
                    "title": "Extracted from text",
                })

    # 7) Excerpts (for fallback when no tables)
    head = logs[:800]
    tail = logs[-800:] if len(logs) > 800 else ""
    mid = None
    for kw in ("Metric", "Variance Table", "DataFrame", "describe", "GroupBy", "Total"):
        idx = logs.find(kw)
        if idx != -1:
            start = max(0, idx - 600)
            end = min(len(logs), idx + 600)
            mid = logs[start:end]
            break

    return {
        "tables": tables,
        "stats": stats,
        "messages": messages,
        "excerpts": {"head": head, "mid": mid, "tail": tail},
    }
