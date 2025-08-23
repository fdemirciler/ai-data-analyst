from __future__ import annotations

"""
Utilities to sanitize LLM responses to enforce HTML-only tables.

- Converts Markdown pipe tables to HTML tables using the same normalized
  structure as `_build_html_table`.
- Leaves existing HTML tables intact.
- Keeps non-table markdown (headings, lists) unchanged.
"""

import re
from typing import List, Tuple
from .output_parser import _build_html_table


def _detect_markdown_tables(lines: List[str]) -> List[Tuple[int, int]]:
    """Return list of (start, end_exclusive) for markdown table blocks."""
    blocks: List[Tuple[int, int]] = []
    i = 0
    sep_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
    while i < len(lines) - 1:
        if "|" in lines[i] and sep_re.match(lines[i + 1] or ""):
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


def sanitize_answer_html_tables(text: str) -> str:
    """Convert any Markdown tables in `text` to normalized HTML tables.

    This does not alter non-table markdown.
    """
    if not text or "|" not in text:
        return text
    lines = text.splitlines()
    blocks = _detect_markdown_tables(lines)
    if not blocks:
        return text

    # Build result with replacements
    out_parts: List[str] = []
    last = 0
    for s, e in blocks:
        # Append text before this block
        out_parts.append("\n".join(lines[last:s]))
        header, rows = _parse_markdown_table(lines[s:e])
        html = _build_html_table(header, rows)
        out_parts.append(html)
        last = e
    # Append the remainder
    out_parts.append("\n".join(lines[last:]))

    # Join, removing potential double newlines
    result = "\n".join([p for p in out_parts if p is not None])
    # Minor cleanup: collapse triple newlines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result
