# Copied from original location
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
from typing import Any, Dict
from . import run_processing_pipeline


def _summarize(payload: Dict[str, Any]) -> str:
    dataset = payload.get("dataset", {})
    cols = dataset.get("column_names", [])
    preview_cols = cols[:8]
    more = "" if len(cols) <= 8 else f" (+{len(cols)-8} more)"
    lines = [
        f"Rows: {dataset.get('rows')}  Columns: {dataset.get('columns')}",
        f"Columns: {', '.join(preview_cols)}{more}",
        f"Mode: {payload.get('mode')}  Version: {payload.get('version')}",
    ]
    if payload.get("mode") == "full":
        col_summaries = payload.get("columns", {})
        sample_keys = list(col_summaries.keys())[:3]
        for k in sample_keys:
            c = col_summaries[k]
            lines.append(
                f"  - {k}: type={c.get('type')} null%={c.get('null_pct'):.2f} unique%={c.get('unique_pct'):.2f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the data processing & profiling pipeline on a CSV or Excel file."
    )
    parser.add_argument("file", help="Path to input CSV or Excel file")
    parser.add_argument("--mode", choices=["full", "schema_only"], default="full")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--treat-first-row-as-data", action="store_true")
    parser.add_argument("--suppress-warnings", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if args.suppress_warnings:
        warnings.filterwarnings(
            "ignore", message="Could not infer format", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore",
            message="Parsing dates involving a day of month",
            category=DeprecationWarning,
        )
    config = {
        "sample_size": args.sample_size,
        "treat_first_row_as_data": args.treat_first_row_as_data,
    }
    result = run_processing_pipeline(str(path), mode=args.mode, config=config)
    payload = result["payload"]
    print(_summarize(payload))
    if args.json:
        print("\n=== JSON Payload ===")
        print(json.dumps(payload, indent=2, default=str))
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nSaved JSON payload to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
