import pandas as pd
from pathlib import Path
import sys
from pathlib import Path as _P

_project_root = _P(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.data_processing_profiling import run_processing_pipeline


def test_pipeline_full_mode(tmp_path: Path):
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Charlie", "Dana"],
            "Age": [30, 25, 35, 40],
            "Score %": ["95%", "88%", "91%", "85%"],
            "Joined": ["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-05"],
        }
    )
    csv_path = tmp_path / "people.csv"
    df.to_csv(csv_path, index=False, header=False)
    result = run_processing_pipeline(str(csv_path), mode="full")
    payload = result["payload"]
    assert payload["dataset"]["rows"] in {3, 4}
    assert payload["dataset"]["columns"] >= 4
    assert (
        "Name" in payload["dataset"]["column_names"]
        or "Metric" in payload["dataset"]["column_names"]
    )
    assert len(payload.get("sample_rows", [])) <= 4


def test_pipeline_schema_only(tmp_path: Path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    csv_path = tmp_path / "simple.csv"
    df.to_csv(csv_path, index=False, header=False)
    result = run_processing_pipeline(str(csv_path), mode="schema_only")
    payload = result["payload"]
    assert payload["mode"] == "schema_only"
    assert "columns" in payload
    assert "sample_rows" not in payload


def test_treat_first_row_as_data(tmp_path: Path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    csv_path = tmp_path / "numbers.csv"
    df.to_csv(csv_path, index=False, header=False)
    result_flag = run_processing_pipeline(
        str(csv_path), mode="full", config={"treat_first_row_as_data": True}
    )
    rows_flag = result_flag["payload"]["dataset"]["rows"]
    assert rows_flag == 3
    assert "Metric" not in result_flag["payload"]
