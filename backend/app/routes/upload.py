from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Any, Dict
from pathlib import Path
import tempfile
import shutil
import pandas as pd

from ..session import session_store
from ..config import settings

try:
    from backend.data_processing_profiling import run_processing_pipeline  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("Failed to import processing pipeline: " + str(e))

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    size = 0
    original_name = file.filename or "uploaded_file"
    allowed_ext = {".csv", ".xlsx", ".xls"}
    ext = Path(original_name).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="session_"))
    raw_path = tmp_dir / f"raw{ext}"
    with raw_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_upload_mb * 1024 * 1024:
                out.close()
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
                raise HTTPException(
                    status_code=400, detail="File exceeds size limit (50MB)"
                )
            out.write(chunk)

    try:
        result = run_processing_pipeline(str(raw_path), mode="full")
    except Exception as e:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    cleaned_df: pd.DataFrame = result["cleaned_df"]
    payload = result["payload"]

    data_path = tmp_dir / "data.parquet"
    try:
        cleaned_df.to_parquet(data_path, index=False)
    except Exception:
        data_path = tmp_dir / "data.csv"
        cleaned_df.to_csv(data_path, index=False)

    session = session_store.create(payload=payload, data_path=data_path)

    return {
        "sessionId": session.session_id,
        "fileId": session.file_id,
        "status": "analyzed",
        "rows": payload.get("dataset", {}).get("rows"),
        "columns": payload.get("dataset", {}).get("column_names", []),
        "sample": payload.get("sample_rows", []),
        "message": "File processed successfully.",
    }
