"""
File handling service for the Agentic Data Analysis Workflow.
Handles file upload, validation, processing, and integration with existing modules.
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime

from fastapi import UploadFile

from ..config import get_settings
from ..models import DataMetadata
from ..utils import (
    FileUploadError,
    InvalidFileFormatError,
    FileSizeExceededError,
    DataCorruptedError,
    get_logger,
    log_function_call,
)

# Import existing data processing modules
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / "data_processing"))

try:
    from data_profiler import DataProfiler
    from enhanced_data_cleaner import EnhancedDataCleaner
    from type_inference import TypeInference
    from enhanced_preprocessor import EnhancedPreprocessor

    PROCESSING_MODULES_AVAILABLE = True
except ImportError as e:
    # Create fallback implementations if modules not available
    PROCESSING_MODULES_AVAILABLE = False


class FileHandler:
    """Handles file upload, validation, and processing."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.logger = get_logger(__name__)

        # Create storage directories
        self.upload_dir = Path(self.settings.data_storage_path)
        self.parquet_dir = Path(self.settings.parquet_storage_path)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processing modules if available
        if PROCESSING_MODULES_AVAILABLE:
            try:
                self.data_profiler = DataProfiler()
                self.data_cleaner = EnhancedDataCleaner()
                self.type_inference = TypeInference()
                self.preprocessor = EnhancedPreprocessor()
                self.logger.info("Data processing modules initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize processing modules: {e}")
                PROCESSING_MODULES_AVAILABLE = False

    async def process_upload(
        self, uploaded_file: UploadFile, session_id: Optional[str] = None
    ) -> Tuple[str, DataMetadata]:
        """
        Process uploaded file and return session ID and metadata.

        Args:
            uploaded_file: FastAPI UploadFile instance
            session_id: Optional session ID (generated if not provided)

        Returns:
            Tuple[str, DataMetadata]: Session ID and data metadata
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        log_function_call(
            "process_upload",
            filename=uploaded_file.filename,
            content_type=uploaded_file.content_type,
            session_id=session_id,
        )

        start_time = datetime.utcnow()

        try:
            # Validate file
            await self._validate_file(uploaded_file)

            # Read file content
            content = await uploaded_file.read()
            await uploaded_file.seek(0)  # Reset file pointer

            # Load data into DataFrame
            df = await self._load_dataframe(uploaded_file.filename, content)

            # Process and clean data
            processed_df = await self._process_dataframe(df, uploaded_file.filename)

            # Save as Parquet
            parquet_path = await self._save_parquet(processed_df, session_id)

            # Generate metadata
            metadata = await self._generate_metadata(
                processed_df,
                uploaded_file.filename,
                len(content),
                start_time,
                parquet_path,
            )

            # Clean up temporary files if any
            await self._cleanup_temp_files(session_id)

            self.logger.info(
                f"File processed successfully: {uploaded_file.filename} -> session {session_id}"
            )

            return session_id, metadata

        except Exception as e:
            self.logger.error(f"Failed to process upload {uploaded_file.filename}: {e}")
            # Clean up on error
            await self._cleanup_temp_files(session_id)
            raise

    async def _validate_file(self, uploaded_file: UploadFile) -> None:
        """Validate uploaded file."""

        # Check file size
        if uploaded_file.size and uploaded_file.size > self.settings.upload_max_size:
            raise FileSizeExceededError(
                uploaded_file.filename,
                uploaded_file.size,
                self.settings.upload_max_size,
            )

        # Check file extension
        if uploaded_file.filename:
            file_extension = Path(uploaded_file.filename).suffix.lower()
            if file_extension not in self.settings.allowed_file_extensions:
                raise InvalidFileFormatError(
                    uploaded_file.filename, self.settings.allowed_file_extensions
                )
        else:
            raise FileUploadError("Filename is required")

    async def _load_dataframe(self, filename: str, content: bytes) -> pd.DataFrame:
        """Load file content into pandas DataFrame."""

        file_extension = Path(filename).suffix.lower()

        try:
            if file_extension == ".csv":
                # Try different encodings and separators
                for encoding in ["utf-8", "latin1", "cp1252"]:
                    try:
                        df = pd.read_csv(
                            pd.io.common.BytesIO(content),
                            encoding=encoding,
                            low_memory=False,
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise DataCorruptedError(filename, "Unable to decode CSV file")

            elif file_extension in [".xlsx", ".xls"]:
                df = pd.read_excel(pd.io.common.BytesIO(content))

            elif file_extension == ".json":
                df = pd.read_json(pd.io.common.BytesIO(content))

            elif file_extension == ".parquet":
                df = pd.read_parquet(pd.io.common.BytesIO(content))

            else:
                raise InvalidFileFormatError(
                    filename, self.settings.allowed_file_extensions
                )

            # Basic validation
            if df.empty:
                raise DataCorruptedError(filename, "File contains no data")

            if len(df.columns) == 0:
                raise DataCorruptedError(filename, "File contains no columns")

            return df

        except pd.errors.EmptyDataError:
            raise DataCorruptedError(filename, "File is empty")
        except pd.errors.ParserError as e:
            raise DataCorruptedError(filename, f"Parse error: {str(e)}")
        except Exception as e:
            raise DataCorruptedError(filename, f"Failed to load file: {str(e)}")

    async def _process_dataframe(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Process and clean DataFrame using available modules."""

        if not PROCESSING_MODULES_AVAILABLE:
            # Fallback processing
            return await self._basic_processing(df)

        try:
            # Run processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Enhanced data cleaning
            processed_df = await loop.run_in_executor(
                None, self.data_cleaner.clean_dataframe, df.copy()
            )

            # Type inference
            processed_df = await loop.run_in_executor(
                None, self.type_inference.infer_and_convert_types, processed_df
            )

            # Additional preprocessing
            processed_df = await loop.run_in_executor(
                None, self.preprocessor.preprocess, processed_df
            )

            self.logger.info(f"Advanced processing completed for {filename}")
            return processed_df

        except Exception as e:
            self.logger.warning(f"Advanced processing failed for {filename}: {e}")
            # Fallback to basic processing
            return await self._basic_processing(df)

    async def _basic_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data processing fallback."""

        processed_df = df.copy()

        # Standardize column names
        processed_df.columns = [
            col.strip().lower().replace(" ", "_").replace("-", "_")
            for col in processed_df.columns
        ]

        # Basic type inference
        for col in processed_df.columns:
            if processed_df[col].dtype == "object":
                # Try datetime conversion
                try:
                    processed_df[col] = pd.to_datetime(processed_df[col])
                    continue
                except:
                    pass

                # Try numeric conversion
                try:
                    processed_df[col] = pd.to_numeric(processed_df[col])
                except:
                    pass

        return processed_df

    async def _save_parquet(self, df: pd.DataFrame, session_id: str) -> str:
        """Save DataFrame as Parquet file."""

        parquet_path = self.parquet_dir / f"{session_id}.parquet"

        try:
            # Save in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, df.to_parquet, str(parquet_path), "pyarrow"
            )

            return str(parquet_path)

        except Exception as e:
            raise FileUploadError(f"Failed to save Parquet file: {str(e)}")

    async def _generate_metadata(
        self,
        df: pd.DataFrame,
        original_filename: str,
        file_size_bytes: int,
        upload_timestamp: datetime,
        parquet_path: str,
    ) -> DataMetadata:
        """Generate comprehensive metadata for the dataset."""

        processing_time = (datetime.utcnow() - upload_timestamp).total_seconds() * 1000

        # Basic shape information
        rows, columns = df.shape

        # Column information
        column_info = {}
        total_missing = 0

        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            total_missing += null_count

            col_info = {
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "null_percentage": (null_count / rows) * 100 if rows > 0 else 0,
                "unique_count": int(df[col].nunique()),
                "unique_percentage": (
                    (df[col].nunique() / rows) * 100 if rows > 0 else 0
                ),
            }

            # Add type-specific statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update(
                    {
                        "min": float(df[col].min()) if not df[col].empty else None,
                        "max": float(df[col].max()) if not df[col].empty else None,
                        "mean": float(df[col].mean()) if not df[col].empty else None,
                        "std": float(df[col].std()) if not df[col].empty else None,
                    }
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info.update(
                    {
                        "min_date": str(df[col].min()) if not df[col].empty else None,
                        "max_date": str(df[col].max()) if not df[col].empty else None,
                    }
                )
            else:
                # Top values for categorical data
                top_values = df[col].value_counts().head(5)
                col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

            column_info[col] = col_info

        # Calculate data quality score
        missing_percentage = (
            (total_missing / (rows * columns)) * 100 if rows * columns > 0 else 0
        )
        duplicate_rows = int(df.duplicated().sum())

        # Simple quality scoring (can be enhanced)
        quality_score = max(
            0.0,
            min(
                1.0,
                1.0
                - (missing_percentage / 100)
                - (duplicate_rows / rows if rows > 0 else 0),
            ),
        )

        # Memory usage
        memory_usage = int(df.memory_usage(deep=True).sum())

        # Generate metadata using existing profiler if available
        if PROCESSING_MODULES_AVAILABLE:
            try:
                # Use data profiler for enhanced metadata
                loop = asyncio.get_event_loop()
                profile_results = await loop.run_in_executor(
                    None, self.data_profiler.profile_dataframe, df
                )

                # Merge profiler results with basic metadata
                if profile_results and "data_quality_score" in profile_results:
                    quality_score = profile_results["data_quality_score"]

            except Exception as e:
                self.logger.warning(f"Failed to use data profiler: {e}")

        # Create metadata object
        metadata = DataMetadata(
            original_filename=original_filename,
            file_size_bytes=file_size_bytes,
            upload_timestamp=upload_timestamp,
            shape=(rows, columns),
            column_info=column_info,
            memory_usage_bytes=memory_usage,
            total_missing_values=total_missing,
            missing_percentage=missing_percentage,
            duplicate_rows=duplicate_rows,
            data_quality_score=quality_score,
            parquet_path=parquet_path,
            processing_time_ms=int(processing_time),
        )

        return metadata

    async def load_session_data(self, parquet_path: str) -> pd.DataFrame:
        """Load session data from Parquet file."""

        try:
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

            # Load in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, pd.read_parquet, parquet_path)

            return df

        except Exception as e:
            raise FileUploadError(f"Failed to load session data: {str(e)}")

    async def get_data_preview(
        self, parquet_path: str, n_rows: int = 10
    ) -> Dict[str, Any]:
        """Get preview of data from Parquet file."""

        try:
            # Read only first n_rows for efficiency
            table = pq.read_table(parquet_path)
            preview_table = table.slice(0, min(n_rows, table.num_rows))
            preview_df = preview_table.to_pandas()

            return {
                "preview_rows": preview_df.to_dict("records"),
                "columns": list(preview_df.columns),
                "total_rows": table.num_rows,
                "data_types": {
                    col: str(dtype) for col, dtype in preview_df.dtypes.items()
                },
            }

        except Exception as e:
            raise FileUploadError(f"Failed to get data preview: {str(e)}")

    async def _cleanup_temp_files(self, session_id: str) -> None:
        """Clean up temporary files for a session."""

        try:
            # Remove any temporary files (implementation depends on specific needs)
            temp_pattern = self.upload_dir / f"temp_{session_id}*"
            for temp_file in Path(self.upload_dir).glob(f"temp_{session_id}*"):
                temp_file.unlink(missing_ok=True)

        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files for {session_id}: {e}")

    async def cleanup_session_files(self, session_id: str, parquet_path: str) -> None:
        """Clean up all files associated with a session."""

        try:
            # Remove Parquet file
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

            # Remove any temporary files
            await self._cleanup_temp_files(session_id)

            self.logger.info(f"Cleaned up files for session: {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup files for session {session_id}: {e}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""

        try:
            # Calculate directory sizes
            upload_size = sum(
                f.stat().st_size for f in self.upload_dir.rglob("*") if f.is_file()
            )
            parquet_size = sum(
                f.stat().st_size for f in self.parquet_dir.rglob("*") if f.is_file()
            )

            # Count files
            upload_files = len(list(self.upload_dir.rglob("*")))
            parquet_files = len(list(self.parquet_dir.rglob("*.parquet")))

            return {
                "upload_directory": {
                    "path": str(self.upload_dir),
                    "size_bytes": upload_size,
                    "file_count": upload_files,
                },
                "parquet_directory": {
                    "path": str(self.parquet_dir),
                    "size_bytes": parquet_size,
                    "file_count": parquet_files,
                },
                "total_size_bytes": upload_size + parquet_size,
                "max_file_size_bytes": self.settings.upload_max_size,
                "allowed_extensions": self.settings.allowed_file_extensions,
            }

        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}


# Global file handler instance
_file_handler: Optional[FileHandler] = None


def get_file_handler() -> FileHandler:
    """Get the global file handler instance."""
    global _file_handler
    if _file_handler is None:
        _file_handler = FileHandler()
    return _file_handler


def initialize_file_handler(settings=None) -> FileHandler:
    """Initialize the global file handler with custom settings."""
    global _file_handler
    _file_handler = FileHandler(settings)
    return _file_handler
