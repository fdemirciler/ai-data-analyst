"""
Data Processing Node for LangGraph Workflow

This node handles the initial data processing phase:
1. Loads raw data from uploaded file
2. Applies data cleaning using existing modules
3. Saves processed data to Parquet format
4. Records processing operations applied

Integrates with existing data processing modules while adding workflow state management.
"""

import asyncio
import time
import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd

from ..state import WorkflowState, WorkflowStateManager
from ...services.file_handler import FileHandler
from ...utils.exceptions import WorkflowError

logger = logging.getLogger(__name__)


async def data_processing_node(state: WorkflowState) -> WorkflowState:
    """
    Process and clean uploaded data file

    This node:
    1. Loads the raw uploaded file
    2. Applies data cleaning and preprocessing
    3. Saves cleaned data to Parquet format
    4. Records metadata about processing operations

    Args:
        state: Current workflow state

    Returns:
        Updated workflow state with processing results
    """
    start_time = time.time()
    node_name = "data_processing"

    try:
        logger.info(f"Starting data processing for session {state['session_id']}")

        # Initialize file handler
        file_handler = FileHandler()

        # Load raw data from uploaded file
        logger.debug(f"Loading data from {state['file_path']}")

        # Read file content
        with open(state["file_path"], "rb") as f:
            file_content = f.read()

        # Load dataframe using file handler
        raw_df = await file_handler._load_dataframe(
            state["original_filename"], file_content
        )

        # Record raw data information
        raw_data_info = {
            "shape": raw_df.shape,
            "columns": list(raw_df.columns),
            "dtypes": raw_df.dtypes.to_dict(),
            "memory_usage_mb": raw_df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": raw_df.isnull().sum().to_dict(),
            "duplicate_rows": raw_df.duplicated().sum(),
        }

        logger.info(
            f"Raw data loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns"
        )

        # Apply data cleaning and preprocessing
        processing_operations = []

        try:
            # Import existing data processing modules
            from ....data_processing.enhanced_data_cleaner import EnhancedDataCleaner
            from ....data_processing.enhanced_preprocessor import (
                EnhancedDataPreprocessor,
            )
            from ....data_processing.type_inference import TypeInferencer

            # Initialize cleaners
            cleaner = EnhancedDataCleaner()
            type_inferrer = TypeInferencer()

            # Step 1: Enhanced type inference
            logger.debug("Applying type inference...")
            type_results = type_inferrer.infer_types(raw_df)
            cleaned_df = raw_df.copy()  # Start with raw data
            if type_results:
                processing_operations.append("enhanced_type_inference")
                logger.debug("Type inference applied - data types analyzed")

            # Step 2: Data cleaning (works with financial data focus)
            logger.debug("Applying data cleaning...")
            initial_shape = cleaned_df.shape
            # The cleaner works with financial data specifically, so we'll try to apply it
            try:
                type_info = (
                    type_results if "type_results" in locals() and type_results else {}
                )
                cleaned_result = cleaner.clean_financial_data(cleaned_df, type_info)
                if cleaned_result.shape != initial_shape:
                    processing_operations.append("financial_data_cleaning")
                    logger.debug(
                        f"Financial data cleaning applied - shape changed from {initial_shape} to {cleaned_result.shape}"
                    )
                    cleaned_df = cleaned_result
            except Exception as clean_error:
                logger.debug(f"Financial cleaning not applicable: {clean_error}")
                # Keep original data if specific cleaning fails

            # Step 3: File-based preprocessing (operates on file level)
            logger.debug("Applying enhanced preprocessing...")
            try:
                preprocessor = EnhancedDataPreprocessor()
                # Since this works on files, we'll skip it for now in the workflow
                # and handle preprocessing in the file_handler instead
                logger.debug(
                    "Enhanced preprocessing skipped - handled at file upload level"
                )
            except Exception as prep_error:
                logger.debug(f"Enhanced preprocessing not applicable: {prep_error}")

            final_df = cleaned_df

        except ImportError as e:
            logger.warning(f"Some data processing modules not available: {e}")
            logger.info("Applying basic data processing fallback...")

            # Fallback to basic processing
            final_df = raw_df.copy()

            # Basic cleaning operations
            # 1. Standardize column names
            original_columns = final_df.columns.tolist()
            final_df.columns = (
                final_df.columns.str.strip().str.lower().str.replace(" ", "_")
            )
            if final_df.columns.tolist() != original_columns:
                processing_operations.append("column_name_standardization")

            # 2. Remove completely empty rows/columns
            final_df = final_df.dropna(how="all")  # Remove all-null rows
            final_df = final_df.loc[
                :, final_df.notna().any()
            ]  # Remove all-null columns
            if final_df.shape != raw_df.shape:
                processing_operations.append("empty_data_removal")

            # 3. Basic type inference for numeric columns
            for col in final_df.columns:
                if final_df[col].dtype == "object":
                    # Try numeric conversion
                    try:
                        numeric_series = pd.to_numeric(final_df[col], errors="coerce")
                        if (
                            numeric_series.notna().sum() > len(final_df) * 0.8
                        ):  # 80% non-null after conversion
                            final_df[col] = numeric_series
                            processing_operations.append("basic_type_inference")
                    except:
                        pass

                    # Try datetime conversion
                    try:
                        if not pd.api.types.is_numeric_dtype(final_df[col]):
                            datetime_series = pd.to_datetime(
                                final_df[col], errors="coerce"
                            )
                            if datetime_series.notna().sum() > len(final_df) * 0.8:
                                final_df[col] = datetime_series
                                processing_operations.append("basic_datetime_inference")
                    except:
                        pass

        # Record cleaned data information
        cleaned_data_info = {
            "shape": final_df.shape,
            "columns": list(final_df.columns),
            "dtypes": final_df.dtypes.to_dict(),
            "memory_usage_mb": final_df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_counts": final_df.isnull().sum().to_dict(),
            "duplicate_rows": final_df.duplicated().sum(),
            "data_quality_score": _calculate_data_quality_score(final_df),
        }

        # Save processed data to Parquet format
        parquet_path = Path(state["parquet_path"])
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Saving processed data to {parquet_path}")
        final_df.to_parquet(parquet_path, index=False, compression="snappy")

        # Calculate processing metrics
        processing_time = time.time() - start_time
        rows_processed = final_df.shape[0]
        processing_rate = rows_processed / processing_time if processing_time > 0 else 0

        logger.info(
            f"Data processing completed: {len(processing_operations)} operations applied, "
            f"{rows_processed} rows processed in {processing_time:.2f}s "
            f"({processing_rate:.0f} rows/sec)"
        )

        # Update workflow state
        results = {
            "raw_data_info": raw_data_info,
            "cleaned_data_info": cleaned_data_info,
            "processing_applied": processing_operations,
        }

        # Record successful completion
        state = WorkflowStateManager.record_node_completion(
            state, node_name, processing_time, results
        )

        # Transition to next node
        state = WorkflowStateManager.transition_to_node(state, "data_profiling")

        return state

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Data processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Record failure
        state = WorkflowStateManager.record_node_failure(
            state, node_name, error_msg, processing_time
        )

        raise WorkflowError(f"Data processing node failed: {str(e)}")


def _calculate_data_quality_score(df: pd.DataFrame) -> float:
    """
    Calculate overall data quality score (0-1 scale)

    Considers:
    - Completeness (non-null values)
    - Consistency (duplicate rows)
    - Validity (appropriate data types)

    Args:
        df: DataFrame to evaluate

    Returns:
        Quality score between 0 and 1
    """
    if df.empty:
        return 0.0

    # Completeness score (percentage of non-null values)
    total_cells = df.size
    non_null_cells = df.count().sum()
    completeness_score = non_null_cells / total_cells if total_cells > 0 else 0

    # Consistency score (inverse of duplicate percentage)
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    consistency_score = 1 - (duplicate_rows / total_rows) if total_rows > 0 else 1

    # Validity score (percentage of columns with appropriate types)
    # Simple heuristic: object columns with high cardinality might be poorly typed
    validity_issues = 0
    for col in df.columns:
        if df[col].dtype == "object":
            unique_ratio = df[col].nunique() / len(df)
            # If object column has very high uniqueness, might need better typing
            if unique_ratio > 0.95 and df[col].nunique() > 100:
                validity_issues += 1

    validity_score = (
        1 - (validity_issues / len(df.columns)) if len(df.columns) > 0 else 1
    )

    # Weighted average (completeness is most important)
    overall_score = (
        0.5 * completeness_score  # 50% weight on completeness
        + 0.3 * consistency_score  # 30% weight on consistency
        + 0.2 * validity_score  # 20% weight on validity
    )

    return round(overall_score, 3)
