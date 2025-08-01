"""
Data Profiling Node for LangGraph Workflow

This node handles comprehensive data profiling and analysis:
1. Analyzes data structure, types, and quality
2. Generates statistical summaries and patterns
3. Creates anonymized sample data for LLM context
4. Calculates quality metrics and insights

Integrates with existing data profiling modules while adding workflow state management.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path

from ..state import WorkflowState, WorkflowStateManager
from ...utils.exceptions import WorkflowError

logger = logging.getLogger(__name__)


async def data_profiling_node(state: WorkflowState) -> WorkflowState:
    """
    Profile and analyze the processed data

    This node:
    1. Loads processed data from Parquet file
    2. Performs comprehensive data profiling
    3. Generates metadata and statistical summaries
    4. Creates anonymized sample for LLM context
    5. Calculates quality metrics

    Args:
        state: Current workflow state with processed data

    Returns:
        Updated workflow state with profiling results
    """
    start_time = time.time()
    node_name = "data_profiling"

    try:
        logger.info(f"Starting data profiling for session {state['session_id']}")

        # Load processed data from Parquet
        parquet_path = Path(state["parquet_path"])
        if not parquet_path.exists():
            raise FileNotFoundError(f"Processed data not found: {parquet_path}")

        logger.debug(f"Loading processed data from {parquet_path}")
        df = pd.read_parquet(parquet_path)

        if df.empty:
            raise ValueError("Processed data is empty")

        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Try to use existing data profiler module
        data_profile = {}
        try:
            from ....data_processing.data_profiler import DataProfiler

            logger.debug("Using existing DataProfiler module")
            profiler = DataProfiler()

            # Generate comprehensive profile
            profile_result = profiler.profile_data(df)
            data_profile.update(profile_result)

            logger.debug("Enhanced data profiling completed")

        except ImportError as e:
            logger.warning(f"DataProfiler module not available: {e}")
            logger.info("Using built-in data profiling...")

            # Fallback to comprehensive built-in profiling
            data_profile = _comprehensive_data_profile(df)

        # Generate metadata for LLM context
        metadata = _generate_metadata(df)

        # Create anonymized sample data for LLM
        sample_data = _create_anonymized_sample(df, max_rows=15)

        # Calculate quality metrics
        quality_metrics = _calculate_quality_metrics(df)

        # Generate statistical insights
        insights = _generate_statistical_insights(df, data_profile)

        # Processing metrics
        processing_time = time.time() - start_time

        logger.info(
            f"Data profiling completed in {processing_time:.2f}s - "
            f"quality score: {quality_metrics.get('overall_score', 0):.2f}"
        )

        # Update workflow state
        results = {
            "data_profile": data_profile,
            "metadata": metadata,
            "sample_data": sample_data,
            "quality_metrics": quality_metrics,
            "statistical_insights": insights,
        }

        # Record successful completion
        state = WorkflowStateManager.record_node_completion(
            state, node_name, processing_time, results
        )

        # Transition to next node
        state = WorkflowStateManager.transition_to_node(state, "query_analysis")

        return state

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Data profiling failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Record failure
        state = WorkflowStateManager.record_node_failure(
            state, node_name, error_msg, processing_time
        )

        raise WorkflowError(f"Data profiling node failed: {str(e)}")


def _comprehensive_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data profile using built-in analysis

    Args:
        df: DataFrame to profile

    Returns:
        Comprehensive profile dictionary
    """
    profile = {
        "basic_info": {
            "shape": df.shape,
            "columns": list(df.columns),
            "index_type": str(type(df.index)),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        },
        "column_profiles": {},
        "data_types": df.dtypes.to_dict(),
        "missing_data": {
            "total_missing": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
            "columns_with_missing": df.columns[df.isnull().any()].tolist(),
            "missing_by_column": df.isnull().sum().to_dict(),
        },
        "duplicates": {
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percentage": (
                (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
            ),
        },
    }

    # Profile each column
    for col in df.columns:
        col_profile = _profile_column(df[col], col)
        profile["column_profiles"][col] = col_profile

    # Add correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        profile["correlations"] = {
            "matrix": df[numeric_cols].corr().to_dict(),
            "high_correlations": _find_high_correlations(df[numeric_cols]),
        }

    return profile


def _profile_column(series: pd.Series, col_name: str) -> Dict[str, Any]:
    """
    Profile individual column

    Args:
        series: Column data
        col_name: Column name

    Returns:
        Column profile dictionary
    """
    profile = {
        "name": col_name,
        "dtype": str(series.dtype),
        "count": len(series),
        "non_null_count": series.notna().sum(),
        "null_count": series.isnull().sum(),
        "null_percentage": (
            (series.isnull().sum() / len(series)) * 100 if len(series) > 0 else 0
        ),
        "unique_count": series.nunique(),
        "unique_percentage": (
            (series.nunique() / len(series)) * 100 if len(series) > 0 else 0
        ),
    }

    # Type-specific profiling
    if pd.api.types.is_numeric_dtype(series):
        profile.update(_profile_numeric_column(series))
    elif pd.api.types.is_datetime64_any_dtype(series):
        profile.update(_profile_datetime_column(series))
    elif pd.api.types.is_categorical_dtype(series) or series.dtype == "object":
        profile.update(_profile_categorical_column(series))

    return profile


def _profile_numeric_column(series: pd.Series) -> Dict[str, Any]:
    """Profile numeric column"""
    non_null_series = series.dropna()

    if len(non_null_series) == 0:
        return {"statistics": "No non-null values"}

    return {
        "statistics": {
            "mean": float(non_null_series.mean()),
            "median": float(non_null_series.median()),
            "std": float(non_null_series.std()),
            "min": float(non_null_series.min()),
            "max": float(non_null_series.max()),
            "q25": float(non_null_series.quantile(0.25)),
            "q75": float(non_null_series.quantile(0.75)),
            "skewness": float(non_null_series.skew()),
            "kurtosis": float(non_null_series.kurtosis()),
        },
        "outliers": {
            "count": _count_outliers(non_null_series),
            "percentage": (_count_outliers(non_null_series) / len(non_null_series))
            * 100,
        },
    }


def _profile_datetime_column(series: pd.Series) -> Dict[str, Any]:
    """Profile datetime column"""
    non_null_series = series.dropna()

    if len(non_null_series) == 0:
        return {"date_range": "No non-null values"}

    return {
        "date_range": {
            "min_date": str(non_null_series.min()),
            "max_date": str(non_null_series.max()),
            "date_span_days": (non_null_series.max() - non_null_series.min()).days,
        },
        "patterns": {
            "has_time": non_null_series.dt.hour.nunique() > 1,
            "frequency_analysis": _analyze_date_frequency(non_null_series),
        },
    }


def _profile_categorical_column(series: pd.Series) -> Dict[str, Any]:
    """Profile categorical/text column"""
    non_null_series = series.dropna()

    if len(non_null_series) == 0:
        return {"categories": "No non-null values"}

    value_counts = non_null_series.value_counts()

    return {
        "categories": {
            "top_values": value_counts.head(10).to_dict(),
            "rare_values_count": (value_counts == 1).sum(),
            "category_distribution": (
                "uniform"
                if value_counts.std() < value_counts.mean() * 0.1
                else "skewed"
            ),
        },
        "text_analysis": {
            "avg_length": non_null_series.astype(str).str.len().mean(),
            "max_length": non_null_series.astype(str).str.len().max(),
            "min_length": non_null_series.astype(str).str.len().min(),
            "has_special_chars": non_null_series.astype(str)
            .str.contains(r"[^a-zA-Z0-9\s]")
            .any(),
        },
    }


def _generate_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate metadata optimized for LLM context

    Args:
        df: DataFrame to analyze

    Returns:
        Metadata dictionary for LLM consumption
    """
    return {
        "dataset_summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "completeness_percentage": round(
                ((df.size - df.isnull().sum().sum()) / df.size) * 100, 2
            ),
        },
        "column_summary": {
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
            "datetime_columns": df.select_dtypes(
                include=["datetime64"]
            ).columns.tolist(),
            "columns_with_nulls": df.columns[df.isnull().any()].tolist(),
        },
        "data_characteristics": {
            "has_duplicates": df.duplicated().any(),
            "has_missing_data": df.isnull().any().any(),
            "primary_data_types": df.dtypes.value_counts().to_dict(),
            "potential_keys": _identify_potential_keys(df),
        },
    }


def _create_anonymized_sample(df: pd.DataFrame, max_rows: int = 15) -> str:
    """
    Create anonymized sample data for LLM context

    Args:
        df: DataFrame to sample
        max_rows: Maximum number of rows to include

    Returns:
        Anonymized sample as formatted string
    """
    sample = df.head(max_rows).copy()

    # Anonymization strategy
    for col in sample.columns:
        if sample[col].dtype == "object":
            # Mask string values but preserve patterns
            sample[col] = (
                sample[col]
                .astype(str)
                .apply(
                    lambda x: (
                        f"[{x[:2]}{'*' * min(3, max(0, len(str(x)) - 2))}]"
                        if len(str(x)) > 2
                        else str(x)
                    )
                )
            )
        elif pd.api.types.is_numeric_dtype(sample[col]):
            # Round numbers and add slight noise
            sample[col] = sample[col].round(2)

    # Format as readable string with column info
    result = f"Sample Data ({len(sample)} rows × {len(sample.columns)} columns):\n"
    result += f"Columns: {', '.join(sample.columns)}\n\n"
    result += sample.to_string(max_rows=max_rows, index=False)

    return result


def _calculate_quality_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive data quality metrics"""
    if df.empty:
        return {"overall_score": 0.0}

    # Completeness
    completeness = 1 - (df.isnull().sum().sum() / df.size)

    # Consistency (duplicate check)
    consistency = 1 - (df.duplicated().sum() / len(df))

    # Validity (proper data types)
    validity_issues = 0
    for col in df.columns:
        if df[col].dtype == "object":
            # High cardinality object columns might need better typing
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95 and df[col].nunique() > 100:
                validity_issues += 1

    validity = 1 - (validity_issues / len(df.columns))

    # Uniqueness (for key columns)
    uniqueness_scores = []
    for col in df.columns:
        uniqueness_scores.append(df[col].nunique() / len(df))
    avg_uniqueness = np.mean(uniqueness_scores)

    # Overall weighted score
    overall_score = (
        0.4 * completeness  # 40% weight on completeness
        + 0.3 * consistency  # 30% weight on consistency
        + 0.2 * validity  # 20% weight on validity
        + 0.1 * avg_uniqueness  # 10% weight on uniqueness
    )

    return {
        "overall_score": round(overall_score, 3),
        "completeness": round(completeness, 3),
        "consistency": round(consistency, 3),
        "validity": round(validity, 3),
        "uniqueness": round(avg_uniqueness, 3),
    }


def _generate_statistical_insights(
    df: pd.DataFrame, profile: Dict[str, Any]
) -> List[str]:
    """Generate key statistical insights about the data"""
    insights = []

    # Dataset size insights
    if len(df) < 100:
        insights.append(
            "Small dataset - statistical analysis may have limited reliability"
        )
    elif len(df) > 100000:
        insights.append("Large dataset - suitable for robust statistical analysis")

    # Missing data insights
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    if missing_pct > 20:
        insights.append(
            f"High missing data ({missing_pct:.1f}%) - consider imputation strategies"
        )
    elif missing_pct > 0:
        insights.append(f"Moderate missing data ({missing_pct:.1f}%) present")

    # Duplicate insights
    dup_pct = (df.duplicated().sum() / len(df)) * 100
    if dup_pct > 5:
        insights.append(f"Significant duplicate records ({dup_pct:.1f}%) detected")

    # Column type insights
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    if numeric_cols > len(df.columns) * 0.7:
        insights.append("Primarily numeric dataset - suitable for statistical modeling")
    elif numeric_cols == 0:
        insights.append("No numeric columns - focus on categorical analysis")

    return insights


def _count_outliers(series: pd.Series, method: str = "iqr") -> int:
    """Count outliers using IQR method"""
    if len(series) < 4:
        return 0

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return ((series < lower_bound) | (series > upper_bound)).sum()


def _find_high_correlations(
    df: pd.DataFrame, threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """Find highly correlated column pairs"""
    corr_matrix = df.corr()
    high_correlations = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_correlations.append(
                    {
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": round(corr_value, 3),
                    }
                )

    return high_correlations


def _analyze_date_frequency(series: pd.Series) -> str:
    """Analyze frequency pattern in datetime series"""
    if len(series) < 2:
        return "insufficient_data"

    # Calculate differences between consecutive dates
    diffs = series.sort_values().diff().dropna()

    # Most common difference
    mode_diff = diffs.mode()
    if len(mode_diff) > 0:
        days = mode_diff.iloc[0].days
        if days == 1:
            return "daily"
        elif days == 7:
            return "weekly"
        elif 28 <= days <= 31:
            return "monthly"
        elif 365 <= days <= 366:
            return "yearly"
        else:
            return f"custom_{days}_days"

    return "irregular"


def _identify_potential_keys(df: pd.DataFrame) -> List[str]:
    """Identify columns that could be primary keys"""
    potential_keys = []

    for col in df.columns:
        # Check if unique and not mostly null
        if df[col].nunique() == len(df) and df[col].notna().sum() > len(df) * 0.95:
            potential_keys.append(col)

    return potential_keys
