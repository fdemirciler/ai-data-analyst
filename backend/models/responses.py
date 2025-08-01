"""
Pydantic models for API responses.
Defines the structure for outgoing API responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class WorkflowStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ExecutionResult(BaseModel):
    """Result of code execution."""

    success: bool = Field(description="Whether execution was successful")
    output: str = Field(default="", description="Standard output from execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time_ms: int = Field(description="Execution time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "output": "Mean: 42.5\nStandard deviation: 15.2",
                "error": None,
                "execution_time_ms": 1250,
            }
        }


class PlotInfo(BaseModel):
    """Information about a generated plot."""

    plot_id: str = Field(description="Unique identifier for the plot")
    title: Optional[str] = Field(None, description="Plot title")
    description: Optional[str] = Field(None, description="Plot description")
    plot_type: Optional[str] = Field(
        None, description="Type of plot (bar, line, scatter, etc.)"
    )
    base64_image: str = Field(description="Base64 encoded image data")

    class Config:
        json_schema_extra = {
            "example": {
                "plot_id": "plot_001",
                "title": "Sales Trend Over Time",
                "description": "Monthly sales data showing upward trend",
                "plot_type": "line",
                "base64_image": "iVBORw0KGgoAAAANSUhEUgAA...",
            }
        }


class AnalysisResponse(BaseModel):
    """Response from data analysis workflow."""

    success: bool = Field(description="Whether analysis was successful")
    session_id: str = Field(description="Session identifier")
    query: str = Field(description="Original user query")
    interpretation: str = Field(description="Human-readable interpretation of results")

    # Code and execution details
    code: Optional[str] = Field(None, description="Generated Python code")
    execution_result: Optional[ExecutionResult] = Field(
        None, description="Code execution results"
    )

    # Visual outputs
    plots: List[PlotInfo] = Field(
        default_factory=list, description="Generated visualizations"
    )

    # Workflow metadata
    workflow_status: WorkflowStatus = Field(description="Current workflow status")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
    nodes_executed: List[str] = Field(
        default_factory=list, description="Workflow nodes that were executed"
    )

    # Timestamps
    started_at: datetime = Field(description="When analysis started")
    completed_at: Optional[datetime] = Field(
        None, description="When analysis completed"
    )

    # Additional metadata
    llm_provider: Optional[str] = Field(None, description="LLM provider used")
    model_name: Optional[str] = Field(None, description="Specific model used")
    token_usage: Optional[Dict[str, int]] = Field(
        None, description="Token usage statistics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "session_id": "abc123-def456-ghi789",
                "query": "What are the main trends in sales data?",
                "interpretation": "The analysis shows a strong upward trend in sales over the past year...",
                "code": "import pandas as pd\ndf.groupby('month')['sales'].mean()",
                "execution_result": {
                    "success": True,
                    "output": "Strong upward trend with 15% growth",
                    "error": None,
                    "execution_time_ms": 1250,
                },
                "plots": [],
                "workflow_status": "completed",
                "retry_count": 0,
                "processing_time_ms": 5500,
                "nodes_executed": [
                    "data_processing",
                    "query_analysis",
                    "code_generation",
                ],
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:05Z",
                "llm_provider": "gemini",
                "model_name": "gemini-2.0-flash-exp",
            }
        }


class SessionInfo(BaseModel):
    """Information about a data session."""

    session_id: str = Field(description="Unique session identifier")
    filename: str = Field(description="Original filename")
    file_size: int = Field(description="File size in bytes")

    # Data characteristics
    rows: int = Field(description="Number of rows in dataset")
    columns: int = Field(description="Number of columns in dataset")
    column_names: List[str] = Field(description="List of column names")

    # Quality metrics
    data_quality_score: float = Field(description="Overall data quality score (0-1)")
    missing_data_percentage: float = Field(description="Percentage of missing data")

    # Timestamps
    created_at: datetime = Field(description="When session was created")
    last_accessed: datetime = Field(description="When session was last accessed")
    expires_at: datetime = Field(description="When session will expire")

    # Processing status
    processing_status: str = Field(
        default="ready", description="Current processing status"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123-def456-ghi789",
                "filename": "sales_data.csv",
                "file_size": 2048576,
                "rows": 15000,
                "columns": 12,
                "column_names": ["date", "product", "sales", "region"],
                "data_quality_score": 0.87,
                "missing_data_percentage": 2.3,
                "created_at": "2024-01-15T10:00:00Z",
                "last_accessed": "2024-01-15T10:30:00Z",
                "expires_at": "2024-01-15T14:00:00Z",
                "processing_status": "ready",
            }
        }


class DataPreview(BaseModel):
    """Preview of dataset."""

    preview_rows: List[Dict[str, Any]] = Field(description="Sample rows from dataset")
    columns: List[str] = Field(description="Column names")
    total_rows: int = Field(description="Total number of rows in full dataset")

    # Data type information
    data_types: Dict[str, str] = Field(description="Data types for each column")

    # Quality metrics per column
    quality_metrics: Dict[str, Dict[str, Union[int, float]]] = Field(
        description="Quality metrics for each column"
    )

    # Statistical summary
    statistical_summary: Optional[Dict[str, Any]] = Field(
        None, description="Basic statistical summary for numeric columns"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "preview_rows": [
                    {"date": "2024-01-01", "product": "Widget A", "sales": 1500},
                    {"date": "2024-01-02", "product": "Widget B", "sales": 2300},
                ],
                "columns": ["date", "product", "sales"],
                "total_rows": 15000,
                "data_types": {
                    "date": "datetime64[ns]",
                    "product": "object",
                    "sales": "int64",
                },
                "quality_metrics": {
                    "date": {"null_count": 0, "unique_count": 365},
                    "product": {"null_count": 5, "unique_count": 150},
                    "sales": {"null_count": 0, "unique_count": 8500},
                },
            }
        }


class ChatHistoryItem(BaseModel):
    """Single item in chat history."""

    message_id: str = Field(description="Unique message identifier")
    query: str = Field(description="User query")
    response: AnalysisResponse = Field(description="Analysis response")
    timestamp: datetime = Field(description="When message was sent")

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg_001",
                "query": "Show me sales trends",
                "response": {
                    "success": True,
                    "interpretation": "Sales show upward trend...",
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class ChatHistoryResponse(BaseModel):
    """Response containing chat history."""

    session_id: str = Field(description="Session identifier")
    messages: List[ChatHistoryItem] = Field(description="Chat history messages")
    total_messages: int = Field(description="Total number of messages in session")
    has_more: bool = Field(description="Whether there are more messages available")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123-def456-ghi789",
                "messages": [],
                "total_messages": 15,
                "has_more": True,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Overall system status")
    timestamp: datetime = Field(description="Health check timestamp")

    # Service statuses
    services: Dict[str, Dict[str, Any]] = Field(
        description="Status of individual services"
    )

    # System metrics
    system_metrics: Optional[Dict[str, Any]] = Field(
        None, description="System performance metrics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "services": {
                    "redis": {"status": "connected", "response_time_ms": 2},
                    "llm_providers": {
                        "status": "available",
                        "active_provider": "gemini",
                    },
                },
                "system_metrics": {"memory_usage_mb": 512, "cpu_usage_percent": 25},
            }
        }


class ErrorResponse(BaseModel):
    """Error response format."""

    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(description="When error occurred")
    request_id: Optional[str] = Field(
        None, description="Request identifier for tracking"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid query format",
                "details": {"field": "query", "issue": "too short"},
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_12345",
            }
        }
