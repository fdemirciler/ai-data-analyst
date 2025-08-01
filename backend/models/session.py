"""
Pydantic models for session management and workflow state.
Defines data structures for session persistence and workflow orchestration.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class SessionStatus(str, Enum):
    """Status of a session."""

    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    ERROR = "error"


class WorkflowNodeStatus(str, Enum):
    """Status of individual workflow nodes."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DataMetadata(BaseModel):
    """Metadata about the dataset."""

    # Basic information
    original_filename: str = Field(description="Original uploaded filename")
    file_size_bytes: int = Field(description="File size in bytes")
    upload_timestamp: datetime = Field(description="When file was uploaded")

    # Data characteristics
    shape: tuple[int, int] = Field(description="Dataset shape (rows, columns)")
    column_info: Dict[str, Dict[str, Any]] = Field(
        description="Information about each column"
    )
    memory_usage_bytes: int = Field(description="Memory usage in bytes")

    # Data quality metrics
    total_missing_values: int = Field(description="Total number of missing values")
    missing_percentage: float = Field(description="Percentage of missing values")
    duplicate_rows: int = Field(description="Number of duplicate rows")
    data_quality_score: float = Field(description="Overall quality score (0-1)")

    # Processing information
    parquet_path: str = Field(description="Path to stored Parquet file")
    processing_time_ms: int = Field(description="Time taken to process the upload")

    class Config:
        json_schema_extra = {
            "example": {
                "original_filename": "sales_data.csv",
                "file_size_bytes": 2048576,
                "upload_timestamp": "2024-01-15T10:00:00Z",
                "shape": [15000, 12],
                "column_info": {
                    "date": {
                        "dtype": "datetime64[ns]",
                        "null_count": 0,
                        "unique_count": 365,
                        "sample_values": ["2024-01-01", "2024-01-02"],
                    }
                },
                "memory_usage_bytes": 1800000,
                "total_missing_values": 150,
                "missing_percentage": 0.83,
                "duplicate_rows": 5,
                "data_quality_score": 0.92,
                "parquet_path": "/app/data/parquet/session_123.parquet",
                "processing_time_ms": 2500,
            }
        }


class ChatMessage(BaseModel):
    """Individual chat message."""

    message_id: str = Field(description="Unique message identifier")
    query: str = Field(description="User query")
    timestamp: datetime = Field(description="When message was sent")

    # Response information
    response_success: bool = Field(description="Whether response was successful")
    interpretation: str = Field(description="Response interpretation")
    code_generated: Optional[str] = Field(None, description="Generated code")
    plots_generated: int = Field(default=0, description="Number of plots generated")

    # Processing metadata
    processing_time_ms: int = Field(description="Time taken to process")
    llm_provider: str = Field(description="LLM provider used")
    retry_count: int = Field(default=0, description="Number of retries")

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg_001",
                "query": "What are the sales trends?",
                "timestamp": "2024-01-15T10:30:00Z",
                "response_success": True,
                "interpretation": "Sales show strong upward trend...",
                "code_generated": "df.groupby('month')['sales'].mean()",
                "plots_generated": 2,
                "processing_time_ms": 3500,
                "llm_provider": "gemini",
                "retry_count": 0,
            }
        }


class WorkflowNodeState(BaseModel):
    """State of individual workflow node."""

    node_name: str = Field(description="Name of the workflow node")
    status: WorkflowNodeStatus = Field(description="Current node status")
    started_at: Optional[datetime] = Field(
        None, description="When node started processing"
    )
    completed_at: Optional[datetime] = Field(None, description="When node completed")
    processing_time_ms: Optional[int] = Field(
        None, description="Processing time in milliseconds"
    )

    # Node-specific data
    input_data: Optional[Dict[str, Any]] = Field(
        None, description="Input data for the node"
    )
    output_data: Optional[Dict[str, Any]] = Field(
        None, description="Output data from the node"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if node failed"
    )

    # Retry information
    retry_count: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=3, description="Maximum retries allowed")

    class Config:
        json_schema_extra = {
            "example": {
                "node_name": "code_generation",
                "status": "completed",
                "started_at": "2024-01-15T10:30:01Z",
                "completed_at": "2024-01-15T10:30:03Z",
                "processing_time_ms": 2000,
                "input_data": {"query": "Show sales trends"},
                "output_data": {"code": "df.plot()"},
                "error_message": None,
                "retry_count": 0,
                "max_retries": 3,
            }
        }


class WorkflowState(BaseModel):
    """Complete state of a workflow execution."""

    # Workflow identification
    workflow_id: str = Field(description="Unique workflow identifier")
    session_id: str = Field(description="Associated session ID")

    # Current state
    status: str = Field(description="Overall workflow status")
    current_node: Optional[str] = Field(None, description="Currently executing node")

    # Timing information
    started_at: datetime = Field(description="When workflow started")
    completed_at: Optional[datetime] = Field(
        None, description="When workflow completed"
    )
    total_processing_time_ms: Optional[int] = Field(
        None, description="Total processing time"
    )

    # Node states
    nodes: Dict[str, WorkflowNodeState] = Field(
        description="State of each workflow node"
    )
    execution_order: List[str] = Field(description="Order in which nodes were executed")

    # Input/Output
    initial_input: Dict[str, Any] = Field(description="Initial workflow input")
    final_output: Optional[Dict[str, Any]] = Field(
        None, description="Final workflow output"
    )

    # Error handling
    error_count: int = Field(
        default=0, description="Total number of errors encountered"
    )
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "wf_12345",
                "session_id": "session_abc123",
                "status": "completed",
                "current_node": None,
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:05Z",
                "total_processing_time_ms": 5000,
                "nodes": {},
                "execution_order": [
                    "data_processing",
                    "query_analysis",
                    "code_generation",
                ],
                "initial_input": {"query": "Show trends"},
                "final_output": {"interpretation": "Upward trend observed"},
                "error_count": 0,
                "warnings": [],
            }
        }


class SessionData(BaseModel):
    """Complete session data structure."""

    # Session identification
    session_id: str = Field(description="Unique session identifier")

    # Session metadata
    status: SessionStatus = Field(description="Current session status")
    created_at: datetime = Field(description="When session was created")
    last_accessed: datetime = Field(description="When session was last accessed")
    expires_at: datetime = Field(description="When session expires")

    # Data information
    data_metadata: DataMetadata = Field(description="Metadata about the dataset")

    # Chat history
    chat_history: List[ChatMessage] = Field(
        default_factory=list, description="History of chat interactions"
    )

    # Workflow states
    active_workflows: Dict[str, WorkflowState] = Field(
        default_factory=dict, description="Currently active workflow states"
    )
    completed_workflows: List[str] = Field(
        default_factory=list, description="IDs of completed workflows"
    )

    # Session statistics
    total_queries: int = Field(
        default=0, description="Total number of queries processed"
    )
    successful_queries: int = Field(
        default=0, description="Number of successful queries"
    )
    total_processing_time_ms: int = Field(
        default=0, description="Total processing time"
    )

    # Configuration
    preferred_llm_provider: Optional[str] = Field(
        None, description="User's preferred LLM provider"
    )
    session_settings: Dict[str, Any] = Field(
        default_factory=dict, description="Session-specific settings"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123",
                "status": "active",
                "created_at": "2024-01-15T10:00:00Z",
                "last_accessed": "2024-01-15T10:30:00Z",
                "expires_at": "2024-01-15T14:00:00Z",
                "data_metadata": {
                    "original_filename": "data.csv",
                    "file_size_bytes": 1024000,
                    "shape": [1000, 10],
                },
                "chat_history": [],
                "active_workflows": {},
                "completed_workflows": [],
                "total_queries": 5,
                "successful_queries": 4,
                "total_processing_time_ms": 15000,
                "preferred_llm_provider": "gemini",
                "session_settings": {},
            }
        }


class SessionSummary(BaseModel):
    """Summary information about a session."""

    session_id: str = Field(description="Session identifier")
    filename: str = Field(description="Original filename")
    status: SessionStatus = Field(description="Session status")
    created_at: datetime = Field(description="Creation timestamp")
    last_accessed: datetime = Field(description="Last access timestamp")

    # Data summary
    rows: int = Field(description="Number of rows")
    columns: int = Field(description="Number of columns")
    data_quality_score: float = Field(description="Data quality score")

    # Activity summary
    total_queries: int = Field(description="Total queries")
    successful_queries: int = Field(description="Successful queries")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123",
                "filename": "sales_data.csv",
                "status": "active",
                "created_at": "2024-01-15T10:00:00Z",
                "last_accessed": "2024-01-15T10:30:00Z",
                "rows": 15000,
                "columns": 12,
                "data_quality_score": 0.92,
                "total_queries": 5,
                "successful_queries": 4,
            }
        }
