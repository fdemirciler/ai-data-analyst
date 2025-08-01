"""
Pydantic models for API requests.
Defines the structure and validation for incoming API requests.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from enum import Enum


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""

    STATISTICAL = "statistical"
    VISUALIZATION = "visualization"
    CORRELATION = "correlation"
    SUMMARY = "summary"
    CUSTOM = "custom"
    EXPLORATORY = "exploratory"
    PREDICTIVE = "predictive"


class LLMProvider(str, Enum):
    """Available LLM providers."""

    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    TOGETHER = "together"


class UploadRequest(BaseModel):
    """
    File upload request validation.
    Note: File validation is handled by FastAPI UploadFile,
    this model is for any additional metadata.
    """

    description: Optional[str] = Field(
        None, max_length=500, description="Optional description of the dataset"
    )

    class Config:
        json_schema_extra = {
            "example": {"description": "Sales data for Q4 2024 analysis"}
        }


class AnalysisRequest(BaseModel):
    """Request for data analysis."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The analysis question or request",
    )
    analysis_type: Optional[AnalysisType] = Field(
        None, description="Type of analysis to perform (auto-detected if not specified)"
    )
    max_retries: Optional[int] = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of retries for failed analysis",
    )
    llm_provider: Optional[LLMProvider] = Field(
        None, description="LLM provider to use (uses default if not specified)"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for the analysis"
    )
    temperature: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate and clean the query string."""
        # Remove excessive whitespace
        cleaned = " ".join(v.strip().split())

        if len(cleaned) < 1:
            raise ValueError("Query cannot be empty")

        # Check for potentially harmful content
        dangerous_patterns = [
            "import os",
            "import sys",
            "subprocess",
            "exec(",
            "eval(",
            "__import__",
            "open(",
            "file(",
            "delete",
            "remove",
        ]

        query_lower = cleaned.lower()
        for pattern in dangerous_patterns:
            if pattern in query_lower:
                raise ValueError(
                    f"Query contains potentially dangerous content: {pattern}"
                )

        return cleaned

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main trends in the sales data?",
                "analysis_type": "statistical",
                "max_retries": 2,
                "llm_provider": "gemini",
                "temperature": 0.1,
            }
        }


class SessionRequest(BaseModel):
    """Request for session operations."""

    session_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique session identifier"
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v):
        """Validate session ID format."""
        # Remove whitespace
        cleaned = v.strip()

        if not cleaned:
            raise ValueError("Session ID cannot be empty")

        # Basic format validation (alphanumeric, hyphens, underscores)
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", cleaned):
            raise ValueError(
                "Session ID can only contain letters, numbers, hyphens, and underscores"
            )

        return cleaned

    class Config:
        json_schema_extra = {"example": {"session_id": "abc123-def456-ghi789"}}


class WorkflowExecutionRequest(BaseModel):
    """Request for direct workflow execution."""

    session_id: str = Field(..., description="Session identifier")
    workflow_type: str = Field(
        default="analysis", description="Type of workflow to execute"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters for workflow execution"
    )
    priority: Optional[int] = Field(
        default=1, ge=1, le=5, description="Execution priority (1=highest, 5=lowest)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123-def456-ghi789",
                "workflow_type": "analysis",
                "parameters": {"include_visualizations": True},
                "priority": 1,
            }
        }


class ChatHistoryRequest(BaseModel):
    """Request for chat history retrieval."""

    session_id: str = Field(..., description="Session identifier")
    limit: Optional[int] = Field(
        default=50, ge=1, le=200, description="Maximum number of messages to retrieve"
    )
    offset: Optional[int] = Field(
        default=0, ge=0, description="Number of messages to skip"
    )

    class Config:
        json_schema_extra = {
            "example": {"session_id": "abc123-def456-ghi789", "limit": 20, "offset": 0}
        }
