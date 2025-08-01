"""
Custom exceptions for the Agentic Data Analysis Workflow.
Provides specific exception types for different error scenarios.
"""

from typing import Optional, Dict, Any


class WorkflowError(Exception):
    """Base exception for workflow-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class SessionError(WorkflowError):
    """Errors related to session management."""

    pass


class SessionNotFoundError(SessionError):
    """Raised when a session cannot be found."""

    def __init__(self, session_id: str):
        super().__init__(
            f"Session '{session_id}' not found",
            error_code="SESSION_NOT_FOUND",
            details={"session_id": session_id},
        )


class SessionExpiredError(SessionError):
    """Raised when a session has expired."""

    def __init__(self, session_id: str):
        super().__init__(
            f"Session '{session_id}' has expired",
            error_code="SESSION_EXPIRED",
            details={"session_id": session_id},
        )


class DataProcessingError(WorkflowError):
    """Errors related to data processing."""

    pass


class FileUploadError(DataProcessingError):
    """Errors during file upload and processing."""

    pass


class InvalidFileFormatError(FileUploadError):
    """Raised when uploaded file format is not supported."""

    def __init__(self, filename: str, supported_formats: list):
        super().__init__(
            f"File '{filename}' has unsupported format. Supported formats: {', '.join(supported_formats)}",
            error_code="INVALID_FILE_FORMAT",
            details={"filename": filename, "supported_formats": supported_formats},
        )


class FileSizeExceededError(FileUploadError):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, filename: str, file_size: int, max_size: int):
        super().__init__(
            f"File '{filename}' size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)",
            error_code="FILE_SIZE_EXCEEDED",
            details={
                "filename": filename,
                "file_size": file_size,
                "max_size": max_size,
            },
        )


class DataCorruptedError(DataProcessingError):
    """Raised when data file is corrupted or unreadable."""

    def __init__(self, filename: str, error_details: str):
        super().__init__(
            f"Data file '{filename}' is corrupted or unreadable: {error_details}",
            error_code="DATA_CORRUPTED",
            details={"filename": filename, "error_details": error_details},
        )


class LLMError(WorkflowError):
    """Errors related to LLM providers."""

    pass


class LLMProviderError(LLMError):
    """Errors with specific LLM providers."""

    def __init__(self, provider: str, error_message: str):
        super().__init__(
            f"LLM provider '{provider}' error: {error_message}",
            error_code="LLM_PROVIDER_ERROR",
            details={"provider": provider, "error_message": error_message},
        )


class LLMProviderNotAvailableError(LLMError):
    """Raised when requested LLM provider is not available."""

    def __init__(self, provider: str, available_providers: list):
        super().__init__(
            f"LLM provider '{provider}' is not available. Available providers: {', '.join(available_providers)}",
            error_code="LLM_PROVIDER_NOT_AVAILABLE",
            details={
                "requested_provider": provider,
                "available_providers": available_providers,
            },
        )


class LLMTokenLimitError(LLMError):
    """Raised when LLM token limit is exceeded."""

    def __init__(self, provider: str, token_count: int, max_tokens: int):
        super().__init__(
            f"Token limit exceeded for provider '{provider}': {token_count} > {max_tokens}",
            error_code="LLM_TOKEN_LIMIT_EXCEEDED",
            details={
                "provider": provider,
                "token_count": token_count,
                "max_tokens": max_tokens,
            },
        )


class CodeExecutionError(WorkflowError):
    """Errors related to code execution."""

    pass


class CodeValidationError(CodeExecutionError):
    """Raised when generated code fails validation."""

    def __init__(self, code: str, validation_error: str):
        super().__init__(
            f"Code validation failed: {validation_error}",
            error_code="CODE_VALIDATION_FAILED",
            details={
                "code": code[:500],  # Truncate for logging
                "validation_error": validation_error,
            },
        )


class CodeExecutionTimeoutError(CodeExecutionError):
    """Raised when code execution exceeds time limit."""

    def __init__(self, timeout_seconds: int):
        super().__init__(
            f"Code execution timed out after {timeout_seconds} seconds",
            error_code="CODE_EXECUTION_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
        )


class UnsafeCodeError(CodeExecutionError):
    """Raised when code contains potentially dangerous operations."""

    def __init__(self, dangerous_patterns: list):
        super().__init__(
            f"Code contains unsafe operations: {', '.join(dangerous_patterns)}",
            error_code="UNSAFE_CODE_DETECTED",
            details={"dangerous_patterns": dangerous_patterns},
        )


class WorkflowExecutionError(WorkflowError):
    """Errors during workflow execution."""

    pass


class WorkflowNodeError(WorkflowExecutionError):
    """Error in specific workflow node."""

    def __init__(self, node_name: str, error_message: str):
        super().__init__(
            f"Error in workflow node '{node_name}': {error_message}",
            error_code="WORKFLOW_NODE_ERROR",
            details={"node_name": node_name, "error_message": error_message},
        )


class WorkflowTimeoutError(WorkflowExecutionError):
    """Raised when workflow execution exceeds time limit."""

    def __init__(self, workflow_id: str, timeout_seconds: int):
        super().__init__(
            f"Workflow '{workflow_id}' timed out after {timeout_seconds} seconds",
            error_code="WORKFLOW_TIMEOUT",
            details={"workflow_id": workflow_id, "timeout_seconds": timeout_seconds},
        )


class MaxRetriesExceededError(WorkflowExecutionError):
    """Raised when maximum retry attempts are exceeded."""

    def __init__(self, operation: str, max_retries: int):
        super().__init__(
            f"Operation '{operation}' failed after {max_retries} retry attempts",
            error_code="MAX_RETRIES_EXCEEDED",
            details={"operation": operation, "max_retries": max_retries},
        )


class ConfigurationError(WorkflowError):
    """Errors related to application configuration."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str):
        super().__init__(
            f"Required configuration '{config_key}' is missing",
            error_code="MISSING_CONFIGURATION",
            details={"config_key": config_key},
        )


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(self, config_key: str, value: Any, expected: str):
        super().__init__(
            f"Invalid configuration for '{config_key}': got {value}, expected {expected}",
            error_code="INVALID_CONFIGURATION",
            details={
                "config_key": config_key,
                "value": str(value),
                "expected": expected,
            },
        )


class RedisError(WorkflowError):
    """Errors related to Redis operations."""

    pass


class RedisConnectionError(RedisError):
    """Raised when Redis connection fails."""

    def __init__(self, redis_url: str):
        super().__init__(
            f"Failed to connect to Redis at {redis_url}",
            error_code="REDIS_CONNECTION_FAILED",
            details={"redis_url": redis_url},
        )


class RedisOperationError(RedisError):
    """Raised when Redis operation fails."""

    def __init__(self, operation: str, error_details: str):
        super().__init__(
            f"Redis operation '{operation}' failed: {error_details}",
            error_code="REDIS_OPERATION_FAILED",
            details={"operation": operation, "error_details": error_details},
        )


class ValidationError(WorkflowError):
    """Errors related to input validation."""

    pass


class QueryValidationError(ValidationError):
    """Raised when user query fails validation."""

    def __init__(self, query: str, validation_issues: list):
        super().__init__(
            f"Query validation failed: {', '.join(validation_issues)}",
            error_code="QUERY_VALIDATION_FAILED",
            details={
                "query": query[:200],  # Truncate for logging
                "validation_issues": validation_issues,
            },
        )


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_CODES = {
    SessionNotFoundError: 404,
    SessionExpiredError: 410,
    InvalidFileFormatError: 400,
    FileSizeExceededError: 413,
    DataCorruptedError: 400,
    LLMProviderNotAvailableError: 503,
    LLMTokenLimitError: 400,
    CodeValidationError: 400,
    UnsafeCodeError: 400,
    CodeExecutionTimeoutError: 408,
    WorkflowTimeoutError: 408,
    MaxRetriesExceededError: 429,
    MissingConfigurationError: 500,
    InvalidConfigurationError: 500,
    RedisConnectionError: 503,
    QueryValidationError: 400,
    ValidationError: 400,
    WorkflowError: 500,
}


def get_http_status_code(exception: Exception) -> int:
    """Get appropriate HTTP status code for an exception."""
    for exc_type, status_code in EXCEPTION_STATUS_CODES.items():
        if isinstance(exception, exc_type):
            return status_code

    # Default to 500 for unknown exceptions
    return 500
