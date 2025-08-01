"""
Logging configuration for the Agentic Data Analysis Workflow.
Provides structured logging with JSON format and multiple handlers.
"""

import logging
import logging.config
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""

        # Create base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True,
) -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
    """

    # Create log directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set up specific loggers
    setup_library_loggers(log_level)


def setup_library_loggers(log_level: str) -> None:
    """Configure logging for third-party libraries."""

    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Redis logging
    logging.getLogger("redis").setLevel(logging.WARNING)

    # LangChain/LangGraph logging
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.INFO)

    # FastAPI/Uvicorn logging
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Pandas/NumPy - reduce noise
    logging.getLogger("pandas").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to other classes."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        return logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )


def log_function_call(func_name: str, **kwargs) -> None:
    """Log function call with parameters."""
    logger = logging.getLogger("function_calls")
    logger.info(
        f"Function called: {func_name}",
        extra={"function_name": func_name, "parameters": kwargs},
    )


def log_workflow_event(
    event_type: str, workflow_id: str, node_name: Optional[str] = None, **extra_data
) -> None:
    """Log workflow-related events."""
    logger = logging.getLogger("workflow")

    log_data = {"event_type": event_type, "workflow_id": workflow_id, **extra_data}

    if node_name:
        log_data["node_name"] = node_name

    logger.info(f"Workflow event: {event_type}", extra=log_data)


def log_session_event(event_type: str, session_id: str, **extra_data) -> None:
    """Log session-related events."""
    logger = logging.getLogger("session")

    log_data = {"event_type": event_type, "session_id": session_id, **extra_data}

    logger.info(f"Session event: {event_type}", extra=log_data)


def log_llm_request(
    provider: str,
    model: str,
    prompt_length: int,
    response_length: Optional[int] = None,
    processing_time_ms: Optional[int] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """Log LLM API requests."""
    logger = logging.getLogger("llm")

    log_data = {
        "provider": provider,
        "model": model,
        "prompt_length": prompt_length,
        "success": success,
    }

    if response_length is not None:
        log_data["response_length"] = response_length

    if processing_time_ms is not None:
        log_data["processing_time_ms"] = processing_time_ms

    if error:
        log_data["error"] = error

    if success:
        logger.info("LLM request completed", extra=log_data)
    else:
        logger.error("LLM request failed", extra=log_data)


def log_code_execution(
    session_id: str,
    code_length: int,
    execution_time_ms: int,
    success: bool,
    output_length: Optional[int] = None,
    error: Optional[str] = None,
    plots_generated: int = 0,
) -> None:
    """Log code execution events."""
    logger = logging.getLogger("code_execution")

    log_data = {
        "session_id": session_id,
        "code_length": code_length,
        "execution_time_ms": execution_time_ms,
        "success": success,
        "plots_generated": plots_generated,
    }

    if output_length is not None:
        log_data["output_length"] = output_length

    if error:
        log_data["error"] = error

    if success:
        logger.info("Code execution completed", extra=log_data)
    else:
        logger.error("Code execution failed", extra=log_data)


def log_performance_metrics(
    operation: str, duration_ms: int, success: bool = True, **metrics
) -> None:
    """Log performance metrics."""
    logger = logging.getLogger("performance")

    log_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
        **metrics,
    }

    logger.info(f"Performance metric: {operation}", extra=log_data)


# Pre-configured logging configurations
LOGGING_CONFIGS = {
    "development": {
        "log_level": "DEBUG",
        "log_format": "text",
        "enable_console": True,
        "log_file": None,
    },
    "production": {
        "log_level": "INFO",
        "log_format": "json",
        "enable_console": True,
        "log_file": "/app/logs/app.log",
    },
    "testing": {
        "log_level": "WARNING",
        "log_format": "text",
        "enable_console": False,
        "log_file": None,
    },
}


def configure_logging_for_environment(environment: str) -> None:
    """Configure logging based on environment."""
    config = LOGGING_CONFIGS.get(environment, LOGGING_CONFIGS["development"])
    setup_logging(**config)
