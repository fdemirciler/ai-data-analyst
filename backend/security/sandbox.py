"""
Secure Sandbox Environment for Python Code Execution
===================================================

This module provides a secure sandboxed environment for executing
Python code with strict resource limits and security boundaries.
"""

import ast
import sys
import io
import contextlib
import signal
import time
import traceback
import threading
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os

# Import resource module if available (Unix-like systems)
try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    resource = None  # type: ignore
    HAS_RESOURCE = False

from .code_validator import CodeValidator, ValidationLevel, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionLimits:
    """Resource limits for code execution"""

    max_execution_time: float = 30.0  # seconds
    max_memory_mb: int = 256  # MB
    max_output_size: int = 10000  # characters
    max_cpu_time: float = 30.0  # seconds
    allow_network: bool = False
    allow_file_io: bool = False


@dataclass
class ExecutionResult:
    """Result of code execution"""

    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used: int = 0  # bytes
    warnings: Optional[List[str]] = None
    globals_after: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TimeoutError(Exception):
    """Raised when code execution exceeds time limit"""

    pass


class MemoryError(Exception):
    """Raised when code execution exceeds memory limit"""

    pass


class SandboxEnvironment:
    """
    Secure sandbox for executing Python code with resource limits.

    Features:
    - Time and memory limits
    - Restricted built-ins and imports
    - Captured stdout/stderr
    - Clean namespace isolation
    - Resource monitoring
    """

    def __init__(self, limits: Optional[ExecutionLimits] = None):
        """
        Initialize sandbox environment.

        Args:
            limits: Resource limits for execution
        """
        self.limits = limits or ExecutionLimits()
        self.validator = CodeValidator(ValidationLevel.MODERATE)

        # Prepare restricted builtins
        self.safe_builtins = self._create_safe_builtins()

        # Create safe globals environment
        self.safe_globals = self._create_safe_globals()

    def execute_code(
        self, code: str, globals_dict: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code in sandboxed environment.

        Args:
            code: Python code to execute
            globals_dict: Optional globals to make available

        Returns:
            ExecutionResult with output and metadata
        """
        start_time = time.time()

        try:
            # First validate the code
            validation = self.validator.validate_code(code)
            if not validation.is_safe:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Code validation failed: {', '.join(validation.violations)}",
                    warnings=validation.warnings,
                )

            # Use sanitized code if available
            code_to_execute = validation.sanitized_code or code

            # Prepare execution environment
            exec_globals = self.safe_globals.copy()
            if globals_dict:
                # Only add safe items from provided globals
                for key, value in globals_dict.items():
                    if self._is_safe_value(key, value):
                        exec_globals[key] = value

            exec_locals = {}

            # Set up resource limits
            self._set_resource_limits()

            # Execute with timeout and output capture
            result = self._execute_with_limits(
                code_to_execute, exec_globals, exec_locals
            )

            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # Include safe globals in result for inspection
            result.globals_after = {
                k: v
                for k, v in exec_locals.items()
                if self._is_safe_value(k, v) and not k.startswith("_")
            }

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Sandbox execution error: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {e}",
                execution_time=execution_time,
            )

    def _execute_with_limits(
        self, code: str, exec_globals: Dict[str, Any], exec_locals: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute code with timeout and output capture"""

        # Set up output capture
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Set up timeout mechanism
        execution_complete = threading.Event()
        execution_result: Dict[str, Any] = {"result": None, "exception": None}

        def execute_target():
            """Target function for threaded execution"""
            try:
                with contextlib.redirect_stdout(stdout_capture):
                    with contextlib.redirect_stderr(stderr_capture):
                        # Execute the code
                        exec(code, exec_globals, exec_locals)

                execution_result["result"] = "success"

            except Exception as e:
                execution_result["exception"] = e
            finally:
                execution_complete.set()

        # Start execution in separate thread
        execution_thread = threading.Thread(target=execute_target)
        execution_thread.daemon = True
        execution_thread.start()

        # Wait for completion or timeout
        execution_complete.wait(timeout=self.limits.max_execution_time)

        if execution_thread.is_alive():
            # Execution timed out
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue()[: self.limits.max_output_size],
                error=f"Execution timed out after {self.limits.max_execution_time} seconds",
            )

        # Check for exceptions
        if execution_result["exception"]:
            error_msg = str(execution_result["exception"])
            error_traceback = traceback.format_exc()

            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue()[: self.limits.max_output_size],
                error=f"{error_msg}\n{error_traceback}",
            )

        # Success case
        output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Combine stdout and stderr
        full_output = output
        if stderr_output:
            full_output += f"\nSTDERR:\n{stderr_output}"

        # Limit output size
        if len(full_output) > self.limits.max_output_size:
            full_output = (
                full_output[: self.limits.max_output_size] + "\n... [output truncated]"
            )

        return ExecutionResult(
            success=True,
            output=full_output,
            warnings=(
                ["Output truncated"]
                if len(full_output) > self.limits.max_output_size
                else []
            ),
        )

    def _create_safe_builtins(self) -> Dict[str, Any]:
        """Create dictionary of safe built-in functions"""
        safe_builtins = {}

        # Allow safe built-ins
        safe_builtin_names = {
            # Type constructors
            "int",
            "float",
            "str",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "frozenset",
            # Utility functions
            "len",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "sum",
            "min",
            "max",
            "abs",
            "round",
            "pow",
            "divmod",
            # Type checking
            "isinstance",
            "issubclass",
            "type",
            "hasattr",
            # String/representation
            "repr",
            "ascii",
            "ord",
            "chr",
            "hex",
            "oct",
            "bin",
            "format",
            # Iteration
            "iter",
            "next",
            "all",
            "any",
            # Output (controlled)
            "print",
        }

        for name in safe_builtin_names:
            if hasattr(__builtins__, name):
                safe_builtins[name] = getattr(__builtins__, name)

        return safe_builtins

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe global environment"""
        safe_globals = {
            "__builtins__": self.safe_builtins,
            "__name__": "__sandbox__",
            "__doc__": "Secure sandbox environment",
        }

        # Add safe standard library modules
        try:
            import math
            import statistics
            import datetime
            import json
            import re
            import collections
            import itertools
            import functools

            safe_globals.update(
                {
                    "math": math,
                    "statistics": statistics,
                    "datetime": datetime,
                    "json": json,
                    "re": re,
                    "collections": collections,
                    "itertools": itertools,
                    "functools": functools,
                }
            )
        except ImportError as e:
            logger.warning(f"Could not import safe module: {e}")

        # Add data analysis libraries if available
        try:
            import pandas as pd
            import numpy as np

            safe_globals.update(
                {
                    "pd": pd,
                    "pandas": pd,
                    "np": np,
                    "numpy": np,
                }
            )
        except ImportError:
            logger.info("Pandas/NumPy not available in sandbox")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            safe_globals.update(
                {
                    "plt": plt,
                    "matplotlib": plt,  # Simplified access
                    "sns": sns,
                    "seaborn": sns,
                }
            )
        except ImportError:
            logger.info("Matplotlib/Seaborn not available in sandbox")

        return safe_globals

    def _set_resource_limits(self) -> None:
        """Set resource limits for the process"""
        if not HAS_RESOURCE:
            logger.warning("Resource limits not available on this platform (Windows)")
            return

        try:
            # Note: resource module limitations vary by platform
            # This will work on Unix-like systems, but not Windows
            if hasattr(resource, "setrlimit") and hasattr(resource, "RLIMIT_AS"):
                # Set memory limit (soft limit)
                memory_limit = (
                    self.limits.max_memory_mb * 1024 * 1024
                )  # Convert to bytes
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))  # type: ignore

            if hasattr(resource, "setrlimit") and hasattr(resource, "RLIMIT_CPU"):
                # Set CPU time limit
                cpu_limit = int(self.limits.max_cpu_time)
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))  # type: ignore

        except (OSError, ValueError, AttributeError) as e:
            logger.warning(f"Could not set resource limits: {e}")

    def _is_safe_value(self, key: str, value: Any) -> bool:
        """Check if a key-value pair is safe to include in results"""
        # Skip private/magic attributes
        if key.startswith("_"):
            return False

        # Skip functions and classes (potential security risk)
        if callable(value):
            return False

        # Skip modules
        if hasattr(value, "__file__"):
            return False

        # Allow basic data types
        safe_types = (int, float, str, bool, list, dict, tuple, set, type(None))

        try:
            # Try to serialize the value (basic safety check)
            import json

            json.dumps(value, default=str)
            return isinstance(value, safe_types)
        except (TypeError, ValueError):
            return False


class SecurePythonExecutor:
    """
    High-level interface for secure Python code execution.

    This class combines validation and sandboxing for safe code execution
    in data analysis contexts.
    """

    def __init__(
        self,
        execution_limits: Optional[ExecutionLimits] = None,
        validation_level: ValidationLevel = ValidationLevel.MODERATE,
    ):
        """
        Initialize secure executor.

        Args:
            execution_limits: Resource limits for execution
            validation_level: Code validation strictness
        """
        self.limits = execution_limits or ExecutionLimits()
        self.sandbox = SandboxEnvironment(self.limits)
        self.validator = CodeValidator(validation_level)

    def execute(
        self, code: str, context_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code securely.

        Args:
            code: Python code to execute
            context_data: Optional data to make available in execution context

        Returns:
            ExecutionResult with output and metadata
        """
        # Prepare context
        execution_context = {}
        if context_data:
            # Add data to context (e.g., DataFrames, analysis results)
            for key, value in context_data.items():
                if self._is_safe_context_value(key, value):
                    execution_context[key] = value

        # Execute in sandbox
        return self.sandbox.execute_code(code, execution_context)

    def _is_safe_context_value(self, key: str, value: Any) -> bool:
        """Check if a context value is safe to pass to sandbox"""
        # Allow pandas DataFrames and Series
        try:
            import pandas as pd

            if isinstance(value, (pd.DataFrame, pd.Series)):
                return True
        except ImportError:
            pass

        # Allow numpy arrays
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return True
        except ImportError:
            pass

        # Allow basic data types
        safe_types = (int, float, str, bool, list, dict, tuple, set, type(None))
        return isinstance(value, safe_types)


# Convenience functions
def execute_python_code_safely(
    code: str,
    context_data: Optional[Dict[str, Any]] = None,
    limits: Optional[ExecutionLimits] = None,
) -> ExecutionResult:
    """
    Convenience function to execute Python code safely.

    Args:
        code: Python code to execute
        context_data: Optional data context
        limits: Optional execution limits

    Returns:
        ExecutionResult with output and metadata
    """
    executor = SecurePythonExecutor(limits)
    return executor.execute(code, context_data)


# Example usage and testing
if __name__ == "__main__":
    # Test the sandbox
    executor = SecurePythonExecutor()

    # Test cases
    test_codes = [
        # Safe data analysis code
        """
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
})

print("Data shape:", data.shape)
print("Mean values:")
print(data.mean())

# Simple analysis
correlation = data.corr()
print("\\nCorrelation:")
print(correlation)
        """,
        # Code with potential issues
        """
import time

print("Starting computation...")
for i in range(1000):
    x = i ** 2
    if i % 100 == 0:
        print(f"Progress: {i}")

print("Done!")
        """,
        # Code that should be blocked
        """
import os
os.system('ls -la')
        """,
    ]

    for i, code in enumerate(test_codes):
        print(f"\n{'='*50}")
        print(f"Test Case {i+1}")
        print("=" * 50)

        result = executor.execute(code)

        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time:.3f}s")

        if result.output:
            print(f"Output:\n{result.output}")

        if result.error:
            print(f"Error: {result.error}")

        if result.warnings:
            print(f"Warnings: {result.warnings}")
