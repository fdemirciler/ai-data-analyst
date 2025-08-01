"""
Code Execution Node for LangGraph Workflow

This node handles secure Python code execution using the security layer:
1. Validates code before execution
2. Executes generated Python code safely in sandbox
3. Captures output, errors, and visualizations
4. Handles timeouts and resource limits
5. Encodes plots as base64 for frontend

This is where the generated code actually runs against the real data.
"""

import asyncio
import time
import logging
import io
import sys
import base64
import traceback
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, List
import warnings

warnings.filterwarnings("ignore")

from ..state import WorkflowState, WorkflowStateManager
from ...utils.exceptions import WorkflowError
from ...security import SecurePythonExecutor, ExecutionLimits, ValidationLevel

logger = logging.getLogger(__name__)


async def code_execution_node(state: WorkflowState) -> WorkflowState:
    """
    Execute generated Python code in a secure environment using security layer

    This node:
    1. Validates code before execution
    2. Sets up secure execution context with data
    3. Executes code in sandbox with monitoring
    4. Captures outputs, plots, and errors
    5. Handles timeouts and resource limits

    Args:
        state: Current workflow state with generated code

    Returns:
        Updated workflow state with execution results
    """
    start_time = time.time()
    node_name = "code_execution"

    try:
        logger.info(f"Starting secure code execution for session {state['session_id']}")

        generated_code = state.get("generated_code", "")
        if not generated_code:
            raise ValueError("No code to execute")

        logger.debug(f"Executing code ({len(generated_code)} characters)")

        # Set up secure executor with appropriate limits
        estimated_time = state.get("estimated_execution_time") or 30.0
        if isinstance(estimated_time, (int, float)):
            max_time = min(float(estimated_time) + 10.0, 120.0)
        else:
            max_time = 60.0

        execution_limits = ExecutionLimits(
            max_execution_time=max_time,
            max_memory_mb=512,
            max_output_size=100000,
            allow_network=False,
            allow_file_io=False,
        )

        secure_executor = SecurePythonExecutor(
            execution_limits=execution_limits, validation_level=ValidationLevel.MODERATE
        )

        # Prepare execution context with data
        execution_context = await _setup_execution_environment(state)

        # Execute code securely
        execution_result = secure_executor.execute(generated_code, execution_context)

        # Process matplotlib plots if any were created
        plots = _process_visualizations()

        # Calculate execution metrics
        processing_time = time.time() - start_time

        logger.info(
            f"Secure code execution completed in {processing_time:.2f}s - "
            f"success: {execution_result.success}, plots: {len(plots)}"
        )

        # Prepare results
        results = {
            "execution_result": {
                "success": execution_result.success,
                "output": execution_result.output,
                "error": execution_result.error,
                "execution_time": execution_result.execution_time,
                "warnings": execution_result.warnings,
                "plots": plots,
            },
            "execution_output": execution_result.output,
            "execution_error": execution_result.error,
            "execution_plots": plots,
            "execution_time": processing_time,
        }

        # Record successful completion
        state = WorkflowStateManager.record_node_completion(
            state, node_name, processing_time, results
        )

        # Transition to next node
        state = WorkflowStateManager.transition_to_node(state, "response_formatting")

        return state

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Code execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Record failure
        state = WorkflowStateManager.record_node_failure(
            state, node_name, error_msg, processing_time
        )

        raise WorkflowError(f"Code execution node failed: {str(e)}")


async def _setup_execution_environment(state: WorkflowState) -> Dict[str, Any]:
    """
    Set up secure execution environment with data and libraries

    Args:
        state: Current workflow state

    Returns:
        Execution environment dictionary
    """
    try:
        # Load data from Parquet file
        parquet_path = state["parquet_path"]
        df = pd.read_parquet(parquet_path)

        logger.debug(
            f"Loaded data for execution: {df.shape[0]} rows × {df.shape[1]} columns"
        )

        # Create safe execution namespace
        execution_env = {
            # Data
            "df": df,
            # Core libraries
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "plt": plt,
            "matplotlib": matplotlib,
            "sns": sns,
            "seaborn": sns,
            # Built-in functions (safe subset)
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            # Math functions
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "var": np.var,
            # Additional safe libraries
            "warnings": warnings,
        }

        # Try to add scipy stats if available
        try:
            from scipy import stats

            execution_env["stats"] = stats
        except ImportError:
            logger.debug("scipy not available for code execution")

        # Try to add basic sklearn if available
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score

            execution_env["LinearRegression"] = LinearRegression
            execution_env["mean_squared_error"] = mean_squared_error
            execution_env["r2_score"] = r2_score
        except ImportError:
            logger.debug("sklearn not available for code execution")

        return execution_env

    except Exception as e:
        logger.error(f"Failed to set up execution environment: {e}")
        raise WorkflowError(f"Execution environment setup failed: {str(e)}")


async def _execute_code_safely(
    code: str, execution_env: Dict[str, Any], timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Execute Python code safely with timeout and output capture

    Args:
        code: Python code to execute
        execution_env: Execution environment namespace
        timeout: Maximum execution time in seconds

    Returns:
        Execution result dictionary
    """
    result = {
        "success": False,
        "output": "",
        "error": None,
        "timeout": False,
        "execution_time": 0.0,
    }

    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    start_time = time.time()

    try:
        # Clear any existing plots
        plt.clf()
        plt.close("all")

        # Execute with output redirection
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Use asyncio to run with timeout
            await asyncio.wait_for(
                _run_code_in_thread(code, execution_env), timeout=timeout
            )

        # Capture execution time
        execution_time = time.time() - start_time

        # Get outputs
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()

        result.update(
            {
                "success": True,
                "output": stdout_content,
                "execution_time": execution_time,
            }
        )

        # Include stderr as warnings if present but execution succeeded
        if stderr_content.strip():
            result["warnings"] = stderr_content

        logger.debug(f"Code executed successfully in {execution_time:.2f}s")

    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        result.update(
            {
                "success": False,
                "timeout": True,
                "error": f"Code execution timed out after {timeout:.1f} seconds",
                "execution_time": execution_time,
            }
        )
        logger.warning(f"Code execution timed out after {timeout}s")

    except Exception as e:
        execution_time = time.time() - start_time
        error_output = stderr_buffer.getvalue()

        result.update(
            {
                "success": False,
                "error": str(e),
                "output": stdout_buffer.getvalue(),
                "execution_time": execution_time,
            }
        )

        # Include traceback for debugging
        if error_output:
            result["error"] = f"{str(e)}\n\nDetails:\n{error_output}"

        logger.error(f"Code execution failed after {execution_time:.2f}s: {e}")

    finally:
        # Always close plot resources
        plt.close("all")

    return result


async def _run_code_in_thread(code: str, execution_env: Dict[str, Any]):
    """
    Run code in thread pool to allow for timeout handling

    Args:
        code: Python code to execute
        execution_env: Execution environment
    """
    loop = asyncio.get_event_loop()

    def _execute():
        # Create a copy of the environment to avoid mutations
        env_copy = execution_env.copy()

        # Execute the code
        exec(code, {"__builtins__": {}}, env_copy)

    # Run in thread pool
    await loop.run_in_executor(None, _execute)


def _process_visualizations() -> List[str]:
    """
    Process any matplotlib figures created during execution

    Returns:
        List of base64-encoded plot images
    """
    plots = []

    try:
        # Get all current figure numbers
        fig_nums = plt.get_fignums()

        for fig_num in fig_nums:
            try:
                fig = plt.figure(fig_num)

                # Configure figure for web display
                fig.set_size_inches(10, 6)
                fig.tight_layout(pad=3.0)

                # Add title if not present
                try:
                    if not fig._suptitle:  # type: ignore
                        fig.suptitle("Analysis Result", fontsize=14, y=0.98)
                except AttributeError:
                    fig.suptitle("Analysis Result", fontsize=14, y=0.98)

                # Save figure to bytes buffer
                img_buffer = io.BytesIO()
                fig.savefig(
                    img_buffer,
                    format="png",
                    dpi=100,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )
                img_buffer.seek(0)

                # Convert to base64
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                plots.append(img_base64)

                # Close the figure to free memory
                plt.close(fig)

            except Exception as e:
                logger.warning(f"Failed to process figure {fig_num}: {e}")
                continue

        logger.debug(f"Processed {len(plots)} visualization(s)")

    except Exception as e:
        logger.error(f"Error processing visualizations: {e}")

    finally:
        # Ensure all figures are closed
        plt.close("all")

    return plots


def _create_fallback_execution_result(error_msg: str) -> Dict[str, Any]:
    """
    Create fallback execution result when code execution fails completely

    Args:
        error_msg: Error message

    Returns:
        Fallback execution result
    """
    return {
        "success": False,
        "output": f"Code execution failed: {error_msg}",
        "error": error_msg,
        "timeout": False,
        "execution_time": 0.0,
        "plots": [],
    }


def _estimate_resource_usage(code: str) -> Dict[str, float]:
    """
    Estimate resource usage of code (simple heuristics)

    Args:
        code: Python code to analyze

    Returns:
        Resource usage estimates
    """
    # Simple heuristics for resource estimation
    lines = len(code.split("\n"))

    # Count potentially expensive operations
    expensive_ops = 0
    expensive_patterns = [
        r"\.plot\(",
        r"plt\.",
        r"sns\.",
        r"\.groupby\(",
        r"\.merge\(",
        r"\.join\(",
        r"for\s+.*in\s+",
        r"while\s+",
        r"\.corr\(",
        r"\.cov\(",
    ]

    for pattern in expensive_patterns:
        import re

        expensive_ops += len(re.findall(pattern, code))

    return {
        "estimated_memory_mb": min(100 + lines * 2 + expensive_ops * 10, 500),
        "estimated_cpu_seconds": min(1 + lines * 0.1 + expensive_ops * 2, 30),
        "complexity_score": min((lines + expensive_ops * 5) / 10, 10.0),
    }
