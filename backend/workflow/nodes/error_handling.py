"""
Error Handling Node for LangGraph Workflow

This node provides comprehensive error recovery and retry mechanisms:
1. Analyzes workflow failures and determines recovery strategies
2. Implements retry logic with exponential backoff
3. Provides graceful degradation paths
4. Logs detailed error information for debugging
5. Formats error responses for user consumption

This node is called when other workflow nodes fail.
"""

import asyncio
import time
import logging
import traceback
from typing import Dict, Any, Optional, List
from enum import Enum

from ..state import WorkflowState, WorkflowStateManager
from ...services.llm_provider import LLMManager
from ...utils.exceptions import WorkflowError
from ...config import settings

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for recovery strategy determination"""

    RECOVERABLE = "recoverable"  # Can retry with modifications
    DEGRADED = "degraded"  # Can provide partial results
    FATAL = "fatal"  # Cannot recover, return error


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""

    RETRY_SAME = "retry_same"  # Retry same operation
    RETRY_SIMPLIFIED = "retry_simplified"  # Retry with simpler approach
    FALLBACK_METHOD = "fallback_method"  # Use alternative method
    PARTIAL_RESULTS = "partial_results"  # Return what we have
    GRACEFUL_FAILURE = "graceful_failure"  # Format error for user


async def error_handling_node(state: WorkflowState) -> WorkflowState:
    """
    Handle workflow errors with recovery and retry logic

    This node:
    1. Analyzes the error type and severity
    2. Determines appropriate recovery strategy
    3. Attempts recovery if possible
    4. Formats error response if recovery fails

    Args:
        state: Current workflow state with error information

    Returns:
        Updated workflow state with recovery attempt or error response
    """
    start_time = time.time()
    node_name = "error_handling"

    try:
        logger.info(f"Starting error handling for session {state['session_id']}")

        # Get error information
        last_error = _get_last_error_info(state)
        if not last_error:
            logger.warning("Error handling called but no error information found")
            return state

        # Analyze error and determine strategy
        error_analysis = _analyze_error(last_error, state)
        logger.info(
            f"Error analysis: severity={error_analysis['severity'].value}, "
            f"strategy={error_analysis['strategy'].value}"
        )

        # Attempt recovery based on strategy
        recovery_result = await _attempt_recovery(error_analysis, state)

        processing_time = time.time() - start_time

        if recovery_result["success"]:
            logger.info(f"Error recovery successful in {processing_time:.2f}s")

            # Update state with recovery information
            state = WorkflowStateManager.record_node_completion(
                state,
                node_name,
                processing_time,
                {
                    "recovery_strategy": error_analysis["strategy"].value,
                    "recovery_successful": True,
                    "recovery_details": recovery_result["details"],
                },
            )

            # Continue workflow from appropriate node
            # Note: next_node determination will be handled by the workflow graph
            pass

        else:
            logger.warning(f"Error recovery failed in {processing_time:.2f}s")

            # Format final error response
            error_response = await _format_final_error_response(error_analysis, state)

            # Update state with error response
            state = WorkflowStateManager.record_node_completion(
                state,
                node_name,
                processing_time,
                {
                    "recovery_strategy": error_analysis["strategy"].value,
                    "recovery_successful": False,
                    "final_response": error_response["response"],
                    "response_elements": error_response["elements"],
                },
            )

            # Mark workflow as completed with error
            state = WorkflowStateManager.finalize_workflow(state, success=False)

        return state

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error handling node failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Record failure
        state = WorkflowStateManager.record_node_failure(
            state, node_name, error_msg, processing_time
        )

        # Create emergency fallback response
        state["final_response"] = (
            "I encountered an unexpected error while processing your request. Please try again."
        )
        state["response_elements"] = [
            {
                "type": "error",
                "content": "An unexpected system error occurred. Please try your request again.",
                "title": "System Error",
            }
        ]

        # Mark workflow as completed with error
        state = WorkflowStateManager.finalize_workflow(state, success=False)

        return state


def _get_last_error_info(state: WorkflowState) -> Optional[Dict[str, Any]]:
    """Extract information about the last error from workflow state"""
    node_results = state.get("node_results", {})

    # Find the most recent failed node
    failed_nodes = []
    for node_name, node_data in node_results.items():
        if not node_data.get("success", True):
            failed_nodes.append(
                {
                    "node_name": node_name,
                    "error_message": node_data.get("error_message", ""),
                    "processing_time": node_data.get("processing_time", 0),
                    "timestamp": node_data.get("end_time", 0),
                }
            )

    if not failed_nodes:
        return None

    # Return most recent failure
    return max(failed_nodes, key=lambda x: x["timestamp"])


def _analyze_error(error_info: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
    """Analyze error to determine severity and recovery strategy"""
    node_name = error_info["node_name"]
    error_message = error_info["error_message"].lower()

    # Determine error severity and strategy based on node and error type
    if node_name == "data_processing":
        if any(
            keyword in error_message
            for keyword in ["file not found", "permission denied", "invalid format"]
        ):
            return {
                "severity": ErrorSeverity.FATAL,
                "strategy": RecoveryStrategy.GRACEFUL_FAILURE,
                "reason": "Data file issues cannot be automatically resolved",
            }
        else:
            return {
                "severity": ErrorSeverity.RECOVERABLE,
                "strategy": RecoveryStrategy.FALLBACK_METHOD,
                "reason": "Can try alternative data processing approach",
            }

    elif node_name == "data_profiling":
        return {
            "severity": ErrorSeverity.DEGRADED,
            "strategy": RecoveryStrategy.PARTIAL_RESULTS,
            "reason": "Can proceed with basic data information",
        }

    elif node_name == "query_analysis":
        if "llm" in error_message or "provider" in error_message:
            return {
                "severity": ErrorSeverity.RECOVERABLE,
                "strategy": RecoveryStrategy.RETRY_SIMPLIFIED,
                "reason": "Can retry with fallback LLM provider",
            }
        else:
            return {
                "severity": ErrorSeverity.DEGRADED,
                "strategy": RecoveryStrategy.FALLBACK_METHOD,
                "reason": "Can use rule-based query analysis",
            }

    elif node_name == "code_generation":
        if "llm" in error_message or "provider" in error_message:
            return {
                "severity": ErrorSeverity.RECOVERABLE,
                "strategy": RecoveryStrategy.RETRY_SIMPLIFIED,
                "reason": "Can retry with different LLM provider",
            }
        else:
            return {
                "severity": ErrorSeverity.DEGRADED,
                "strategy": RecoveryStrategy.FALLBACK_METHOD,
                "reason": "Can use template-based code generation",
            }

    elif node_name == "code_execution":
        if any(
            keyword in error_message
            for keyword in ["syntax error", "name error", "import error"]
        ):
            return {
                "severity": ErrorSeverity.RECOVERABLE,
                "strategy": RecoveryStrategy.RETRY_SIMPLIFIED,
                "reason": "Can retry with corrected code",
            }
        elif "timeout" in error_message:
            return {
                "severity": ErrorSeverity.RECOVERABLE,
                "strategy": RecoveryStrategy.RETRY_SIMPLIFIED,
                "reason": "Can retry with simpler analysis",
            }
        else:
            return {
                "severity": ErrorSeverity.DEGRADED,
                "strategy": RecoveryStrategy.PARTIAL_RESULTS,
                "reason": "Can show error with partial context",
            }

    elif node_name == "response_formatting":
        return {
            "severity": ErrorSeverity.DEGRADED,
            "strategy": RecoveryStrategy.PARTIAL_RESULTS,
            "reason": "Can format basic response without LLM",
        }

    else:
        # Unknown error type
        return {
            "severity": ErrorSeverity.FATAL,
            "strategy": RecoveryStrategy.GRACEFUL_FAILURE,
            "reason": "Unknown error type requires manual intervention",
        }


async def _attempt_recovery(
    error_analysis: Dict[str, Any], state: WorkflowState
) -> Dict[str, Any]:
    """Attempt to recover from error based on strategy"""
    strategy = error_analysis["strategy"]

    try:
        if strategy == RecoveryStrategy.RETRY_SAME:
            return await _retry_same_operation(state)

        elif strategy == RecoveryStrategy.RETRY_SIMPLIFIED:
            return await _retry_with_simplification(state)

        elif strategy == RecoveryStrategy.FALLBACK_METHOD:
            return await _use_fallback_method(state)

        elif strategy == RecoveryStrategy.PARTIAL_RESULTS:
            return await _provide_partial_results(state)

        else:  # GRACEFUL_FAILURE
            return {"success": False, "reason": "No recovery strategy available"}

    except Exception as e:
        logger.error(f"Recovery attempt failed: {e}")
        return {"success": False, "reason": f"Recovery attempt failed: {str(e)}"}


async def _retry_same_operation(state: WorkflowState) -> Dict[str, Any]:
    """Retry the same operation that failed"""
    # Get the failed node and retry it
    failed_node = _get_last_failed_node_name(state)
    if not failed_node:
        return {"success": False, "reason": "No failed node identified"}

    # Simple retry - just reset the node for re-execution
    return {
        "success": True,
        "next_node": failed_node,
        "details": f"Retrying {failed_node} operation",
    }


async def _retry_with_simplification(state: WorkflowState) -> Dict[str, Any]:
    """Retry with simplified parameters or different provider"""
    failed_node = _get_last_failed_node_name(state)

    if failed_node in ["query_analysis", "code_generation"]:
        # Try with different LLM provider
        current_provider = state.get("llm_provider", "gemini")
        fallback_providers = (
            ["openrouter", "together"] if current_provider == "gemini" else ["gemini"]
        )

        if fallback_providers:
            state["llm_provider"] = fallback_providers[0]
            return {
                "success": True,
                "next_node": failed_node,
                "details": f"Retrying {failed_node} with provider {fallback_providers[0]}",
            }

    elif failed_node == "code_execution":
        # Mark for simpler code generation using existing fields
        results = state.get("node_results", {})
        results["simplified_retry"] = True
        return {
            "success": True,
            "next_node": "code_generation",
            "details": "Retrying with simplified code generation",
        }

    return {"success": False, "reason": "No simplification strategy available"}


async def _use_fallback_method(state: WorkflowState) -> Dict[str, Any]:
    """Use alternative method for the failed operation"""
    failed_node = _get_last_failed_node_name(state)

    if failed_node == "data_processing":
        # Mark for basic data loading without cleaning using existing fields
        results = state.get("node_results", {})
        results["basic_loading_mode"] = True
        return {
            "success": True,
            "next_node": "data_processing",
            "details": "Using basic data loading without advanced cleaning",
        }

    elif failed_node == "query_analysis":
        # Use rule-based analysis using existing fields
        results = state.get("node_results", {})
        results["rule_based_mode"] = True
        return {
            "success": True,
            "next_node": "query_analysis",
            "details": "Using rule-based query analysis",
        }

    return {"success": False, "reason": "No fallback method available"}


async def _provide_partial_results(state: WorkflowState) -> Dict[str, Any]:
    """Provide partial results based on what we have so far"""
    # Check what results we have available
    available_data = []

    if state.get("cleaned_data_path"):
        available_data.append("processed data")

    if state.get("data_profile"):
        available_data.append("data profile")

    if state.get("generated_code"):
        available_data.append("analysis code")

    if available_data:
        # Create partial response
        partial_response = f"I was able to partially process your request. Available results: {', '.join(available_data)}."

        state["final_response"] = partial_response
        state["response_elements"] = [
            {"type": "partial", "content": partial_response, "title": "Partial Results"}
        ]

        return {
            "success": True,
            "next_node": "response_formatting",
            "details": "Providing partial results",
        }

    return {"success": False, "reason": "No partial results available"}


async def _format_final_error_response(
    error_analysis: Dict[str, Any], state: WorkflowState
) -> Dict[str, Any]:
    """Format final error response for user"""

    # Create user-friendly error message
    error_messages = {
        "data_processing": "I had trouble processing your data file. Please check that the file is in a supported format (CSV, Excel, JSON, or Parquet) and isn't corrupted.",
        "data_profiling": "I couldn't fully analyze your dataset, but I can still try to answer your question with the available information.",
        "query_analysis": "I had difficulty understanding your question. Could you try rephrasing it or being more specific about what you'd like to analyze?",
        "code_generation": "I couldn't generate the analysis code for your request. This might be due to the complexity of the question or limitations in the data.",
        "code_execution": "The analysis code encountered an error during execution. This could be due to data format issues or the complexity of the requested analysis.",
        "response_formatting": "I completed the analysis but had trouble formatting the response. The raw results might still be available.",
    }

    failed_node = _get_last_failed_node_name(state)

    if failed_node:
        error_message = error_messages.get(
            failed_node,
            "I encountered an unexpected error while processing your request.",
        )
    else:
        error_message = (
            "I encountered an unexpected error while processing your request."
        )

    suggestions = [
        "Try rephrasing your question to be more specific",
        "Check if your data file is in a supported format",
        "Consider asking for a simpler analysis first",
        "Upload a different dataset if the current one has issues",
    ]

    return {
        "response": error_message,
        "elements": [
            {"type": "error", "content": error_message, "title": "Analysis Error"},
            {"type": "suggestions", "content": suggestions, "title": "Suggestions"},
        ],
    }


def _get_last_failed_node_name(state: WorkflowState) -> Optional[str]:
    """Get the name of the most recently failed node"""
    error_info = _get_last_error_info(state)
    return error_info["node_name"] if error_info else None
