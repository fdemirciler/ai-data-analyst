"""
Main LangGraph Workflow Definition

This module defines the complete agentic data analysis workflow using LangGraph.
The workflow consists of the following nodes:
1. data_processing - Load and clean data
2. data_profiling - Analyze data characteristics
3. query_analysis - Understand user intent
4. code_generation - Generate Python analysis code
5. code_execution - Execute code safely
6. response_formatting - Format results for user
7. error_handling - Handle failures and retries

The workflow includes conditional routing, error handling, and state management.
"""

import logging
from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import WorkflowState
from .nodes.data_processing import data_processing_node
from .nodes.data_profiling import data_profiling_node
from .nodes.query_analysis import query_analysis_node
from .nodes.code_generation import code_generation_node
from .nodes.code_execution import code_execution_node
from .nodes.response_formatting import response_formatting_node
from .nodes.error_handling import error_handling_node

logger = logging.getLogger(__name__)


def create_workflow_graph() -> StateGraph:
    """
    Create the complete LangGraph workflow for agentic data analysis.

    Returns:
        Configured StateGraph ready for execution
    """

    # Create the graph with our WorkflowState
    workflow = StateGraph(WorkflowState)

    # Add all workflow nodes
    workflow.add_node("data_processing", data_processing_node)
    workflow.add_node("data_profiling", data_profiling_node)
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("code_execution", code_execution_node)
    workflow.add_node("response_formatting", response_formatting_node)
    workflow.add_node("error_handling", error_handling_node)

    # Set entry point
    workflow.set_entry_point("data_processing")

    # Define the main workflow path
    workflow.add_edge("data_processing", "data_profiling")
    workflow.add_edge("data_profiling", "query_analysis")
    workflow.add_edge("query_analysis", "code_generation")
    workflow.add_edge("code_generation", "code_execution")
    workflow.add_edge("code_execution", "response_formatting")

    # Add conditional routing for error handling
    workflow.add_conditional_edges(
        "data_processing",
        _should_handle_error,
        {"error": "error_handling", "continue": "data_profiling"},
    )

    workflow.add_conditional_edges(
        "data_profiling",
        _should_handle_error,
        {"error": "error_handling", "continue": "query_analysis"},
    )

    workflow.add_conditional_edges(
        "query_analysis",
        _should_handle_error,
        {"error": "error_handling", "continue": "code_generation"},
    )

    workflow.add_conditional_edges(
        "code_generation",
        _should_handle_error,
        {"error": "error_handling", "continue": "code_execution"},
    )

    workflow.add_conditional_edges(
        "code_execution",
        _should_handle_error,
        {"error": "error_handling", "continue": "response_formatting"},
    )

    workflow.add_conditional_edges(
        "response_formatting",
        _should_handle_error,
        {"error": "error_handling", "continue": END},
    )

    # Error handling can either retry or end
    workflow.add_conditional_edges(
        "error_handling",
        _determine_error_outcome,
        {
            "retry_data_processing": "data_processing",
            "retry_data_profiling": "data_profiling",
            "retry_query_analysis": "query_analysis",
            "retry_code_generation": "code_generation",
            "retry_code_execution": "code_execution",
            "retry_response_formatting": "response_formatting",
            "end": END,
        },
    )

    # Final transitions to END
    workflow.add_edge("response_formatting", END)

    return workflow


def _should_handle_error(state: WorkflowState) -> Literal["error", "continue"]:
    """
    Determine if we should route to error handling based on the current state.

    Args:
        state: Current workflow state

    Returns:
        "error" if there was a failure, "continue" otherwise
    """
    # Check if the current node has completed successfully
    current_node = state.get("current_node")
    if not current_node:
        return "continue"

    node_results = state.get("node_results", {})
    current_result = node_results.get(current_node, {})

    # If the node failed, go to error handling
    if not current_result.get("success", True):
        logger.info(f"Routing to error handling due to failure in {current_node}")
        return "error"

    return "continue"


def _determine_error_outcome(state: WorkflowState) -> str:
    """
    Determine what to do after error handling based on recovery results.

    Args:
        state: Current workflow state with error handling results

    Returns:
        Next node to route to or "end" to terminate
    """
    # Check error handling results
    error_handling_results = state.get("node_results", {}).get("error_handling", {})

    recovery_successful = error_handling_results.get("recovery_successful", False)

    if not recovery_successful:
        logger.info("Error recovery failed, ending workflow")
        return "end"

    # Determine which node to retry based on recovery strategy
    recovery_details = error_handling_results.get("recovery_details", {})
    next_node = recovery_details.get("next_node")

    if next_node:
        logger.info(f"Error recovery successful, retrying from {next_node}")
        return f"retry_{next_node}"

    # Default fallback
    logger.warning(
        "Error recovery completed but no next node specified, ending workflow"
    )
    return "end"


def create_compiled_workflow():
    """
    Create and compile the complete workflow with memory checkpointing.

    Returns:
        Compiled workflow ready for execution
    """
    workflow_graph = create_workflow_graph()

    # Add memory checkpointing for state persistence
    memory = MemorySaver()

    # Compile the workflow
    compiled_workflow = workflow_graph.compile(checkpointer=memory)

    logger.info("Workflow compiled successfully with memory checkpointing")

    return compiled_workflow


# Create the compiled workflow instance that will be used by the API
workflow = create_compiled_workflow()


async def execute_workflow(
    initial_state: WorkflowState, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute the complete workflow with the given initial state.

    Args:
        initial_state: Starting state for the workflow
        config: Optional configuration for the workflow execution

    Returns:
        Final workflow state after completion
    """
    if config is None:
        config = {"configurable": {"thread_id": initial_state["session_id"]}}

    try:
        logger.info(
            f"Starting workflow execution for session {initial_state['session_id']}"
        )

        # Execute the workflow
        final_state = None
        async for state in workflow.astream(initial_state, config=config):  # type: ignore
            final_state = state

            # Log progress
            current_node = (
                state.get("current_node") if isinstance(state, dict) else None
            )
            if current_node:
                logger.debug(f"Completed node: {current_node}")

        if final_state is None:
            raise RuntimeError("Workflow execution failed - no final state returned")

        logger.info(
            f"Workflow execution completed for session {initial_state['session_id']}"
        )
        return dict(final_state) if final_state else {}

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)

        # Return a failed state
        failed_state = dict(initial_state)
        failed_state["workflow_status"] = "failed"
        failed_state["error_message"] = str(e)
        failed_state["final_response"] = (
            "I encountered an unexpected error while processing your request. Please try again."
        )
        failed_state["response_elements"] = [
            {
                "type": "error",
                "content": "An unexpected system error occurred. Please try your request again.",
                "title": "System Error",
            }
        ]

        return failed_state


async def get_workflow_status(session_id: str) -> Dict[str, Any]:
    """
    Get the current status of a workflow execution.

    Args:
        session_id: Session ID to check status for

    Returns:
        Dictionary with workflow status information
    """
    try:
        config = {"configurable": {"thread_id": session_id}}

        # Get the current state from checkpointer
        current_state = await workflow.aget_state(config)  # type: ignore

        if not current_state:
            return {
                "status": "not_found",
                "message": "No workflow found for this session",
            }

        values = current_state.values

        return {
            "status": values.get("workflow_status", "unknown"),
            "current_node": values.get("current_node"),
            "completed_nodes": list(values.get("node_results", {}).keys()),
            "progress_percentage": _calculate_progress(dict(values) if values else {}),
            "error_message": values.get("error_message"),
            "has_final_response": bool(values.get("final_response")),
        }

    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        return {"status": "error", "message": f"Failed to retrieve status: {str(e)}"}


def _calculate_progress(state: Dict[str, Any]) -> int:
    """Calculate workflow progress percentage based on completed nodes"""
    total_nodes = 6  # data_processing, data_profiling, query_analysis, code_generation, code_execution, response_formatting
    completed_nodes = len(state.get("node_results", {}))

    # Don't count error_handling in progress calculation
    if "error_handling" in state.get("node_results", {}):
        completed_nodes -= 1

    return min(100, int((completed_nodes / total_nodes) * 100))
