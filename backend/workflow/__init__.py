"""
LangGraph Workflow Package

This package contains the complete agentic data analysis workflow implementation
using LangGraph for orchestration and state management.

Components:
- state: WorkflowState TypedDict and state management utilities
- nodes: Individual workflow nodes (data processing, profiling, analysis, etc.)
- graph: Main workflow graph definition and execution logic

Usage:
    from backend.workflow import workflow, execute_workflow

    initial_state = {...}
    final_state = await execute_workflow(initial_state)
"""

from .graph import workflow, execute_workflow, get_workflow_status
from .state import WorkflowState, WorkflowStateManager

__all__ = [
    "workflow",
    "execute_workflow",
    "get_workflow_status",
    "WorkflowState",
    "WorkflowStateManager",
]
