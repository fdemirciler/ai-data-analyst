"""Comprehensive logging utilities for tracking agent workflow execution."""

import logging
import sys
from typing import Any, Dict
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_workflow.log", mode="a", encoding="utf-8"),
    ],
)

# Create specialized loggers
orchestrator_logger = logging.getLogger("agent.orchestrator")
llm_logger = logging.getLogger("agent.llm")
sandbox_logger = logging.getLogger("agent.sandbox")
safety_logger = logging.getLogger("agent.safety")
chat_logger = logging.getLogger("routes.chat")
session_logger = logging.getLogger("session")


def log_workflow_start(session_id: str, message: str, dataset_info: Dict[str, Any]):
    """Log the start of a new workflow execution."""
    chat_logger.info(f"WORKFLOW START - Session: {session_id}")
    chat_logger.info(f"User message: {message}")
    chat_logger.info(
        f"Dataset: {dataset_info.get('dataset', {}).get('rows', '?')} rows x {dataset_info.get('dataset', {}).get('columns', '?')} cols"
    )


def log_orchestrator_stage(stage: str, details: str = ""):
    """Log orchestrator stage progression."""
    orchestrator_logger.info(f"STAGE: {stage.upper()} {details}")


def log_llm_generation(prompt_preview: str, code_length: int, success: bool):
    """Log LLM code generation attempts."""
    llm_logger.info(f"LLM GENERATION - Prompt preview: {prompt_preview[:100]}...")
    llm_logger.info(f"Generated code length: {code_length} chars, Success: {success}")


def log_safety_check(code_preview: str, violations: list):
    """Log safety validation results."""
    safety_logger.info(f"SAFETY CHECK - Code preview: {code_preview[:100]}...")
    if violations:
        safety_logger.warning(f"SAFETY VIOLATIONS: {violations}")
    else:
        safety_logger.info("SAFETY CHECK PASSED")


def log_sandbox_execution(exec_result: Dict[str, Any]):
    """Log sandbox execution details."""
    status = exec_result.get("status", "unknown")
    logs = exec_result.get("logs", "")

    sandbox_logger.info(f"SANDBOX EXECUTION - Status: {status}")

    if status == "skipped":
        sandbox_logger.warning(
            "SANDBOX EXECUTION SKIPPED - Check enable_docker_sandbox setting"
        )
    elif status == "ok":
        sandbox_logger.info(f"EXECUTION SUCCESS - Output length: {len(logs)} chars")
        if logs:
            sandbox_logger.info(f"Execution output preview: {logs[:200]}...")
    elif status == "error":
        sandbox_logger.error(f"EXECUTION FAILED - Error: {logs[:500]}")
    else:
        sandbox_logger.warning(f"UNKNOWN STATUS: {status}")


def log_final_decision(answer_source: str, answer_length: int):
    """Log the final answer generation decision."""
    orchestrator_logger.info(
        f"FINAL ANSWER - Source: {answer_source}, Length: {answer_length} chars"
    )


def log_config_state():
    """Log current configuration state."""
    from .config import settings

    orchestrator_logger.info(f"CONFIG STATE:")
    orchestrator_logger.info(f"   LLM Enabled: {settings.enable_llm}")
    orchestrator_logger.info(f"   Gemini Key Present: {bool(settings.gemini_api_key)}")
    orchestrator_logger.info(f"   Docker Sandbox: {settings.enable_docker_sandbox}")
    orchestrator_logger.info(f"   Environment: {settings.environment}")


def log_error(component: str, error: Exception, context: str = ""):
    """Log errors with context."""
    logger = logging.getLogger(f"agent.{component}")
    logger.error(f"ERROR in {component}: {str(error)} | Context: {context}")


def log_artifact_storage(
    session_id: str, artifact_index: int, artifact_summary: Dict[str, Any]
):
    """Log artifact storage."""
    session_logger.info(
        f"ARTIFACT STORED - Session: {session_id}, Index: {artifact_index}"
    )
    session_logger.info(f"   Code length: {len(artifact_summary.get('code', ''))}")
    session_logger.info(
        f"   Exec status: {artifact_summary.get('exec', {}).get('status', 'unknown')}"
    )
    session_logger.info(f"   Violations: {len(artifact_summary.get('violations', []))}")
