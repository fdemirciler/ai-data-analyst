"""
Response Formatting Node for LangGraph Workflow

This node handles final response preparation:
1. Formats execution results for frontend consumption
2. Generates natural language explanations using LLM
3. Structures response elements (text, charts, tables)
4. Creates rich conversational responses
5. Handles error message formatting

This is where technical results become user-friendly insights.
"""

import asyncio
import time
import logging
import json
import re
from typing import Dict, Any, List, Optional

from ..state import WorkflowState, WorkflowStateManager
from ...services.llm_provider import LLMManager
from ...utils.exceptions import WorkflowError
from ...config import settings

logger = logging.getLogger(__name__)


async def response_formatting_node(state: WorkflowState) -> WorkflowState:
    """
    Format analysis results into user-friendly response

    This node:
    1. Analyzes execution results and outputs
    2. Generates natural language explanations
    3. Structures response elements for frontend
    4. Creates rich conversational responses
    5. Handles error formatting if needed

    Args:
        state: Current workflow state with execution results

    Returns:
        Updated workflow state with formatted response
    """
    start_time = time.time()
    node_name = "response_formatting"

    try:
        logger.info(f"Starting response formatting for session {state['session_id']}")

        # Initialize LLM manager
        llm_manager = LLMManager(settings)

        # Determine if execution was successful
        execution_result = state.get("execution_result") or {}
        execution_success = execution_result.get("success", False)

        if execution_success:
            # Format successful analysis response
            response_data = await _format_successful_response(state, llm_manager)
        else:
            # Format error response
            response_data = await _format_error_response(state, llm_manager)

        # Processing metrics
        processing_time = time.time() - start_time

        logger.info(
            f"Response formatting completed in {processing_time:.2f}s - "
            f"success: {execution_success}"
        )

        # Update workflow state
        results = {
            "final_response": response_data["response"],
            "response_elements": response_data["elements"],
            "insights": response_data["insights"],
        }

        # Record successful completion
        state = WorkflowStateManager.record_node_completion(
            state, node_name, processing_time, results
        )

        # Mark workflow as completed
        state = WorkflowStateManager.finalize_workflow(state, success=True)

        return state

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Response formatting failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Record failure
        state = WorkflowStateManager.record_node_failure(
            state, node_name, error_msg, processing_time
        )

        raise WorkflowError(f"Response formatting node failed: {str(e)}")


async def _format_successful_response(
    state: WorkflowState, llm_manager: LLMManager
) -> Dict[str, Any]:
    """
    Format successful analysis response with LLM interpretation
    """
    execution_result = state.get("execution_result") or {}
    execution_output = execution_result.get("output", "")
    execution_plots = state.get("execution_plots", [])

    # Prepare context for LLM interpretation
    interpretation_context = _prepare_interpretation_context(state)

    # Generate natural language interpretation
    logger.debug("Generating natural language interpretation...")
    interpretation_prompt = _create_interpretation_prompt(interpretation_context)

    llm_response = await llm_manager.generate(
        interpretation_prompt,
        provider=state["llm_provider"],
        temperature=0.2,
        max_tokens=1000,
    )

    # Parse interpretation response
    interpretation_data = _parse_interpretation_response(llm_response.content)

    # Structure response elements
    response_elements = []

    # Add text interpretation
    response_elements.append(
        {
            "type": "text",
            "content": interpretation_data["main_response"],
            "title": "Analysis Results",
        }
    )

    # Add key insights if available
    if interpretation_data["insights"]:
        response_elements.append(
            {
                "type": "insights",
                "content": interpretation_data["insights"],
                "title": "Key Insights",
            }
        )

    # Add technical output if meaningful
    if execution_output.strip() and len(execution_output.strip()) > 20:
        formatted_output = _format_technical_output(execution_output)
        if formatted_output:
            response_elements.append(
                {
                    "type": "code_output",
                    "content": formatted_output,
                    "title": "Technical Results",
                }
            )

    # Add visualizations
    for i, plot_data in enumerate(execution_plots):
        response_elements.append(
            {
                "type": "visualization",
                "content": plot_data,
                "title": f"Visualization {i + 1}",
                "format": "base64_png",
            }
        )

    # Add recommendations if available
    if interpretation_data["recommendations"]:
        response_elements.append(
            {
                "type": "recommendations",
                "content": interpretation_data["recommendations"],
                "title": "Recommendations",
            }
        )

    return {
        "response": interpretation_data["main_response"],
        "elements": response_elements,
        "insights": interpretation_data["insights"],
    }


async def _format_error_response(
    state: WorkflowState, llm_manager: LLMManager
) -> Dict[str, Any]:
    """
    Format error response with helpful guidance
    """
    execution_result = state.get("execution_result") or {}
    error_message = execution_result.get("error", "Unknown error occurred")

    # Generate helpful error interpretation
    error_context = {
        "original_query": state["query"],
        "error_message": error_message,
        "generated_code": state.get("generated_code", ""),
        "dataset_info": {
            "shape": (state.get("cleaned_data_info") or {}).get("shape", [0, 0]),
            "columns": (state.get("cleaned_data_info") or {}).get("columns", []),
        },
    }

    error_prompt = _create_error_interpretation_prompt(error_context)

    try:
        llm_response = await llm_manager.generate(
            error_prompt,
            provider=state["llm_provider"],
            temperature=0.1,
            max_tokens=800,
        )

        error_interpretation = _parse_error_response(llm_response.content)

    except Exception as e:
        logger.warning(f"Failed to generate error interpretation: {e}")
        error_interpretation = _create_fallback_error_response(error_message)

    # Structure error response elements
    response_elements = [
        {
            "type": "error",
            "content": error_interpretation["explanation"],
            "title": "Analysis Error",
        }
    ]

    if error_interpretation["suggestions"]:
        response_elements.append(
            {
                "type": "suggestions",
                "content": error_interpretation["suggestions"],
                "title": "Suggestions",
            }
        )

    # Add any partial results if available
    execution_output = execution_result.get("output", "")
    if execution_output.strip():
        response_elements.append(
            {
                "type": "partial_output",
                "content": execution_output,
                "title": "Partial Results",
            }
        )

    return {
        "response": error_interpretation["explanation"],
        "elements": response_elements,
        "insights": [],
    }


def _prepare_interpretation_context(state: WorkflowState) -> Dict[str, Any]:
    """Prepare context for LLM interpretation of results"""
    execution_result = state.get("execution_result") or {}
    generated_code = state.get("generated_code") or ""

    return {
        "original_query": state["query"],
        "user_intent": state.get("user_intent", "unknown"),
        "analysis_type": state.get("analysis_type", "unknown"),
        "execution_output": execution_result.get("output", ""),
        "execution_time": execution_result.get("execution_time", 0),
        "has_plots": len(state.get("execution_plots", [])) > 0,
        "plot_count": len(state.get("execution_plots", [])),
        "dataset_info": {
            "filename": state.get("original_filename", "dataset"),
            "shape": (state.get("cleaned_data_info") or {}).get("shape", [0, 0]),
            "columns": (state.get("cleaned_data_info") or {}).get("columns", []),
        },
        "generated_code": generated_code.split("\n")[:10] if generated_code else [],
        "conversation_history": state.get("conversation_history", [])[-2:],
    }


def _create_interpretation_prompt(context: Dict[str, Any]) -> str:
    """Create prompt for LLM interpretation of successful results"""
    output_preview = context["execution_output"][:1000] + (
        "..." if len(context["execution_output"]) > 1000 else ""
    )

    prompt = f"""You are an expert data analyst interpreting analysis results for a user. Provide a clear, insightful response.

CURRENT USER QUERY: "{context['original_query']}"

ANALYSIS DETAILS:
- Intent: {context['user_intent']}
- Type: {context['analysis_type']}
- Dataset: {context['dataset_info']['filename']} ({context['dataset_info']['shape'][0]} rows × {context['dataset_info']['shape'][1]} columns)
- Execution time: {context['execution_time']:.2f} seconds
- Visualizations created: {context['plot_count']}

ANALYSIS OUTPUT:
{output_preview}

Respond in JSON format:
```json
{{
    "main_response": "Clear, conversational response directly answering the user's question...",
    "insights": [
        "Key insight 1 discovered in the analysis",
        "Key insight 2 that might surprise or inform the user"
    ],
    "recommendations": [
        "Actionable recommendation based on findings",
        "Suggested follow-up analysis"
    ]
}}
```

Make your response conversational and specific to the actual results."""

    return prompt


def _create_error_interpretation_prompt(context: Dict[str, Any]) -> str:
    """Create prompt for LLM interpretation of errors"""
    prompt = f"""You are helping a user understand why their data analysis failed.

USER QUERY: "{context['original_query']}"
ERROR MESSAGE: {context['error_message']}

DATASET INFO:
- Shape: {context['dataset_info']['shape'][0]} rows × {context['dataset_info']['shape'][1]} columns
- Columns: {', '.join(context['dataset_info']['columns'][:10])}

Provide a helpful response in JSON format:
```json
{{
    "explanation": "Clear, friendly explanation of what went wrong and why",
    "suggestions": [
        "Specific suggestion 1 for how to rephrase the question",
        "Suggestion 2 for alternative analysis approach"
    ]
}}
```

Focus on being helpful and encouraging."""

    return prompt


def _parse_interpretation_response(llm_response: str) -> Dict[str, Any]:
    """Parse LLM interpretation response"""
    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = llm_response.strip()

        data = json.loads(json_str)

        return {
            "main_response": data.get(
                "main_response", "Analysis completed successfully."
            ),
            "insights": data.get("insights", []),
            "recommendations": data.get("recommendations", []),
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse interpretation response: {e}")
        return {
            "main_response": (
                llm_response.strip()
                if llm_response.strip()
                else "Analysis completed successfully."
            ),
            "insights": [],
            "recommendations": [],
        }


def _parse_error_response(llm_response: str) -> Dict[str, Any]:
    """Parse LLM error interpretation response"""
    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = llm_response.strip()

        data = json.loads(json_str)

        return {
            "explanation": data.get(
                "explanation", "An error occurred during analysis."
            ),
            "suggestions": data.get("suggestions", []),
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse error response: {e}")
        return _create_fallback_error_response(llm_response)


def _create_fallback_error_response(error_message: str) -> Dict[str, Any]:
    """Create fallback error response when LLM parsing fails"""
    return {
        "explanation": f"I encountered an issue while analyzing your data: {error_message}. This might be due to data format issues or the complexity of the requested analysis.",
        "suggestions": [
            "Try rephrasing your question to be more specific",
            "Check if your data has the expected format and columns",
            "Consider asking for a simpler analysis first",
        ],
    }


def _format_technical_output(output: str) -> Optional[str]:
    """Format and clean technical output for display"""
    if not output or len(output.strip()) < 10:
        return None

    lines = output.split("\n")
    cleaned_lines = []

    for line in lines:
        if line.strip() and not any(
            noise in line.lower()
            for noise in ["warning:", "deprecated", "futurewarning", "/usr/local"]
        ):
            cleaned_lines.append(line)

    if not cleaned_lines:
        return None

    if len(cleaned_lines) > 20:
        cleaned_lines = cleaned_lines[:20] + ["... (output truncated)"]

    return "\n".join(cleaned_lines)
