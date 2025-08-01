"""
Query Analysis Node for LangGraph Workflow

This node handles user query understanding and analysis planning:
1. Analyzes user intent and classification
2. Extracts required columns and analysis type
3. Creates structured analysis plan
4. Considers conversation context
5. Validates feasibility against available data

This is where the LLM first engages to understand what the user wants to achieve.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
import json
import re

from ..state import WorkflowState, WorkflowStateManager
from ...services.llm_provider import LLMManager
from ...utils.exceptions import WorkflowError
from ...config import settings

logger = logging.getLogger(__name__)


async def query_analysis_node(state: WorkflowState) -> WorkflowState:
    """
    Analyze user query and create analysis plan

    This node:
    1. Uses LLM to understand user intent
    2. Classifies the type of analysis needed
    3. Identifies required data columns
    4. Creates structured execution plan
    5. Validates against available data

    Args:
        state: Current workflow state with data profile and user query

    Returns:
        Updated workflow state with analysis plan
    """
    start_time = time.time()
    node_name = "query_analysis"

    try:
        logger.info(f"Starting query analysis for session {state['session_id']}")
        logger.debug(f"User query: {state['query']}")

        # Initialize LLM manager
        llm_manager = LLMManager(settings)

        # Prepare context for LLM
        context = _prepare_analysis_context(state)

        # Generate analysis plan using LLM
        logger.debug("Generating analysis plan with LLM...")
        analysis_prompt = _create_analysis_prompt(
            state["query"], context, state.get("conversation_history", [])
        )

        llm_response = await llm_manager.generate(
            analysis_prompt,
            provider=state["llm_provider"],
            temperature=0.1,
            max_tokens=1500,
        )

        # Parse LLM response into structured plan
        analysis_plan = _parse_analysis_response(llm_response.content)

        # Validate plan against available data
        validation_result = _validate_analysis_plan(analysis_plan, state)

        if not validation_result["valid"]:
            logger.warning(
                f"Analysis plan validation failed: {validation_result['reason']}"
            )
            # Try to create a fallback plan
            analysis_plan = _create_fallback_plan(state["query"], state)

        # Extract key information from plan
        user_intent = analysis_plan.get("intent", "general_analysis")
        analysis_type = analysis_plan.get("analysis_type", "exploratory")
        required_columns = analysis_plan.get("required_columns", [])

        # Processing metrics
        processing_time = time.time() - start_time

        logger.info(
            f"Query analysis completed in {processing_time:.2f}s - "
            f"intent: {user_intent}, type: {analysis_type}"
        )

        # Update workflow state
        results = {
            "user_intent": user_intent,
            "analysis_plan": analysis_plan,
            "required_columns": required_columns,
            "analysis_type": analysis_type,
        }

        # Record successful completion
        state = WorkflowStateManager.record_node_completion(
            state, node_name, processing_time, results
        )

        # Transition to next node
        state = WorkflowStateManager.transition_to_node(state, "code_generation")

        return state

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Query analysis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Record failure
        state = WorkflowStateManager.record_node_failure(
            state, node_name, error_msg, processing_time
        )

        raise WorkflowError(f"Query analysis node failed: {str(e)}")


def _prepare_analysis_context(state: WorkflowState) -> Dict[str, Any]:
    """
    Prepare comprehensive context for LLM analysis

    Args:
        state: Current workflow state

    Returns:
        Context dictionary for LLM
    """
    context = {
        "dataset_info": {
            "filename": state.get("original_filename", "unknown"),
            "shape": (state.get("cleaned_data_info") or {}).get("shape", [0, 0]),
            "columns": (state.get("cleaned_data_info") or {}).get("columns", []),
            "data_types": (state.get("cleaned_data_info") or {}).get("dtypes", {}),
            "quality_score": (state.get("quality_metrics") or {}).get(
                "overall_score", 0
            ),
        },
        "sample_data": state.get("sample_data", ""),
        "data_profile": state.get("data_profile", {}),
        "processing_applied": state.get("processing_applied", []),
        "available_columns": {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "all": (state.get("cleaned_data_info") or {}).get("columns", []),
        },
    }

    # Categorize columns by type
    dtypes = context["dataset_info"]["data_types"]
    for col, dtype in dtypes.items():
        if "int" in str(dtype) or "float" in str(dtype):
            context["available_columns"]["numeric"].append(col)
        elif "datetime" in str(dtype):
            context["available_columns"]["datetime"].append(col)
        else:
            context["available_columns"]["categorical"].append(col)

    return context


def _create_analysis_prompt(
    query: str, context: Dict[str, Any], conversation_history: List[Dict[str, Any]]
) -> str:
    """
    Create comprehensive prompt for query analysis

    Args:
        query: User's question
        context: Data context
        conversation_history: Previous conversation

    Returns:
        Formatted prompt for LLM
    """

    # Format conversation history
    history_context = ""
    if conversation_history:
        history_context = "Previous conversation:\n"
        for i, item in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
            history_context += f"{i}. User: {item.get('query', '')}\n"
            if item.get("analysis_type"):
                history_context += f"   Analysis: {item.get('analysis_type')} (Success: {item.get('success', False)})\n"
        history_context += "\n"

    prompt = f"""You are an expert data analyst tasked with understanding a user's query and creating an analysis plan.

{history_context}CURRENT USER QUERY: "{query}"

DATASET INFORMATION:
- Filename: {context['dataset_info']['filename']}
- Shape: {context['dataset_info']['shape'][0]} rows × {context['dataset_info']['shape'][1]} columns
- Data Quality Score: {context['dataset_info']['quality_score']:.2f}/1.0
- Processing Applied: {', '.join(context['processing_applied']) if context['processing_applied'] else 'None'}

AVAILABLE COLUMNS:
- Numeric columns ({len(context['available_columns']['numeric'])}): {', '.join(context['available_columns']['numeric'][:10])}{'...' if len(context['available_columns']['numeric']) > 10 else ''}
- Categorical columns ({len(context['available_columns']['categorical'])}): {', '.join(context['available_columns']['categorical'][:10])}{'...' if len(context['available_columns']['categorical']) > 10 else ''}
- Date/Time columns ({len(context['available_columns']['datetime'])}): {', '.join(context['available_columns']['datetime'])}
- All columns: {', '.join(context['available_columns']['all'])}

SAMPLE DATA:
{context['sample_data'][:1500]}{'...' if len(context['sample_data']) > 1500 else ''}

TASK: Analyze the user's query and create a structured analysis plan. Consider:

1. **Intent Classification**: What does the user want to achieve?
   - statistical_analysis: Descriptive statistics, correlations, distributions
   - visualization: Charts, plots, graphs
   - comparison: Compare groups, categories, time periods
   - prediction: Forecasting, modeling
   - exploration: General data exploration
   - filtering: Data filtering and subsetting
   - aggregation: Grouping and summarizing
   - correlation: Relationship analysis
   - trend_analysis: Time-based patterns
   - custom: Specific business logic

2. **Analysis Type**: How should this be approached?
   - descriptive: Basic statistics and summaries
   - diagnostic: Why something happened
   - predictive: What might happen
   - prescriptive: What should be done

3. **Required Columns**: Which columns are needed for this analysis?

4. **Feasibility**: Can this analysis be performed with available data?

5. **Complexity**: Simple query or multi-step analysis?

Respond with a JSON object in this exact format:
```json
{{
    "intent": "statistical_analysis|visualization|comparison|prediction|exploration|filtering|aggregation|correlation|trend_analysis|custom",
    "analysis_type": "descriptive|diagnostic|predictive|prescriptive",
    "confidence": 0.95,
    "required_columns": ["column1", "column2"],
    "feasible": true,
    "complexity": "simple|moderate|complex",
    "reasoning": "Brief explanation of the analysis approach",
    "suggested_approach": [
        "Step 1: Load and examine the data",
        "Step 2: Perform specific analysis",
        "Step 3: Create visualizations if needed"
    ],
    "expected_outputs": [
        "Statistical summary",
        "Visualization",
        "Key insights"
    ],
    "potential_issues": [
        "Possible data quality concerns",
        "Missing data considerations"
    ]
}}
```

Ensure the JSON is valid and complete. Focus on being practical and actionable."""

    return prompt


def _parse_analysis_response(llm_response: str) -> Dict[str, Any]:
    """
    Parse LLM response into structured analysis plan

    Args:
        llm_response: Raw LLM response

    Returns:
        Structured analysis plan dictionary
    """
    try:
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_str = llm_response.strip()

        # Parse JSON
        plan = json.loads(json_str)

        # Validate required fields
        required_fields = ["intent", "analysis_type", "feasible", "reasoning"]
        for field in required_fields:
            if field not in plan:
                logger.warning(f"Missing required field in analysis plan: {field}")

        # Set defaults for missing fields
        plan.setdefault("confidence", 0.8)
        plan.setdefault("required_columns", [])
        plan.setdefault("complexity", "moderate")
        plan.setdefault("suggested_approach", [])
        plan.setdefault("expected_outputs", [])
        plan.setdefault("potential_issues", [])

        return plan

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse analysis plan JSON: {e}")
        logger.debug(f"Raw LLM response: {llm_response}")

        # Return fallback plan
        return {
            "intent": "exploration",
            "analysis_type": "descriptive",
            "confidence": 0.5,
            "required_columns": [],
            "feasible": True,
            "complexity": "simple",
            "reasoning": "Failed to parse detailed plan, using fallback analysis",
            "suggested_approach": ["Perform basic data exploration"],
            "expected_outputs": ["Basic statistics and insights"],
            "potential_issues": ["Analysis plan parsing failed"],
        }

    except Exception as e:
        logger.error(f"Unexpected error parsing analysis plan: {e}")
        return {
            "intent": "exploration",
            "analysis_type": "descriptive",
            "confidence": 0.3,
            "required_columns": [],
            "feasible": True,
            "complexity": "simple",
            "reasoning": f"Error in analysis: {str(e)}",
            "suggested_approach": ["Basic data analysis"],
            "expected_outputs": ["Simple data summary"],
            "potential_issues": [f"Analysis error: {str(e)}"],
        }


def _validate_analysis_plan(
    plan: Dict[str, Any], state: WorkflowState
) -> Dict[str, Any]:
    """
    Validate analysis plan against available data

    Args:
        plan: Analysis plan to validate
        state: Current workflow state

    Returns:
        Validation result with valid flag and reasons
    """
    validation = {"valid": True, "reason": "", "warnings": []}

    available_columns = (state.get("cleaned_data_info") or {}).get("columns", [])
    required_columns = plan.get("required_columns", [])

    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in available_columns]
    if missing_columns:
        validation["valid"] = False
        validation["reason"] = f"Required columns not found: {missing_columns}"
        return validation

    # Check feasibility flag
    if not plan.get("feasible", True):
        validation["valid"] = False
        validation["reason"] = "Analysis marked as not feasible"
        return validation

    # Check data quality
    quality_score = (state.get("quality_metrics") or {}).get("overall_score", 1.0)
    if quality_score < 0.3:
        validation["warnings"].append(
            "Low data quality may affect analysis reliability"
        )

    # Check for sufficient data
    row_count = (state.get("cleaned_data_info") or {}).get("shape", [0])[0]
    if row_count < 10:
        validation["warnings"].append("Very small dataset may limit analysis options")

    return validation


def _create_fallback_plan(query: str, state: WorkflowState) -> Dict[str, Any]:
    """
    Create simple fallback analysis plan when LLM plan fails

    Args:
        query: User query
        state: Current workflow state

    Returns:
        Simple fallback analysis plan
    """
    available_columns = (state.get("cleaned_data_info") or {}).get("columns", [])

    # Simple keyword-based intent detection
    query_lower = query.lower()

    if any(
        word in query_lower for word in ["plot", "chart", "graph", "visualize", "show"]
    ):
        intent = "visualization"
        analysis_type = "descriptive"
    elif any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
        intent = "comparison"
        analysis_type = "diagnostic"
    elif any(
        word in query_lower for word in ["correlation", "relationship", "related"]
    ):
        intent = "correlation"
        analysis_type = "diagnostic"
    elif any(
        word in query_lower for word in ["trend", "time", "over time", "temporal"]
    ):
        intent = "trend_analysis"
        analysis_type = "descriptive"
    else:
        intent = "exploration"
        analysis_type = "descriptive"

    return {
        "intent": intent,
        "analysis_type": analysis_type,
        "confidence": 0.6,
        "required_columns": available_columns[:5],  # Use first 5 columns
        "feasible": True,
        "complexity": "simple",
        "reasoning": f"Fallback analysis for query: {query[:100]}...",
        "suggested_approach": [
            "Load the data",
            "Perform basic analysis based on detected intent",
            "Generate summary and insights",
        ],
        "expected_outputs": [
            "Data summary",
            "Basic analysis results",
            "Simple insights",
        ],
        "potential_issues": ["Using fallback analysis due to plan generation failure"],
    }
