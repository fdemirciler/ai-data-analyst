"""
Code Generation Node for LangGraph Workflow

This node handles LLM-powered Python code generation:
1. Uses analysis plan to create contextual prompts
2. Generates Python code for data analysis
3. Validates code safety and syntax
4. Provides code explanations and reasoning

This is where the LLM creates executable Python code based on user intent and data context.
"""

import asyncio
import time
import logging
import ast
import re
from typing import Dict, Any, List, Set

from ..state import WorkflowState, WorkflowStateManager
from ...services.llm_provider import LLMManager
from ...utils.exceptions import WorkflowError
from ...config import settings

logger = logging.getLogger(__name__)


async def code_generation_node(state: WorkflowState) -> WorkflowState:
    """
    Generate Python code for data analysis using LLM

    This node:
    1. Creates context-rich prompts for LLM
    2. Generates Python code based on analysis plan
    3. Validates code safety and syntax
    4. Provides code explanation and reasoning

    Args:
        state: Current workflow state with analysis plan

    Returns:
        Updated workflow state with generated code
    """
    start_time = time.time()
    node_name = "code_generation"

    try:
        logger.info(f"Starting code generation for session {state['session_id']}")

        # Initialize LLM manager
        llm_manager = LLMManager(settings)

        # Create comprehensive context for code generation
        code_context = _prepare_code_generation_context(state)

        # Generate Python code using LLM
        logger.debug("Generating Python code with LLM...")
        code_prompt = _create_code_generation_prompt(code_context)

        llm_response = await llm_manager.generate(
            code_prompt,
            provider=state["llm_provider"],
            temperature=0.1,  # Low temperature for more deterministic code
            max_tokens=2000,
        )

        # Extract and validate generated code
        code_result = _parse_code_response(llm_response.content)

        # Validate code safety
        safety_check = _validate_code_safety(code_result["code"])
        if not safety_check["safe"]:
            logger.warning(
                f"Generated code failed safety check: {safety_check['reason']}"
            )
            # Try to generate safer code
            code_result = await _generate_safer_code(
                llm_manager, code_context, safety_check, state
            )

        # Validate syntax
        syntax_check = _validate_syntax(code_result["code"])
        if not syntax_check["valid"]:
            logger.warning(f"Generated code has syntax errors: {syntax_check['error']}")
            # Try to fix syntax
            code_result = await _fix_code_syntax(
                llm_manager, code_result, syntax_check, state
            )

        # Estimate execution time (simple heuristic)
        estimated_time = _estimate_execution_time(code_result["code"], state)

        # Processing metrics
        processing_time = time.time() - start_time

        logger.info(
            f"Code generation completed in {processing_time:.2f}s - "
            f"code length: {len(code_result['code'])} chars"
        )

        # Update workflow state
        results = {
            "generated_code": code_result["code"],
            "code_explanation": code_result["explanation"],
            "estimated_execution_time": estimated_time,
        }

        # Record successful completion
        state = WorkflowStateManager.record_node_completion(
            state, node_name, processing_time, results
        )

        # Transition to next node
        state = WorkflowStateManager.transition_to_node(state, "code_execution")

        return state

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Code generation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Record failure
        state = WorkflowStateManager.record_node_failure(
            state, node_name, error_msg, processing_time
        )

        raise WorkflowError(f"Code generation node failed: {str(e)}")


def _prepare_code_generation_context(state: WorkflowState) -> Dict[str, Any]:
    """
    Prepare comprehensive context for code generation

    Args:
        state: Current workflow state

    Returns:
        Context dictionary for LLM code generation
    """
    return {
        "query": state["query"],
        "analysis_plan": state.get("analysis_plan", {}),
        "user_intent": state.get("user_intent", "exploration"),
        "analysis_type": state.get("analysis_type", "descriptive"),
        "required_columns": state.get("required_columns", []),
        "dataset_info": {
            "filename": state.get("original_filename", "data.parquet"),
            "shape": (state.get("cleaned_data_info") or {}).get("shape", [0, 0]),
            "columns": (state.get("cleaned_data_info") or {}).get("columns", []),
            "data_types": (state.get("cleaned_data_info") or {}).get("dtypes", {}),
            "sample_data": state.get("sample_data", ""),
            "quality_score": (state.get("quality_metrics") or {}).get(
                "overall_score", 0
            ),
        },
        "data_profile": state.get("data_profile", {}),
        "conversation_history": state.get("conversation_history", []),
        "processing_applied": state.get("processing_applied", []),
    }


def _create_code_generation_prompt(context: Dict[str, Any]) -> str:
    """
    Create comprehensive prompt for Python code generation

    Args:
        context: Code generation context

    Returns:
        Formatted prompt for LLM
    """

    # Format data types for prompt
    data_types_str = ""
    if context["dataset_info"]["data_types"]:
        data_types_str = "Data types:\n"
        for col, dtype in list(context["dataset_info"]["data_types"].items())[
            :15
        ]:  # Limit to first 15
            data_types_str += f"  {col}: {dtype}\n"

    # Format analysis plan
    plan_str = ""
    if context["analysis_plan"]:
        plan = context["analysis_plan"]
        plan_str = f"""
Analysis Plan:
- Intent: {plan.get('intent', 'unknown')}
- Type: {plan.get('analysis_type', 'unknown')}
- Approach: {', '.join(plan.get('suggested_approach', [])[:3])}
- Expected outputs: {', '.join(plan.get('expected_outputs', [])[:3])}
"""

    # Format conversation context
    history_str = ""
    if context["conversation_history"]:
        history_str = "Previous conversation context:\n"
        for i, item in enumerate(
            context["conversation_history"][-2:], 1
        ):  # Last 2 items
            history_str += f"{i}. Previous query: {item.get('query', '')[:100]}\n"

    prompt = f"""You are an expert Python data analyst. Generate clean, efficient Python code to analyze data and answer the user's question.

USER QUERY: "{context['query']}"

{plan_str}

DATASET INFORMATION:
- File: {context['dataset_info']['filename']}
- Shape: {context['dataset_info']['shape'][0]} rows × {context['dataset_info']['shape'][1]} columns
- Quality Score: {context['dataset_info']['quality_score']:.2f}/1.0

AVAILABLE COLUMNS ({len(context['dataset_info']['columns'])}):
{', '.join(context['dataset_info']['columns'][:20])}{'...' if len(context['dataset_info']['columns']) > 20 else ''}

{data_types_str}

SAMPLE DATA:
{context['dataset_info']['sample_data'][:800]}{'...' if len(context['dataset_info']['sample_data']) > 800 else ''}

{history_str}

REQUIREMENTS:
1. **Data Access**: The data is already loaded as a pandas DataFrame named 'df'
2. **Libraries**: You can use pandas, numpy, matplotlib, seaborn, scipy, sklearn (basic)
3. **Visualizations**: Create clear, well-labeled plots when appropriate
4. **Safety**: Do not access files, networks, or system resources
5. **Output**: Use print() for text results, matplotlib for plots
6. **Error Handling**: Include basic error handling for edge cases

ANALYSIS TYPE: {context['analysis_type']} 
USER INTENT: {context['user_intent']}

Generate Python code that:
- Directly addresses the user's question
- Follows the analysis plan approach
- Includes appropriate visualizations if needed
- Provides clear, interpretable results
- Handles potential data issues gracefully

Respond with this exact format:
```python
# Analysis: [Brief description of what this code does]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# [Your analysis code here]
# Make sure to:
# 1. Examine the data structure if needed
# 2. Perform the requested analysis
# 3. Create visualizations when helpful
# 4. Print key findings and insights
# 5. Handle edge cases appropriately

print("Analysis completed successfully!")
```

EXPLANATION:
[Provide a clear 2-3 sentence explanation of what the code does and why this approach was chosen]

Generate code that is production-ready, well-commented, and directly answers the user's question."""

    return prompt


def _parse_code_response(llm_response: str) -> Dict[str, str]:
    """
    Parse LLM response to extract code and explanation

    Args:
        llm_response: Raw LLM response

    Returns:
        Dictionary with code and explanation
    """
    result = {"code": "", "explanation": ""}

    try:
        # Extract Python code block
        code_pattern = r"```python\s*(.*?)\s*```"
        code_match = re.search(code_pattern, llm_response, re.DOTALL)

        if code_match:
            result["code"] = code_match.group(1).strip()
        else:
            # Fallback: look for any code-like content
            lines = llm_response.split("\n")
            code_lines = []
            in_code_block = False

            for line in lines:
                if (
                    "import " in line
                    or "pd." in line
                    or "plt." in line
                    or "print(" in line
                ):
                    in_code_block = True

                if in_code_block:
                    code_lines.append(line)

                if line.strip() and not any(
                    keyword in line
                    for keyword in ["import", "pd.", "plt.", "print", "#", "df"]
                ):
                    in_code_block = False

            if code_lines:
                result["code"] = "\n".join(code_lines).strip()

        # Extract explanation
        explanation_pattern = r"EXPLANATION:\s*(.*?)(?:\n\n|\Z)"
        explanation_match = re.search(explanation_pattern, llm_response, re.DOTALL)

        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip()
        else:
            # Fallback: use first few sentences after code
            remaining_text = llm_response.replace(result["code"], "").strip()
            sentences = remaining_text.split(".")[:3]  # First 3 sentences
            result["explanation"] = ".".join(sentences).strip() + "."

        # Ensure we have some code
        if not result["code"]:
            result["code"] = (
                "# No valid code could be extracted\nprint('Code generation failed')"
            )
            result["explanation"] = (
                "Failed to extract valid Python code from LLM response"
            )

        # Ensure we have some explanation
        if not result["explanation"]:
            result["explanation"] = "This code performs the requested data analysis."

        return result

    except Exception as e:
        logger.error(f"Error parsing code response: {e}")
        return {
            "code": f"# Error parsing generated code: {str(e)}\nprint('Code parsing failed')",
            "explanation": f"Error occurred while parsing the generated code: {str(e)}",
        }


def _validate_code_safety(code: str) -> Dict[str, Any]:
    """
    Validate that generated code is safe to execute

    Args:
        code: Python code to validate

    Returns:
        Dictionary with safety validation results
    """
    result = {"safe": True, "reason": "", "issues": []}

    # Define dangerous patterns
    dangerous_patterns = [
        # File system operations
        r"\bopen\s*\(",
        r"\bfile\s*\(",
        r"\bwith\s+open\s*",
        r"\.write\s*\(",
        r"\.read\s*\(",
        # System operations
        r"\bos\.",
        r"\bsys\.",
        r"\bsubprocess\.",
        r"\bexec\s*\(",
        r"\beval\s*\(",
        r"\bcompile\s*\(",
        r"__import__",
        # Network operations
        r"\burllib\.",
        r"\brequests\.",
        r"\bhttp\.",
        r"\bsocket\.",
        # Dangerous builtins
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
        r"\bvars\s*\(",
        r"\bdir\s*\(",
        r"\bgetattr\s*\(",
        r"\bsetattr\s*\(",
        r"\bdelattr\s*\(",
        # Input operations
        r"\binput\s*\(",
        r"\braw_input\s*\(",
        # Exit operations
        r"\bexit\s*\(",
        r"\bquit\s*\(",
    ]

    code_lower = code.lower()

    for pattern in dangerous_patterns:
        if re.search(pattern, code_lower):
            result["safe"] = False
            result["issues"].append(f"Dangerous pattern detected: {pattern}")

    if result["issues"]:
        result["reason"] = f"Code contains {len(result['issues'])} safety issues"

    return result


def _validate_syntax(code: str) -> Dict[str, Any]:
    """
    Validate Python syntax of generated code

    Args:
        code: Python code to validate

    Returns:
        Dictionary with syntax validation results
    """
    result = {"valid": True, "error": "", "line_number": None}

    try:
        ast.parse(code)
    except SyntaxError as e:
        result["valid"] = False
        result["error"] = str(e)
        result["line_number"] = e.lineno
    except Exception as e:
        result["valid"] = False
        result["error"] = f"Unexpected parsing error: {str(e)}"

    return result


async def _generate_safer_code(
    llm_manager: LLMManager,
    context: Dict[str, Any],
    safety_issues: Dict[str, Any],
    state: WorkflowState,
) -> Dict[str, str]:
    """
    Generate safer version of code after safety check failure
    """
    logger.debug("Attempting to generate safer code...")

    safer_prompt = f"""The previous code generation contained unsafe operations. Generate a safer version.

ORIGINAL QUERY: "{context['query']}"
ANALYSIS TYPE: {context['analysis_type']}

SAFETY ISSUES DETECTED:
{'; '.join(safety_issues['issues'])}

Generate safe Python code that:
1. Uses ONLY pandas, numpy, matplotlib, seaborn, scipy operations
2. Does NOT access files, system, network, or external resources
3. Does NOT use exec, eval, open, or system calls
4. Works with the existing DataFrame 'df'
5. Still addresses the user's original question

Provide clean, safe code in ```python``` blocks.
```python
# Safe analysis code here
```

EXPLANATION: Brief explanation of the safe approach.
"""

    try:
        response = await llm_manager.generate(
            safer_prompt,
            provider=state["llm_provider"],
            temperature=0.1,
            max_tokens=1500,
        )

        return _parse_code_response(response.content)

    except Exception as e:
        logger.error(f"Failed to generate safer code: {e}")
        return {
            "code": "# Safe fallback code\nprint('Basic data information:')\nprint(f'Dataset shape: {df.shape}')\nprint(f'Columns: {list(df.columns)}')\nprint('\\nFirst few rows:')\nprint(df.head())",
            "explanation": "Generated safe fallback code due to safety validation failure.",
        }


async def _fix_code_syntax(
    llm_manager: LLMManager,
    code_result: Dict[str, str],
    syntax_error: Dict[str, Any],
    state: WorkflowState,
) -> Dict[str, str]:
    """
    Attempt to fix syntax errors in generated code
    """
    logger.debug("Attempting to fix code syntax...")

    fix_prompt = f"""Fix the syntax error in this Python code:

ORIGINAL CODE:
```python
{code_result['code']}
```

SYNTAX ERROR:
Line {syntax_error.get('line_number', 'unknown')}: {syntax_error['error']}

Provide the corrected code:
```python
# Fixed code here
```

EXPLANATION: Brief explanation of what was fixed.
"""

    try:
        response = await llm_manager.generate(
            fix_prompt, provider=state["llm_provider"], temperature=0.1, max_tokens=1500
        )

        fixed_result = _parse_code_response(response.content)

        # Validate the fix
        if _validate_syntax(fixed_result["code"])["valid"]:
            return fixed_result
        else:
            raise Exception("Fix attempt still has syntax errors")

    except Exception as e:
        logger.error(f"Failed to fix syntax: {e}")
        return {
            "code": "# Fallback code due to syntax errors\nprint('Analysis could not be completed due to syntax errors')\nprint(f'Dataset shape: {df.shape}')\ndf.info()",
            "explanation": "Generated fallback code due to persistent syntax errors.",
        }


def _estimate_execution_time(code: str, state: WorkflowState) -> float:
    """
    Estimate code execution time based on operations and data size

    Args:
        code: Generated Python code
        state: Current workflow state

    Returns:
        Estimated execution time in seconds
    """
    base_time = 1.0  # Base execution time

    # Get data size
    shape = (state.get("cleaned_data_info") or {}).get("shape", [1000, 10])
    rows, cols = shape[0], shape[1]

    # Data size factor
    size_factor = min((rows * cols) / 100000, 5.0)  # Cap at 5x for very large datasets

    # Operation complexity factors
    complexity_patterns = {
        r"\.plot\(": 2.0,  # Plotting operations
        r"plt\.": 2.0,  # Matplotlib operations
        r"sns\.": 2.5,  # Seaborn operations
        r"\.corr\(": 1.5,  # Correlation calculations
        r"\.groupby\(": 1.3,  # Groupby operations
        r"\.merge\(": 1.4,  # Merge operations
        r"\.sort_values\(": 1.2,  # Sorting operations
        r"for\s+.*in\s+": 1.5,  # Loops
        r"sklearn": 3.0,  # Machine learning
        r"\.fit\(": 2.5,  # Model fitting
    }

    complexity_multiplier = 1.0
    for pattern, multiplier in complexity_patterns.items():
        if re.search(pattern, code):
            complexity_multiplier *= multiplier

    # Cap complexity multiplier
    complexity_multiplier = min(complexity_multiplier, 10.0)

    estimated_time = base_time * size_factor * complexity_multiplier

    # Cap at reasonable maximum
    return min(estimated_time, 30.0)
