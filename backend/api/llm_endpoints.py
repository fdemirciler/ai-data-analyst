"""
LLM Analysis Endpoints - Simplified Version

This module provides endpoints for real-time LLM-powered data analysis.
It integrates with the existing LangGraph workflow system to provide
intelligent code generation and analysis capabilities.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..workflow.state import WorkflowState
from ..workflow.graph import execute_workflow
from ..services.llm_provider import LLMManager
from ..config import settings

logger = logging.getLogger(__name__)


class AnalysisRequest(BaseModel):
    """Request model for LLM analysis"""

    query: str = Field(
        ..., min_length=1, max_length=2000, description="User's analysis question"
    )
    session_id: str = Field(..., description="Session ID for context")
    stream: Optional[bool] = Field(
        default=False, description="Whether to stream the response"
    )
    max_retries: Optional[int] = Field(
        default=2, ge=0, le=5, description="Maximum retry attempts"
    )


class AnalysisResponse(BaseModel):
    """Response model for LLM analysis"""

    success: bool
    session_id: str
    query: str
    analysis_id: str

    # LLM-generated content
    interpretation: str
    generated_code: Optional[str] = None
    code_explanation: Optional[str] = None

    # Execution results
    execution_output: Optional[str] = None
    execution_error: Optional[str] = None
    visualizations: Optional[list] = None

    # Metadata
    retry_count: int = 0
    processing_time: float
    timestamp: datetime

    # Response elements for frontend
    response_elements: list = Field(default_factory=list)


class LLMAnalysisService:
    """Service for handling LLM-powered analysis requests"""

    def __init__(self):
        self.llm_manager = LLMManager(settings)
        # External session storage will be passed in
        self.session_storage: Dict[str, Dict[str, Any]] = {}

    def set_session_storage(self, session_storage: Dict[str, Dict[str, Any]]):
        """Set external session storage reference"""
        self.session_storage = session_storage

    async def analyze_query(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Process analysis request using LLM workflow

        Args:
            request: Analysis request with query and session info

        Returns:
            Complete analysis response with LLM-generated insights
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())

        try:
            logger.info(f"🧠 Starting LLM analysis for session {request.session_id}")

            # Get session data and validate
            if request.session_id not in self.session_storage:
                raise HTTPException(status_code=404, detail="Session not found")

            session_data = self.session_storage[request.session_id]

            if "uploaded_file" not in session_data:
                raise HTTPException(
                    status_code=400, detail="No data file uploaded for analysis"
                )

            # Prepare workflow state
            workflow_state = self._prepare_workflow_state(
                request, session_data, analysis_id
            )

            # Execute LLM workflow
            logger.debug("Executing LLM workflow...")
            final_state = await execute_workflow(workflow_state)

            # Extract results
            response = self._build_response(
                request, final_state, analysis_id, start_time
            )

            # Update session with analysis results
            self._update_session_history(request.session_id, request.query, response)

            logger.info(
                f"✅ LLM analysis completed for session {request.session_id} in {response.processing_time:.2f}s"
            )
            return response

        except HTTPException:
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"LLM analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return error response
            return AnalysisResponse(
                success=False,
                session_id=request.session_id,
                query=request.query,
                analysis_id=analysis_id,
                interpretation=f"I encountered an error while analyzing your request: {str(e)}",
                processing_time=processing_time,
                timestamp=datetime.now(),
                response_elements=[
                    {
                        "type": "error",
                        "content": f"Analysis failed: {str(e)}",
                        "title": "Analysis Error",
                    }
                ],
            )

    def _prepare_workflow_state(
        self, request: AnalysisRequest, session_data: Dict[str, Any], analysis_id: str
    ) -> WorkflowState:
        """Prepare initial workflow state for LLM analysis"""

        file_info = session_data["uploaded_file"]

        # Get conversation history for context
        conversation_history = session_data.get("conversation_history", [])

        return {
            "session_id": request.session_id,
            "analysis_id": analysis_id,
            "query": request.query,
            "file_path": file_info["file_path"],
            "original_filename": file_info["filename"],
            "max_retries": request.max_retries,
            "conversation_history": conversation_history,
            "llm_provider": "gemini",  # Primary provider
            "workflow_status": "initializing",
            "current_node": "data_processing",
            "node_results": {},
            "retry_count": 0,
            "start_time": time.time(),
        }

    def _build_response(
        self,
        request: AnalysisRequest,
        final_state: Dict[str, Any],
        analysis_id: str,
        start_time: float,
    ) -> AnalysisResponse:
        """Build structured response from workflow results"""

        processing_time = time.time() - start_time

        # Extract workflow results
        workflow_successful = final_state.get("workflow_status") == "completed"
        node_results = final_state.get("node_results", {})

        # Extract generated code and explanation
        code_gen_results = node_results.get("code_generation", {})
        generated_code = code_gen_results.get("generated_code", "")
        code_explanation = code_gen_results.get("code_explanation", "")

        # Extract execution results
        exec_results = node_results.get("code_execution", {})
        execution_output = exec_results.get("execution_output", "")
        execution_error = exec_results.get("execution_error")
        visualizations = exec_results.get("plots", [])

        # Get final response and interpretation
        final_response = final_state.get("final_response", "")
        response_elements = final_state.get("response_elements", [])

        # If no interpretation available, create one
        if not final_response:
            if workflow_successful and execution_output:
                final_response = f"I've successfully analyzed your data. Here are the key findings:\n\n{execution_output}"
            elif execution_error:
                final_response = f"I encountered an issue while analyzing your data: {execution_error}"
            else:
                final_response = "I've processed your request. Please check the generated code and results below."

        # Build response elements if not provided
        if not response_elements:
            response_elements = []

            if generated_code:
                response_elements.append(
                    {
                        "type": "code",
                        "content": generated_code,
                        "title": "Generated Analysis Code",
                        "explanation": code_explanation,
                    }
                )

            if execution_output:
                response_elements.append(
                    {
                        "type": "output",
                        "content": execution_output,
                        "title": "Analysis Results",
                    }
                )

            if visualizations:
                response_elements.append(
                    {
                        "type": "visualizations",
                        "content": visualizations,
                        "title": "Generated Visualizations",
                    }
                )

            if execution_error:
                response_elements.append(
                    {
                        "type": "error",
                        "content": execution_error,
                        "title": "Execution Error",
                    }
                )

        return AnalysisResponse(
            success=workflow_successful,
            session_id=request.session_id,
            query=request.query,
            analysis_id=analysis_id,
            interpretation=final_response,
            generated_code=generated_code,
            code_explanation=code_explanation,
            execution_output=execution_output,
            execution_error=execution_error,
            visualizations=visualizations,
            retry_count=final_state.get("retry_count", 0),
            processing_time=processing_time,
            timestamp=datetime.now(),
            response_elements=response_elements,
        )

    def _update_session_history(
        self, session_id: str, query: str, response: AnalysisResponse
    ):
        """Update session with conversation history"""
        try:
            if session_id in self.session_storage:
                session_data = self.session_storage[session_id]
                history = session_data.get("conversation_history", [])

                # Add new conversation entry
                history.append(
                    {
                        "query": query,
                        "response_id": response.analysis_id,
                        "success": response.success,
                        "timestamp": response.timestamp.isoformat(),
                        "generated_code": response.generated_code,
                        "execution_output": response.execution_output,
                    }
                )

                # Keep only last 10 conversations to avoid memory issues
                if len(history) > 10:
                    history = history[-10:]

                session_data["conversation_history"] = history
                session_data["last_analysis"] = response.timestamp

        except Exception as e:
            logger.warning(f"Failed to update session history: {e}")


# Initialize service instance
llm_analysis_service = LLMAnalysisService()


async def analyze_with_llm(request: AnalysisRequest) -> AnalysisResponse:
    """
    Main endpoint function for LLM analysis

    This replaces the fake code generation with real LLM-powered analysis
    using your existing sophisticated backend infrastructure.
    """
    return await llm_analysis_service.analyze_query(request)


# Function to set session storage reference from main app
def set_session_storage(session_storage: Dict[str, Dict[str, Any]]):
    """Set the session storage reference"""
    llm_analysis_service.set_session_storage(session_storage)
