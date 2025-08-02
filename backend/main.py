"""
FastAPI Main Application
========================

This is the main FastAPI server that provides REST API endpoints
for the data analysis workflow system.
"""

import os
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Import our workflow and services
# from workflow.main import WorkflowOrchestrator
from workflow.state import WorkflowState  # Import state separately
from services.session_manager import SessionManager
from services.file_handler import FileHandler

# from models.session import DataMetadata, SessionData
from security import SecurePythonExecutor, ExecutionLimits
from config.settings import get_settings

# Import LLM endpoints
from api.llm_endpoints import (
    analyze_with_llm,
    AnalysisRequest as LLMAnalysisRequest,
    AnalysisResponse as LLMAnalysisResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Analysis Workflow API",
    description="AI-powered data analysis and visualization platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
session_manager = SessionManager()
file_handler = FileHandler()
# workflow_orchestrator = WorkflowOrchestrator()

# Secure code executor for custom analysis
secure_executor = SecurePythonExecutor(
    execution_limits=ExecutionLimits(
        max_execution_time=60.0,  # Allow longer execution for analysis
        max_memory_mb=512,
        max_output_size=50000,
    )
)


# Pydantic models for API
class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    status: str = "active"


class AnalysisRequest(BaseModel):
    session_id: str
    analysis_type: str = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CustomCodeRequest(BaseModel):
    session_id: str
    code: str = Field(..., description="Python code to execute")
    context_variables: Dict[str, Any] = Field(default_factory=dict)


class WorkflowStatusResponse(BaseModel):
    session_id: str
    status: str
    current_step: Optional[str] = None
    progress: float = Field(ge=0.0, le=1.0)
    message: Optional[str] = None
    error: Optional[str] = None


class AnalysisResultResponse(BaseModel):
    session_id: str
    analysis_type: str
    result: Dict[str, Any]
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    execution_time: float
    timestamp: datetime


# Global storage for workflow states (in production, use Redis/database)
active_workflows: Dict[str, WorkflowState] = {}
workflow_results: Dict[str, Dict[str, Any]] = {}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "services": {
            "session_manager": "active",
            "file_handler": "active",
            "workflow_orchestrator": "active",
        },
    }


# Session management endpoints
@app.post("/api/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new analysis session"""
    try:
        session_id = str(uuid.uuid4())
        session_data = await session_manager.create_session(session_id)

        logger.info(f"Created new session: {session_id}")

        return SessionResponse(
            session_id=session_id,
            created_at=session_data["created_at"],
            status="active",
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session information"""
    try:
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionResponse(
            session_id=session_id,
            created_at=session_data["created_at"],
            status=session_data.get("status", "active"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup resources"""
    try:
        # Clean up workflow state
        if session_id in active_workflows:
            del active_workflows[session_id]
        if session_id in workflow_results:
            del workflow_results[session_id]

        # Delete session
        success = await session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        logger.info(f"Deleted session: {session_id}")
        return {"message": "Session deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# File upload endpoints
@app.post("/api/sessions/{session_id}/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload a data file for analysis"""
    try:
        # Verify session exists
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Validate file type
        allowed_extensions = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}",
            )

        # Save file
        file_info = await file_handler.save_uploaded_file(file, session_id)

        # Update session with file info
        await session_manager.update_session(
            session_id, {"uploaded_file": file_info, "status": "file_uploaded"}
        )

        logger.info(f"File uploaded for session {session_id}: {file.filename}")

        return {
            "message": "File uploaded successfully",
            "file_info": file_info,
            "session_id": session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Workflow execution endpoints
@app.post("/api/sessions/{session_id}/analyze", response_model=WorkflowStatusResponse)
async def start_analysis(
    session_id: str, request: AnalysisRequest, background_tasks: BackgroundTasks
):
    """Start data analysis workflow"""
    try:
        # Verify session and file
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        if "uploaded_file" not in session_data:
            raise HTTPException(
                status_code=400, detail="No file uploaded for this session"
            )

        # Initialize workflow state
        initial_state = WorkflowState(
            session_id=session_id,
            file_path=session_data["uploaded_file"]["file_path"],
            analysis_type=request.analysis_type,
            parameters=request.parameters,
            current_step="initializing",
            status="running",
        )

        active_workflows[session_id] = initial_state

        # Start workflow in background
        background_tasks.add_task(run_workflow_async, session_id, initial_state)

        logger.info(f"Started analysis workflow for session {session_id}")

        return WorkflowStatusResponse(
            session_id=session_id,
            status="running",
            current_step="initializing",
            progress=0.0,
            message="Analysis workflow started",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(session_id: str):
    """Get current workflow status"""
    try:
        if session_id not in active_workflows:
            # Check if we have completed results
            if session_id in workflow_results:
                return WorkflowStatusResponse(
                    session_id=session_id,
                    status="completed",
                    current_step="finished",
                    progress=1.0,
                    message="Analysis completed successfully",
                )
            else:
                raise HTTPException(
                    status_code=404, detail="No active workflow for this session"
                )

        workflow_state = active_workflows[session_id]

        return WorkflowStatusResponse(
            session_id=session_id,
            status=workflow_state.status,
            current_step=workflow_state.current_step,
            progress=workflow_state.progress,
            message=workflow_state.message,
            error=workflow_state.error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/results", response_model=AnalysisResultResponse)
async def get_analysis_results(session_id: str):
    """Get analysis results"""
    try:
        if session_id not in workflow_results:
            # Check if workflow is still running
            if session_id in active_workflows:
                workflow_state = active_workflows[session_id]
                if workflow_state.status == "running":
                    raise HTTPException(
                        status_code=202, detail="Analysis still in progress"
                    )
                elif workflow_state.status == "error":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Analysis failed: {workflow_state.error}",
                    )

            raise HTTPException(
                status_code=404, detail="No results available for this session"
            )

        results = workflow_results[session_id]

        return AnalysisResultResponse(
            session_id=session_id,
            analysis_type=results.get("analysis_type", "unknown"),
            result=results.get("analysis_results", {}),
            visualizations=results.get("visualizations", []),
            insights=results.get("insights", []),
            execution_time=results.get("execution_time", 0.0),
            timestamp=results.get("timestamp", datetime.now()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import LLM endpoints
from api.llm_endpoints import (
    analyze_with_llm,
    AnalysisRequest as LLMAnalysisRequest,
    AnalysisResponse as LLMAnalysisResponse,
)


# LLM Analysis endpoint - THE REAL AI INTEGRATION
@app.post("/api/sessions/{session_id}/analyze-llm", response_model=LLMAnalysisResponse)
async def analyze_with_llm_endpoint(session_id: str, query: str):
    """
    Analyze data using real LLM integration (replaces fake AI)

    This endpoint uses your sophisticated LLM infrastructure:
    - LangGraph workflow orchestration
    - Real LLM providers (Gemini, OpenRouter, Together.ai)
    - Code generation, execution, and interpretation
    - Context-aware analysis with conversation history
    """
    try:
        request = LLMAnalysisRequest(
            query=query, session_id=session_id, stream=False, max_retries=2
        )

        # Use the real LLM analysis service
        response = await analyze_with_llm(request)

        logger.info(
            f"LLM analysis completed for session {session_id}: success={response.success}"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in LLM analysis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Custom code execution endpoint
@app.post("/api/sessions/{session_id}/execute")
async def execute_custom_code(session_id: str, request: CustomCodeRequest):
    """Execute custom Python code in secure sandbox"""
    try:
        # Verify session exists
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get context data if available
        context_data = {}
        if session_id in workflow_results:
            # Add analysis results to context
            results = workflow_results[session_id]
            if "dataframe" in results:
                context_data["df"] = results["dataframe"]
            if "analysis_results" in results:
                context_data["analysis_results"] = results["analysis_results"]

        # Add any additional context variables
        context_data.update(request.context_variables)

        # Execute code securely
        execution_result = secure_executor.execute(request.code, context_data)

        logger.info(f"Executed custom code for session {session_id}")

        return {
            "session_id": session_id,
            "success": execution_result.success,
            "output": execution_result.output,
            "error": execution_result.error,
            "execution_time": execution_result.execution_time,
            "warnings": execution_result.warnings,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing custom code for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# File download endpoints
@app.get("/api/sessions/{session_id}/download/{file_type}")
async def download_file(session_id: str, file_type: str):
    """Download generated files (visualizations, reports, etc.)"""
    try:
        if session_id not in workflow_results:
            raise HTTPException(
                status_code=404, detail="No results available for this session"
            )

        results = workflow_results[session_id]

        # Handle different file types
        if file_type == "report" and "report_path" in results:
            return FileResponse(
                results["report_path"],
                filename=f"analysis_report_{session_id}.html",
                media_type="text/html",
            )
        elif file_type == "data" and "processed_data_path" in results:
            return FileResponse(
                results["processed_data_path"],
                filename=f"processed_data_{session_id}.csv",
                media_type="text/csv",
            )
        else:
            raise HTTPException(
                status_code=404, detail=f"File type '{file_type}' not available"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background workflow execution
async def run_workflow_async(session_id: str, initial_state: WorkflowState):
    """Run the workflow asynchronously in the background"""
    try:
        logger.info(f"Starting background workflow for session {session_id}")

        # Run the workflow
        final_state = await workflow_orchestrator.run_workflow(initial_state)

        # Store results
        workflow_results[session_id] = {
            "analysis_type": final_state.analysis_type,
            "analysis_results": final_state.analysis_results,
            "visualizations": final_state.visualizations,
            "insights": final_state.insights,
            "dataframe": final_state.dataframe,
            "execution_time": final_state.execution_time,
            "timestamp": datetime.now(),
        }

        # Clean up active workflow
        if session_id in active_workflows:
            del active_workflows[session_id]

        # Update session status
        await session_manager.update_session(
            session_id, {"status": "completed", "completed_at": datetime.now()}
        )

        logger.info(f"Completed background workflow for session {session_id}")

    except Exception as e:
        logger.error(f"Error in background workflow for session {session_id}: {e}")

        # Update workflow state with error
        if session_id in active_workflows:
            active_workflows[session_id].status = "error"
            active_workflows[session_id].error = str(e)

        # Update session status
        await session_manager.update_session(
            session_id, {"status": "error", "error": str(e)}
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"detail": "Resource not found"})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Data Analysis Workflow API")

    # Initialize services
    await session_manager.initialize()

    # Create necessary directories
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Data Analysis Workflow API")

    # Clean up active workflows
    active_workflows.clear()
    workflow_results.clear()

    # Close session manager
    await session_manager.close()

    logger.info("API shutdown complete")


# Main entry point
if __name__ == "__main__":
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    # Run the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
    )
