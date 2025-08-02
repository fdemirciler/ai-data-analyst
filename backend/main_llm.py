"""
Simplified FastAPI Main Application with LLM Integration

This is a simplified version of the main FastAPI server that focuses on
providing the core LLM analysis endpoint that replaces the fake AI logic.
"""

import os
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our core services
from services.session_manager import SessionManager
from services.file_handler import FileHandler
from security import SecurePythonExecutor, ExecutionLimits
from config.settings import get_settings

# Import LLM endpoints
from api.llm_endpoints import (
    analyze_with_llm,
    AnalysisRequest as LLMAnalysisRequest,
    AnalysisResponse as LLMAnalysisResponse,
    set_session_storage,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Analysis Workflow API with LLM Integration",
    description="AI-powered data analysis platform with real LLM integration",
    version="2.0.0",
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

# Secure code executor for custom analysis
secure_executor = SecurePythonExecutor(
    execution_limits=ExecutionLimits(
        max_execution_time=60.0,  # Allow longer execution for analysis
        max_memory_mb=512,
        max_output_size=50000,
    )
)


# Basic Pydantic models for API
class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    status: str = "active"


class CustomCodeRequest(BaseModel):
    session_id: str
    code: str = Field(..., description="Python code to execute")
    context_variables: Dict[str, Any] = Field(default_factory=dict)


# Global storage for session data (in production, use Redis/database)
session_storage: Dict[str, Dict[str, Any]] = {}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "2.0.0",
        "llm_integration": "active",
        "services": {
            "session_manager": "active",
            "file_handler": "active",
            "llm_system": "active",
        },
    }


# Session management endpoints
@app.post("/api/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new analysis session"""
    try:
        session_id = str(uuid.uuid4())

        # Create basic session data
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "status": "active",
            "conversation_history": [],
        }

        # Store in our simple storage
        session_storage[session_id] = session_data

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
        if session_id not in session_storage:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = session_storage[session_id]

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


# File upload endpoints
@app.post("/api/sessions/{session_id}/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload a data file for analysis"""
    try:
        # Verify session exists
        if session_id not in session_storage:
            raise HTTPException(status_code=404, detail="Session not found")

        # Validate file type
        allowed_extensions = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
        file_extension = Path(file.filename or "unknown").suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}",
            )

        # Save file (simplified - create uploads directory if needed)
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)

        file_path = uploads_dir / f"{session_id}_{file.filename}"

        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Create basic file info
        file_info = {
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "uploaded_at": datetime.now().isoformat(),
        }

        # Try to get basic info about the file
        try:
            if file_extension == ".csv":
                import pandas as pd

                df = pd.read_csv(file_path)
                file_info.update(
                    {
                        "rows": len(df),
                        "columns": len(df.columns),
                    }
                )
            elif file_extension in [".xlsx", ".xls"]:
                import pandas as pd

                df = pd.read_excel(file_path)
                file_info.update(
                    {
                        "rows": len(df),
                        "columns": len(df.columns),
                    }
                )
            else:
                file_info.update(
                    {
                        "rows": 0,
                        "columns": 0,
                    }
                )
        except Exception as e:
            logger.warning(f"Could not parse file info: {e}")
            file_info.update(
                {
                    "rows": 0,
                    "columns": 0,
                }
            )

        # Update session with file info
        session_storage[session_id]["uploaded_file"] = file_info
        session_storage[session_id]["status"] = "file_uploaded"

        logger.info(f"File uploaded for session {session_id}: {file.filename}")

        return {
            "message": "File uploaded successfully",
            "filename": file_info["filename"],
            "rows": file_info["rows"],
            "columns": file_info["columns"],
            "file_id": session_id,  # Use session_id as file_id for simplicity
            "upload_time": file_info["uploaded_at"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===============================================
# 🚀 REAL LLM INTEGRATION ENDPOINT - THE MAIN EVENT!
# ===============================================


@app.post("/api/sessions/{session_id}/analyze-llm", response_model=LLMAnalysisResponse)
async def analyze_with_llm_endpoint(
    session_id: str, query: str = Query(..., description="User's analysis question")
):
    """
    🧠 REAL LLM ANALYSIS ENDPOINT

    This endpoint replaces the fake AI logic with actual LLM integration:
    - Uses your sophisticated LangGraph workflow system
    - Integrates with real LLM providers (Gemini, OpenRouter, Together.ai)
    - Provides intelligent code generation and analysis
    - Includes context-aware conversation history

    This is where the magic happens - no more fake templates!
    """
    try:
        logger.info(f"🧠 Starting REAL LLM analysis for session {session_id}")
        logger.info(f"Query: {query}")

        # Verify session exists
        if session_id not in session_storage:
            raise HTTPException(status_code=404, detail="Session not found")

        # Create LLM analysis request
        request = LLMAnalysisRequest(
            query=query, session_id=session_id, stream=False, max_retries=2
        )

        # Call the REAL LLM analysis service
        logger.debug("Calling real LLM analysis service...")
        response = await analyze_with_llm(request)

        logger.info(
            f"🎉 LLM analysis completed for session {session_id}: success={response.success}"
        )
        logger.debug(f"Processing time: {response.processing_time:.2f}s")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Error in REAL LLM analysis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")


# Custom code execution endpoint (for direct code execution)
@app.post("/api/sessions/{session_id}/execute")
async def execute_custom_code(session_id: str, request: CustomCodeRequest):
    """Execute custom Python code in secure sandbox"""
    try:
        # Verify session exists
        if session_id not in session_storage:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = session_storage[session_id]

        # Prepare context with uploaded data if available
        context_data = {}

        # Load data if file was uploaded
        if "uploaded_file" in session_data:
            file_info = session_data["uploaded_file"]
            file_path = file_info["file_path"]

            try:
                # Load the data into context
                if file_path.endswith(".csv"):
                    import pandas as pd

                    context_data["df"] = pd.read_csv(file_path)
                elif file_path.endswith((".xlsx", ".xls")):
                    import pandas as pd

                    context_data["df"] = pd.read_excel(file_path)
                elif file_path.endswith(".parquet"):
                    import pandas as pd

                    context_data["df"] = pd.read_parquet(file_path)

                logger.info(
                    f"Loaded data for execution: {context_data['df'].shape if 'df' in context_data else 'No data'}"
                )
            except Exception as e:
                logger.warning(f"Could not load data for execution: {e}")

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
            "memory_used": getattr(execution_result, "memory_used", 0),
            "warnings": getattr(execution_result, "warnings", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing custom code for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Data Analysis Workflow API with REAL LLM Integration")

    # Set session storage for LLM service
    set_session_storage(session_storage)

    # Create necessary directories
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    logger.info("✅ API startup complete - LLM integration ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Data Analysis Workflow API")

    # Clean up session storage
    session_storage.clear()

    logger.info("API shutdown complete")


# Main entry point
if __name__ == "__main__":
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    # Run the server
    uvicorn.run(
        "main_llm:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
    )
