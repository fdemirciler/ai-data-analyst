"""
Simplified FastAPI Main Application
==================================

This is a basic FastAPI server that provides essential REST API endpoints
for the data analysis workflow system, designed to work with existing architecture.
"""

import os
import uuid
import asyncio
import logging
import re
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Add the 'backend' directory to sys.path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Import basic components that we know work
from security import SecurePythonExecutor, ExecutionLimits

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
        "http://localhost:5174",  # Frontend is running on this port
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Secure code executor for custom analysis
secure_executor = SecurePythonExecutor(
    execution_limits=ExecutionLimits(
        max_execution_time=60.0,  # Allow longer execution for analysis
        max_memory_mb=512,
        max_output_size=50000,
    )
)

# Simple in-memory storage for development (replace with Redis/DB in production)
active_sessions: Dict[str, Dict[str, Any]] = {}
session_results: Dict[str, Dict[str, Any]] = {}
uploaded_files: Dict[str, Dict[str, Any]] = {}


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


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "services": {"fastapi": "active", "security": "active", "sandbox": "active"},
    }


# Session management endpoints
@app.post("/api/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new analysis session"""
    try:
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now(),
            "status": "active",
            "last_accessed": datetime.now(),
        }

        active_sessions[session_id] = session_data

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
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = active_sessions[session_id]
        session_data["last_accessed"] = datetime.now()

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
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Clean up all session data
        del active_sessions[session_id]
        if session_id in session_results:
            del session_results[session_id]
        if session_id in uploaded_files:
            del uploaded_files[session_id]

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
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate file type
        allowed_extensions = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}",
            )

        # Create upload directory if it doesn't exist
        upload_dir = Path("uploads") / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Immediately process file to get dimensions
        import pandas as pd

        try:
            # Load data based on file type to get dimensions
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                # For other types, set default values
                df = None

            if df is not None and not df.empty:
                rows, columns = df.shape
                logger.info(
                    f"Successfully processed file {file.filename}: {rows} rows × {columns} columns"
                )
            else:
                rows, columns = 0, 0
                logger.warning(
                    f"File {file.filename} processed but resulted in empty dataframe"
                )

        except Exception as e:
            logger.error(f"Error processing file dimensions for {file.filename}: {e}")
            # Set reasonable defaults instead of 0,0 to help with debugging
            rows, columns = -1, -1

        # Store file info with dimensions
        file_info = {
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "upload_timestamp": datetime.now(),
            "file_type": file_extension,
            "rows": rows,
            "columns": columns,
        }

        uploaded_files[session_id] = file_info
        active_sessions[session_id]["uploaded_file"] = file_info
        active_sessions[session_id]["status"] = "file_uploaded"

        logger.info(
            f"File uploaded for session {session_id}: {file.filename} ({rows} rows × {columns} columns)"
        )

        # Return response in format expected by frontend
        return {
            "filename": file.filename,
            "rows": rows,
            "columns": columns,
            "file_id": f"{session_id}_{file.filename}",
            "upload_time": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Basic data analysis endpoint (simplified)
@app.post("/api/sessions/{session_id}/analyze", response_model=WorkflowStatusResponse)
async def start_analysis(
    session_id: str, request: AnalysisRequest, background_tasks: BackgroundTasks
):
    """Start basic data analysis"""
    try:
        # Verify session and file
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = active_sessions[session_id]
        if "uploaded_file" not in session_data:
            raise HTTPException(
                status_code=400, detail="No file uploaded for this session"
            )

        # Update session status
        session_data["status"] = "processing"
        session_data["analysis_type"] = request.analysis_type
        session_data["parameters"] = request.parameters

        # Start basic analysis in background
        background_tasks.add_task(
            run_basic_analysis, session_id, session_data["uploaded_file"]
        )

        logger.info(f"Started basic analysis for session {session_id}")

        return WorkflowStatusResponse(
            session_id=session_id,
            status="processing",
            current_step="analyzing",
            progress=0.0,
            message="Basic analysis started",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/status", response_model=WorkflowStatusResponse)
async def get_analysis_status(session_id: str):
    """Get current analysis status"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = active_sessions[session_id]
        status = session_data.get("status", "active")

        # Check if analysis is complete
        if session_id in session_results:
            return WorkflowStatusResponse(
                session_id=session_id,
                status="completed",
                current_step="finished",
                progress=1.0,
                message="Analysis completed successfully",
            )
        elif status == "processing":
            return WorkflowStatusResponse(
                session_id=session_id,
                status="processing",
                current_step="analyzing",
                progress=0.5,
                message="Analysis in progress",
            )
        else:
            return WorkflowStatusResponse(
                session_id=session_id,
                status=status,
                current_step="ready",
                progress=0.0,
                message="Ready for analysis",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/results", response_model=AnalysisResultResponse)
async def get_analysis_results(session_id: str):
    """Get analysis results"""
    try:
        if session_id not in session_results:
            if session_id not in active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")

            session_data = active_sessions[session_id]
            if session_data.get("status") == "processing":
                raise HTTPException(
                    status_code=202, detail="Analysis still in progress"
                )
            else:
                raise HTTPException(status_code=404, detail="No results available")

        results = session_results[session_id]

        return AnalysisResultResponse(
            session_id=session_id,
            analysis_type=results.get("analysis_type", "basic"),
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


# Custom code execution endpoint
@app.post("/api/sessions/{session_id}/execute")
async def execute_custom_code(session_id: str, request: CustomCodeRequest):
    """Execute custom Python code in secure sandbox"""
    try:
        # Verify session exists
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get context data if available
        context_data = {}
        if session_id in session_results:
            # Add analysis results to context
            results = session_results[session_id]
            if "dataframe" in results:
                context_data["df"] = results["dataframe"]
            if "analysis_results" in results:
                context_data["analysis_results"] = results["analysis_results"]

        # If no dataframe is loaded but we have an uploaded file, load it now
        if "df" not in context_data and session_id in active_sessions:
            session_data = active_sessions[session_id]
            if "uploaded_file" in session_data:
                file_info = session_data["uploaded_file"]
                try:
                    import pandas as pd

                    df = pd.read_csv(file_info["file_path"])
                    context_data["df"] = df
                    logger.info(
                        f"Loaded dataframe for session {session_id}: {df.shape}"
                    )
                    logger.info(f"Dataframe columns: {list(df.columns)}")
                    logger.info(f"Dataframe head: {df.head().to_string()}")
                except Exception as e:
                    logger.error(
                        f"Failed to load dataframe for session {session_id}: {e}"
                    )

        # Add any additional context variables
        context_data.update(request.context_variables)

        # Debug: log what's in context_data
        logger.info(f"Context data keys: {list(context_data.keys())}")
        if "df" in context_data:
            logger.info(
                f"DataFrame in context: {type(context_data['df'])}, shape: {context_data['df'].shape}"
            )
        else:
            logger.info("No 'df' found in context_data")

        # Execute code securely
        execution_result = secure_executor.execute(request.code, context_data)

        logger.info(f"Executed custom code for session {session_id}")
        logger.info(f"Execution success: {execution_result.success}")
        logger.info(f"Execution output length: {len(execution_result.output or '')}")
        logger.info(f"Execution output: {repr(execution_result.output)}")
        logger.info(f"Execution error: {execution_result.error}")

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


# ===============================================
# 🧠 REAL LLM INTEGRATION ENDPOINT
# ===============================================


@app.post("/api/sessions/{session_id}/analyze-llm")
async def analyze_with_llm_endpoint(
    session_id: str, query: str = Query(..., description="User's analysis question")
):
    """
    🧠 REAL LLM ANALYSIS ENDPOINT

    This endpoint uses real LLM integration for intelligent data analysis.
    It generates contextual Python code based on user queries and data structure.
    """
    try:
        logger.info(f"🧠 Starting REAL LLM analysis for session {session_id}")
        logger.info(f"Query: {query}")

        # Verify session exists
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = active_sessions[session_id]

        # Check if file was uploaded
        if "uploaded_file" not in session_data:
            raise HTTPException(
                status_code=400, detail="No data file uploaded for analysis"
            )

        # Get file information
        file_info = session_data["uploaded_file"]
        file_path = file_info["file_path"]

        # Load data to understand structure
        try:
            if file_path.endswith(".csv"):
                import pandas as pd

                df = pd.read_csv(file_path)
            elif file_path.endswith((".xlsx", ".xls")):
                import pandas as pd

                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type")

            logger.info(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load data: {str(e)}"
            )

        # Generate LLM-powered analysis
        try:
            # Import LLM components
            from services.llm_provider import LLMManager
            from config import settings

            # Initialize LLM manager
            llm_manager = LLMManager(settings)

            # Create context-aware prompt
            prompt = f"""You are an expert data analyst. Generate Python code to analyze the uploaded dataset and answer the user's question.

DATASET INFO:
- Filename: {file_info['filename']}
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Data types: {dict(df.dtypes)}

SAMPLE DATA (first 3 rows):
{df.head(3).to_string()}

USER QUERY: "{query}"

Generate Python code that:
1. Analyzes the data to answer the user's question
2. Uses pandas for data manipulation
3. Creates appropriate visualizations if helpful using matplotlib/seaborn
4. Provides clear, interpretable results
5. The data is already loaded as 'df'

Respond with only the Python code in this format:
```python
# Analysis code here
```

Then provide a brief explanation of what the code does."""

            logger.debug("Generating code with LLM...")
            llm_response = await llm_manager.generate(
                prompt, provider="gemini", temperature=0.1, max_tokens=2000
            )

            # Extract code from response
            import re

            code_match = re.search(
                r"```python\s*(.*?)\s*```", llm_response.content, re.DOTALL
            )
            if code_match:
                generated_code = code_match.group(1).strip()
            else:
                generated_code = llm_response.content.strip()

            logger.info(f"Generated code length: {len(generated_code)} chars")
            logger.debug(f"Generated code:\n---\n{generated_code}\n---")

            # Execute the generated code
            context_data = {"df": df}
            execution_result = secure_executor.execute(generated_code, context_data)

            logger.info(
                f"LLM-generated code executed. Success: {execution_result.success}"
            )
            if not execution_result.success:
                logger.error(f"Execution error: {execution_result.error}")

            # Prepare response
            response = {
                "success": execution_result.success,
                "session_id": session_id,
                "query": query,
                "analysis_id": str(uuid.uuid4()),
                "generated_code": generated_code,
                "code_explanation": "LLM-generated analysis code",
                "execution_output": execution_result.output,
                "execution_error": execution_result.error,
                "processing_time": execution_result.execution_time,
                "timestamp": datetime.now().isoformat(),
                "retry_count": 0,
                "visualizations": [],
                "analysis_type": "llm",
            }

            if execution_result.success:
                response["interpretation"] = (
                    f"Successfully analyzed your data. {execution_result.output}"
                )
            else:
                response["interpretation"] = (
                    f"Analysis encountered an issue: {execution_result.error}"
                )

            # Add response elements for frontend
            response_elements = []

            if generated_code:
                response_elements.append(
                    {
                        "type": "code",
                        "content": generated_code,
                        "title": "Generated Analysis Code",
                        "explanation": "AI-generated code to answer your question",
                    }
                )

            if execution_result.output:
                response_elements.append(
                    {
                        "type": "output",
                        "content": execution_result.output,
                        "title": "Analysis Results",
                    }
                )

            if execution_result.error:
                response_elements.append(
                    {
                        "type": "error",
                        "content": execution_result.error,
                        "title": "Execution Error",
                    }
                )

            response["response_elements"] = response_elements

            logger.info(f"✅ LLM analysis completed: success={response['success']}")
            return response

        except ImportError as e:
            logger.warning(
                f"LLM components not available, falling back to rule-based analysis. Error: {e}"
            )
            # Fallback to rule-based generation
            return await _fallback_analysis(session_id, query, df, file_info)

        except Exception as e:
            logger.error(
                f"LLM analysis failed, falling back to rule-based analysis. Error: {e}"
            )
            # Fallback to rule-based generation
            return await _fallback_analysis(session_id, query, df, file_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Error in LLM analysis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")


async def _fallback_analysis(
    session_id: str, query: str, df, file_info: Dict[str, Any]
):
    """Fallback analysis when LLM is not available"""
    logger.info("Using fallback analysis (LLM not available)")

    query_lower = query.lower()
    df.columns = [col.lower() for col in df.columns]  # Normalize column names

    # Generate appropriate code based on query
    if "average" in query_lower and "department" in query_lower:
        code = """
# Calculate average salary per department
if 'department' in df.columns and any('salary' in col for col in df.columns):
    # Find salary column
    salary_col = next(col for col in df.columns if 'salary' in col)
    avg_by_dept = df.groupby('department')[salary_col].mean().sort_values(ascending=False)
    
    print("Average Salary by Department:")
    print("=" * 30)
    for dept, avg_salary in avg_by_dept.items():
        print(f"{dept}: ${avg_salary:,.2f}")
    
    # Create visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    avg_by_dept.plot(kind='bar')
    plt.title(f'Average {salary_col.title()} by Department')
    plt.xlabel('Department')
    plt.ylabel(f'Average {salary_col.title()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Could not find 'department' and 'salary' columns after normalization.")
    print("Available columns:", list(df.columns))
"""

    elif "summary" in query_lower or "overview" in query_lower:
        code = """
# Data overview and summary
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

print("\\n=== DATA TYPES ===")
print(df.dtypes)

print("\\n=== MISSING VALUES ===")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Missing values per column:")
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
else:
    print("✅ No missing values found!")

print("\\n=== SAMPLE DATA ===")
print("First 5 rows:")
print(df.head())

print("\\n=== SUMMARY STATISTICS ===")
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    print("Numeric columns summary:")
    print(df[numeric_cols].describe())
"""
    else:
        code = f"""
# Basic analysis for: {query}
print("=== ANALYSIS RESULTS ===")
print(f"Dataset shape: {{df.shape[0]}} rows × {{df.shape[1]}} columns")
print(f"Columns: {{list(df.columns)}}")

# Show basic statistics
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    print("\\nNumeric columns summary:")
    print(df[numeric_cols].describe())
    
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\\nCategorical columns:")
    for col in categorical_cols[:3]:
        print(f"\\n{{col}} value counts:")
        print(df[col].value_counts().head())
"""

    # Execute the code
    context_data = {"df": df}
    execution_result = secure_executor.execute(code, context_data)

    # Return response
    return {
        "success": execution_result.success,
        "session_id": session_id,
        "query": query,
        "analysis_id": str(uuid.uuid4()),
        "generated_code": code,
        "code_explanation": "Rule-based analysis code (LLM fallback)",
        "execution_output": execution_result.output,
        "execution_error": execution_result.error,
        "processing_time": execution_result.execution_time,
        "timestamp": datetime.now().isoformat(),
        "retry_count": 0,
        "visualizations": [],
        "interpretation": (
            execution_result.output
            if execution_result.success
            else f"Analysis failed: {execution_result.error}"
        ),
        "response_elements": [
            {
                "type": "code",
                "content": code,
                "title": "Generated Analysis Code",
                "explanation": "Fallback analysis code",
            },
            {
                "type": "output",
                "content": execution_result.output,
                "title": "Analysis Results",
            },
        ],
    }


# Background analysis function
async def run_basic_analysis(session_id: str, file_info: Dict[str, Any]):
    """Run basic data analysis in the background"""
    try:
        logger.info(f"Starting basic analysis for session {session_id}")

        # Simulate basic data analysis
        import pandas as pd

        file_path = file_info["file_path"]
        file_extension = Path(file_path).suffix.lower()

        # Load data based on file type
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Basic analysis
        analysis_results = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_summary": (
                df.describe().to_dict()
                if len(df.select_dtypes(include="number").columns) > 0
                else {}
            ),
        }

        # Generate basic insights
        insights = [
            f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns",
            f"Memory usage: {analysis_results['memory_usage'] / 1024 / 1024:.2f} MB",
            f"Total missing values: {df.isnull().sum().sum()}",
        ]

        # Store results
        session_results[session_id] = {
            "analysis_type": "basic",
            "analysis_results": analysis_results,
            "dataframe": df,  # Keep for custom code execution
            "visualizations": [],
            "insights": insights,
            "execution_time": 2.0,  # Simulated
            "timestamp": datetime.now(),
        }

        # Update session status
        active_sessions[session_id]["status"] = "completed"

        logger.info(f"Completed basic analysis for session {session_id}")

    except Exception as e:
        logger.error(f"Error in basic analysis for session {session_id}: {e}")
        active_sessions[session_id]["status"] = "error"
        active_sessions[session_id]["error"] = str(e)


# Main entry point
if __name__ == "__main__":
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    # Run the server
    uvicorn.run(
        "main_simple:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
    )
