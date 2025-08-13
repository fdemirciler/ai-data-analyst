"""FastAPI backend application package for the agentic data analysis workflow.

Current MVP scope:
 - /health endpoint
 - /api/upload endpoint: single file upload (<50MB) -> process via data_processing_profiling pipeline
 - In-memory session store (UUID based)

Later phases (not yet implemented here):
 - /api/chat endpoint using LangGraph orchestrated agent
 - Docker sandbox execution & script safety validation
 - Streaming responses
"""

from .main import create_app  # noqa: F401
