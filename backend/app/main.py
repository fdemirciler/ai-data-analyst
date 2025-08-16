from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
import sys
from pathlib import Path

# Ensure project root (parent of backend) is on sys.path BEFORE importing route modules
backend_root = Path(__file__).resolve().parents[2]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from .config import settings
from .routes import upload  # noqa: F401
from .routes.upload import router as upload_router
from .routes.chat import router as chat_router
from .session import initialize_session_store


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Data Analysis Backend",
        version="0.1.0",
        default_response_class=ORJSONResponse,
    )

    # sys.path already adjusted at import time for reload compatibility

    # Enable CORS for local development to allow the Vite dev server to call the API
    if settings.environment in {"dev", "development"}:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5174",
                "http://127.0.0.1:5174",
                "http://localhost:5175",
                "http://127.0.0.1:5175",
                "https://aidataanalyst.fly.dev",
                "https://ai-data-analyst-two.vercel.app",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    def health():  # pragma: no cover
        return {"status": "ok", "env": settings.environment}

    app.include_router(upload_router)
    app.include_router(chat_router)

    # Initialize session store (Redis or in-memory) on startup
    @app.on_event("startup")
    async def _startup():  # pragma: no cover - initialization side-effect
        initialize_session_store()
    # Startup diagnostics (non-secret)
    print(
        "Config: env=",
        settings.environment,
        " enable_llm=",
        settings.enable_llm,
        " gemini_key_present=",
        bool(settings.gemini_api_key),
    )
    return app


app = create_app()
