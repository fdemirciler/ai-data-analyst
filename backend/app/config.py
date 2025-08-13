from __future__ import annotations

import os
from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    # General
    environment: str = "dev"
    enable_llm: bool = True
    enable_docker_sandbox: bool = False

    # LLM Provider: 'google' or 'together'
    llm_provider: str = "google"

    # Google Gemini Settings
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"

    # Together.ai Settings
    together_api_key: str | None = None
    together_model: str = "meta-llama/Llama-3.1-70B-Instruct-Turbo"

    # File Uploads
    max_upload_mb: int = 50

    class Config:
        arbitrary_types_allowed = True


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    # Lazy load dotenv if available
    try:
        from dotenv import load_dotenv  # type: ignore

        # Load default .env from CWD or nearest
        load_dotenv(override=True)  # Override to ensure .env is prioritized
        # Additionally try project root .env and '@.env'
        try:
            # Project root (two levels up from backend/app)
            root = Path(__file__).resolve().parents[2]
        except Exception:
            root = Path.cwd()
        for name in (".env", "@.env"):
            env_path = root / name
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=True)  # Prioritize .env
    except Exception:
        pass

    # Support alternate env name for Google Generative AI
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    return Settings(
        environment=os.getenv("ENV", "dev"),
        enable_llm=_env_bool("ENABLE_LLM", True),
        # Allow legacy SANDBOX_ENABLED plus new ENABLE_DOCKER_SANDBOX
        enable_docker_sandbox=_env_bool(
            "ENABLE_DOCKER_SANDBOX",
            _env_bool("SANDBOX_ENABLED", False),
        ),
        llm_provider=os.getenv("LLM_PROVIDER", "google"),
        gemini_api_key=gemini_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        together_model=os.getenv(
            "TOGETHER_MODEL", "meta-llama/Llama-3.1-70B-Instruct-Turbo"
        ),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
    )


settings = load_settings()
