from __future__ import annotations

import os
from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    # General
    environment: str = "dev"
    enable_llm: bool = True
    enable_docker_sandbox: bool = False
    # Session backend
    enable_redis_sessions: bool = False
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int | None = 86400
    redis_key_prefix: str = "ai-da"
    # Session trimming caps (keep last N items)
    redis_max_messages: int = 10
    redis_max_artifacts: int = 10

    # LLM Provider: 'google' or 'together'
    llm_provider: str = "google"
    # Sampling / decoding
    llm_temperature: float = 0.3

    # Google Gemini Settings
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # Together.ai Settings
    together_api_key: str | None = None
    together_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

    # File Uploads
    max_upload_mb: int = 50

    # Structured Summary / Prompt composition
    summary_max_tables: int = 5
    summary_max_rows: int = 50
    summary_max_cols: int = 15
    summary_prompt_char_budget: int = 60000
    summary_include_errors_limit: int = 5
    # Max length for summarized LLM answer text (aggregated), 0 disables cap
    summary_max_answer_chars: int = 12000

    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}


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

    # Parse optional TTL
    ttl_raw = os.getenv("SESSION_TTL_SECONDS", "86400")
    ttl_val: int | None
    try:
        ttl_int = int(ttl_raw)
        ttl_val = ttl_int if ttl_int > 0 else None
    except Exception:
        ttl_val = None

    # Parse optional temperature
    temp_raw = os.getenv("LLM_TEMPERATURE", "0.3")
    try:
        temp_val = float(temp_raw)
    except Exception:
        temp_val = 0.3

    # Structured summary env overrides
    try:
        sum_max_tables = int(os.getenv("SUMMARY_MAX_TABLES", "5"))
    except Exception:
        sum_max_tables = 5
    try:
        sum_max_rows = int(os.getenv("SUMMARY_MAX_ROWS", "50"))
    except Exception:
        sum_max_rows = 50
    try:
        sum_max_cols = int(os.getenv("SUMMARY_MAX_COLS", "15"))
    except Exception:
        sum_max_cols = 15
    try:
        sum_prompt_budget = int(os.getenv("SUMMARY_PROMPT_CHAR_BUDGET", "60000"))
    except Exception:
        sum_prompt_budget = 60000
    try:
        sum_err_limit = int(os.getenv("SUMMARY_INCLUDE_ERRORS_LIMIT", "5"))
    except Exception:
        sum_err_limit = 5
    try:
        sum_max_answer_chars = int(os.getenv("SUMMARY_MAX_ANSWER_CHARS", "12000"))
    except Exception:
        sum_max_answer_chars = 12000

    return Settings(
        environment=os.getenv("ENV", "dev"),
        enable_llm=_env_bool("ENABLE_LLM", True),
        # Allow legacy SANDBOX_ENABLED plus new ENABLE_DOCKER_SANDBOX
        enable_docker_sandbox=_env_bool(
            "ENABLE_DOCKER_SANDBOX",
            _env_bool("SANDBOX_ENABLED", False),
        ),
        enable_redis_sessions=_env_bool("ENABLE_REDIS_SESSIONS", False),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        session_ttl_seconds=ttl_val,
        redis_key_prefix=os.getenv("REDIS_KEY_PREFIX", "ai-da"),
        redis_max_messages=int(os.getenv("REDIS_MAX_MESSAGES", "10")),
        redis_max_artifacts=int(os.getenv("REDIS_MAX_ARTIFACTS", "10")),
        llm_provider=os.getenv("LLM_PROVIDER", "google"),
        llm_temperature=temp_val,
        gemini_api_key=gemini_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        together_model=os.getenv(
            "TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        ),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
        summary_max_tables=sum_max_tables,
        summary_max_rows=sum_max_rows,
        summary_max_cols=sum_max_cols,
        summary_prompt_char_budget=sum_prompt_budget,
        summary_include_errors_limit=sum_err_limit,
        summary_max_answer_chars=sum_max_answer_chars,
    )


settings = load_settings()
