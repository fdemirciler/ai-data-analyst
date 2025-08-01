"""
Configuration management for the Agentic Data Analysis Workflow.
Handles environment variables, LLM provider settings, and application configuration.
"""

import os
from typing import List, Optional, Literal, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation and environment variable support."""

    # Application Configuration
    app_name: str = "Agentic Data Analysis Workflow"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_url: str = "http://localhost:3000"

    # Redis Configuration
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    session_ttl: int = 3600  # 1 hour in seconds

    # LLM Provider Configuration
    default_llm_provider: Literal["gemini", "openrouter", "together"] = "gemini"

    # Primary: Google Gemini
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash-exp"

    # Secondary: OpenRouter
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-3.5-sonnet"

    # Tertiary: Together.ai
    together_api_key: Optional[str] = None
    together_model: str = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"

    # Data Storage Configuration
    data_storage_path: str = "/app/data"
    parquet_storage_path: str = "/app/data/parquet"
    upload_max_size: int = 52428800  # 50MB
    allowed_file_extensions: List[str] = ["csv", "xlsx", "json", "parquet"]

    # Security Configuration
    secret_key: str = "dev-secret-key-change-in-production-min-32-chars"
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    allowed_hosts: List[str] = ["localhost", "127.0.0.1"]

    # Code Execution Security
    execution_timeout: int = 30  # seconds
    max_memory_mb: int = 512
    sandbox_enabled: bool = True

    # Workflow Configuration
    max_retries: int = 3
    max_chat_history: int = 50
    workflow_timeout: int = 300  # 5 minutes

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_file: str = "/app/logs/app.log"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @field_validator("google_api_key", "openrouter_api_key", "together_api_key")
    @classmethod
    def validate_api_keys(cls, v, info):
        """Ensure at least one API key is provided."""
        return v

    @field_validator("data_storage_path", "parquet_storage_path", "log_file")
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist for file paths."""
        if v and "/" in v:
            Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_comma_separated(cls, v):
        """Parse comma-separated strings into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("allowed_file_extensions", mode="before")
    @classmethod
    def parse_file_extensions(cls, v):
        """Parse file extensions and ensure they start with a dot."""
        if isinstance(v, str):
            extensions = [ext.strip() for ext in v.split(",") if ext.strip()]
        else:
            extensions = v

        # Ensure extensions start with a dot
        return [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    def get_provider_config(self, provider_name: str) -> dict:
        """Get configuration for a specific LLM provider."""
        configs = {
            "gemini": {
                "api_key": self.google_api_key,
                "model": self.gemini_model,
                "base_url": None,
            },
            "openrouter": {
                "api_key": self.openrouter_api_key,
                "model": self.openrouter_model,
                "base_url": "https://openrouter.ai/api/v1",
            },
            "together": {
                "api_key": self.together_api_key,
                "model": self.together_model,
                "base_url": "https://api.together.xyz/v1",
            },
        }

        if provider_name not in configs:
            raise ValueError(f"Unknown provider: {provider_name}")

        return configs[provider_name]

    def get_available_providers(self) -> List[str]:
        """Get list of providers with valid API keys."""
        providers = []

        if self.google_api_key:
            providers.append("gemini")
        if self.openrouter_api_key:
            providers.append("openrouter")
        if self.together_api_key:
            providers.append("together")

        return providers


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def validate_configuration():
    """Validate the current configuration and log any issues."""
    import logging

    logger = logging.getLogger(__name__)

    # Check if at least one LLM provider is configured
    available_providers = settings.get_available_providers()
    if not available_providers:
        raise ValueError(
            "No LLM providers configured. Please set at least one API key."
        )

    logger.info(f"Available LLM providers: {', '.join(available_providers)}")
    logger.info(f"Default provider: {settings.default_llm_provider}")

    # Validate default provider is available
    if settings.default_llm_provider not in available_providers:
        logger.warning(
            f"Default provider '{settings.default_llm_provider}' not available. "
            f"Available providers: {', '.join(available_providers)}"
        )

    # Check data directories
    storage_path = Path(settings.data_storage_path)
    if not storage_path.exists():
        logger.info(f"Creating data storage directory: {storage_path}")
        storage_path.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(settings.parquet_storage_path)
    if not parquet_path.exists():
        logger.info(f"Creating Parquet storage directory: {parquet_path}")
        parquet_path.mkdir(parents=True, exist_ok=True)

    # Check log directory
    log_dir = Path(settings.log_file).parent
    if not log_dir.exists():
        logger.info(f"Creating log directory: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Configuration validation completed successfully")


if __name__ == "__main__":
    # For testing configuration
    try:
        validate_configuration()
        print("Configuration is valid!")
        print(f"Available providers: {settings.get_available_providers()}")
        print(f"Redis URL: {settings.redis_url}")
    except Exception as e:
        print(f"Configuration error: {e}")
