"""
Flexible LLM provider system supporting Gemini, OpenRouter, and Together.ai.
Provides a unified interface for different LLM providers with automatic failover.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List
from dataclasses import dataclass

import httpx
import google.generativeai as genai
from openai import AsyncOpenAI

from ..config import get_settings
from ..utils import (
    LLMError,
    LLMProviderError,
    LLMProviderNotAvailableError,
    LLMTokenLimitError,
    get_logger,
    log_llm_request,
)


@dataclass
class LLMResponse:
    """Standard response format for all LLM providers."""

    content: str
    provider: str
    model: str
    token_usage: Optional[Dict[str, int]] = None
    processing_time_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.logger = get_logger(__name__)

    @abstractmethod
    async def generate_response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000, **kwargs
    ) -> LLMResponse:
        """Generate response from LLM."""
        pass

    @abstractmethod
    async def stream_response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from LLM."""
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass

    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum token limit for this provider/model."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.max_tokens_limit = 1000000  # Gemini's context window

    async def generate_response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000, **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini."""
        start_time = time.time()

        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature, max_output_tokens=max_tokens, candidate_count=1
            )

            # Safety settings to allow code generation
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                ),
            )

            processing_time = int((time.time() - start_time) * 1000)

            # Extract content
            content = response.text if response.text else ""

            # Extract token usage if available
            token_usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            # Log the request
            log_llm_request(
                provider="gemini",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(content),
                processing_time_ms=processing_time,
                success=True,
            )

            return LLMResponse(
                content=content,
                provider="gemini",
                model=self.model,
                token_usage=token_usage,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            log_llm_request(
                provider="gemini",
                model=self.model,
                prompt_length=len(prompt),
                processing_time_ms=processing_time,
                success=False,
                error=error_msg,
            )

            raise LLMProviderError("gemini", error_msg)

    async def stream_response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from Gemini (currently returns full response)."""
        response = await self.generate_response(
            prompt, temperature, max_tokens, **kwargs
        )
        yield response.content

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens (rough approximation)."""
        return len(text.split()) * 1.3  # Rough estimate

    def get_max_tokens(self) -> int:
        """Get maximum token limit."""
        return self.max_tokens_limit


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        super().__init__(api_key, model, base_url)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.max_tokens_limit = 200000  # Varies by model

    async def generate_response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000, **kwargs
    ) -> LLMResponse:
        """Generate response using OpenRouter."""
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            processing_time = int((time.time() - start_time) * 1000)

            content = response.choices[0].message.content or ""

            # Extract token usage
            token_usage = None
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            log_llm_request(
                provider="openrouter",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(content),
                processing_time_ms=processing_time,
                success=True,
            )

            return LLMResponse(
                content=content,
                provider="openrouter",
                model=self.model,
                token_usage=token_usage,
                processing_time_ms=processing_time,
                metadata={"response_id": response.id},
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            log_llm_request(
                provider="openrouter",
                model=self.model,
                prompt_length=len(prompt),
                processing_time_ms=processing_time,
                success=False,
                error=error_msg,
            )

            raise LLMProviderError("openrouter", error_msg)

    async def stream_response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenRouter."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise LLMProviderError("openrouter", str(e))

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using word count approximation."""
        return int(len(text.split()) * 1.3)

    def get_max_tokens(self) -> int:
        """Get maximum token limit."""
        return self.max_tokens_limit


class TogetherAIProvider(LLMProvider):
    """Together.ai provider implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        base_url: str = "https://api.together.xyz/v1",
    ):
        super().__init__(api_key, model, base_url)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.max_tokens_limit = 131072  # Common for Llama models

    async def generate_response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000, **kwargs
    ) -> LLMResponse:
        """Generate response using Together.ai."""
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            processing_time = int((time.time() - start_time) * 1000)

            content = response.choices[0].message.content or ""

            # Extract token usage
            token_usage = None
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            log_llm_request(
                provider="together",
                model=self.model,
                prompt_length=len(prompt),
                response_length=len(content),
                processing_time_ms=processing_time,
                success=True,
            )

            return LLMResponse(
                content=content,
                provider="together",
                model=self.model,
                token_usage=token_usage,
                processing_time_ms=processing_time,
                metadata={"response_id": response.id},
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            log_llm_request(
                provider="together",
                model=self.model,
                prompt_length=len(prompt),
                processing_time_ms=processing_time,
                success=False,
                error=error_msg,
            )

            raise LLMProviderError("together", error_msg)

    async def stream_response(
        self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from Together.ai."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise LLMProviderError("together", str(e))

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using word count approximation."""
        return int(len(text.split()) * 1.3)

    def get_max_tokens(self) -> int:
        """Get maximum token limit."""
        return self.max_tokens_limit


class LLMManager:
    """Manages multiple LLM providers with automatic failover."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.providers: Dict[str, LLMProvider] = {}
        self.current_provider = self.settings.default_llm_provider
        self.logger = get_logger(__name__)

        # Initialize available providers
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all available LLM providers."""

        # Initialize Gemini
        if self.settings.google_api_key:
            try:
                self.providers["gemini"] = GeminiProvider(
                    api_key=self.settings.google_api_key,
                    model=self.settings.gemini_model,
                )
                self.logger.info("Gemini provider initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini provider: {e}")

        # Initialize OpenRouter
        if self.settings.openrouter_api_key:
            try:
                self.providers["openrouter"] = OpenRouterProvider(
                    api_key=self.settings.openrouter_api_key,
                    model=self.settings.openrouter_model,
                )
                self.logger.info("OpenRouter provider initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenRouter provider: {e}")

        # Initialize Together.ai
        if self.settings.together_api_key:
            try:
                self.providers["together"] = TogetherAIProvider(
                    api_key=self.settings.together_api_key,
                    model=self.settings.together_model,
                )
                self.logger.info("Together.ai provider initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Together.ai provider: {e}")

        if not self.providers:
            raise LLMProviderNotAvailableError("No providers configured", [])

        # Validate current provider
        if self.current_provider not in self.providers:
            available = list(self.providers.keys())
            self.logger.warning(
                f"Default provider '{self.current_provider}' not available. "
                f"Switching to: {available[0]}"
            )
            self.current_provider = available[0]

    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

    def switch_provider(self, provider_name: str) -> None:
        """Switch to a different LLM provider."""
        if provider_name not in self.providers:
            available = self.get_available_providers()
            raise LLMProviderNotAvailableError(provider_name, available)

        self.current_provider = provider_name
        self.logger.info(f"Switched to LLM provider: {provider_name}")

    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        retry_on_failure: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response using specified or current provider.

        Args:
            prompt: Input prompt
            provider: Specific provider to use (optional)
            temperature: Response randomness (0.0-2.0)
            max_tokens: Maximum tokens to generate
            retry_on_failure: Whether to try other providers on failure
            **kwargs: Additional provider-specific parameters
        """
        provider_name = provider or self.current_provider

        # Try primary provider
        if provider_name in self.providers:
            try:
                return await self.providers[provider_name].generate_response(
                    prompt, temperature, max_tokens, **kwargs
                )
            except Exception as e:
                self.logger.error(f"Provider '{provider_name}' failed: {e}")

                if not retry_on_failure or provider:
                    # Don't retry if explicitly requested provider or retry disabled
                    raise

        # Try other providers if retry enabled and no specific provider requested
        if retry_on_failure and not provider:
            for fallback_provider in self.providers:
                if fallback_provider != provider_name:
                    try:
                        self.logger.info(
                            f"Trying fallback provider: {fallback_provider}"
                        )
                        return await self.providers[
                            fallback_provider
                        ].generate_response(prompt, temperature, max_tokens, **kwargs)
                    except Exception as e:
                        self.logger.error(
                            f"Fallback provider '{fallback_provider}' failed: {e}"
                        )
                        continue

        # All providers failed
        available = self.get_available_providers()
        raise LLMProviderNotAvailableError(provider_name or "any", available)

    async def stream(
        self,
        prompt: str,
        provider: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream response using specified or current provider."""
        provider_name = provider or self.current_provider

        if provider_name not in self.providers:
            available = self.get_available_providers()
            raise LLMProviderNotAvailableError(provider_name, available)

        async for chunk in self.providers[provider_name].stream_response(
            prompt, temperature, max_tokens, **kwargs
        ):
            yield chunk

    def estimate_tokens(self, text: str, provider: Optional[str] = None) -> int:
        """Estimate token count for text using specified provider."""
        provider_name = provider or self.current_provider

        if provider_name not in self.providers:
            # Fallback to simple estimation
            return int(len(text.split()) * 1.3)

        return self.providers[provider_name].estimate_tokens(text)

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers."""
        info = {
            "current_provider": self.current_provider,
            "available_providers": self.get_available_providers(),
            "provider_details": {},
        }

        for name, provider in self.providers.items():
            info["provider_details"][name] = {
                "model": provider.model,
                "max_tokens": provider.get_max_tokens(),
                "base_url": provider.base_url,
            }

        return info


# Global LLM manager instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def initialize_llm_manager(settings=None) -> LLMManager:
    """Initialize the global LLM manager with custom settings."""
    global _llm_manager
    _llm_manager = LLMManager(settings)
    return _llm_manager
