"""Lightweight LLM wrapper interfaces used by reasoners."""
from __future__ import annotations

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Minimal synchronous chat‑LLM interface.

    • Accepts a list[dict] *messages* like the OpenAI Chat format.
    • Returns *content* (str) of the assistant reply.
    • Implementations SHOULD be stateless; auth + model name given at init.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))
    max_tokens: Optional[int] = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS")) if os.getenv("LLM_MAX_TOKENS") else None)
    timeout: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "60")))
    enable_cost_tracking: bool = field(default_factory=lambda: os.getenv("LLM_ENABLE_COST_TRACKING", "true").lower() == "true")
    fallback_models: Optional[List[str]] = field(default_factory=lambda: os.getenv("LLM_FALLBACK_MODELS", "").split(",") if os.getenv("LLM_FALLBACK_MODELS") else None)


class LiteLLMChatLLM(BaseLLM):
    """Wrapper around litellm.completion with cost tracking and fallback support."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        config: Optional[LLMConfig] = None,
    ) -> None:
        import litellm
        
        # Use provided config or create from individual params/env
        if config:
            self.config = config
        else:
            self.config = LLMConfig()
            # Override with any explicitly provided params
            if model is not None:
                self.config.model = model
            if temperature is not None:
                self.config.temperature = temperature
            if max_tokens is not None:
                self.config.max_tokens = max_tokens
        
        # Check if model is configured
        if not self.config.model:
            logger.error("No LLM model configured! Please set LLM_MODEL in your .env file.")
            raise ValueError("No LLM model configured. Set LLM_MODEL environment variable or pass a model parameter.")
        
        self._client = litellm
        
        # Set timeout
        litellm.request_timeout = self.config.timeout
        
        # Cost tracking state
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_tokens = 0
        
        logger.info(f"Initialized LLM with model: {self.config.model}")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to the LLM and return the response."""
        # Use fallback if configured
        if self.config.fallback_models:
            return self._chat_with_fallback(messages, **kwargs)
        else:
            return self._chat_single(self.config.model, messages, **kwargs)

    def _chat_single(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Execute a single chat call."""
        start_time = time.time()
        
        try:
            resp = self._client.completion(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            content = resp.choices[0].message.content or ""
            
            # Track cost if enabled
            if self.config.enable_cost_tracking:
                self._track_completion(model, resp, start_time)
            
            return content
            
        except Exception as e:
            if self.config.enable_cost_tracking:
                self._track_error(model, e, start_time)
            raise

    def _chat_with_fallback(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Try primary model first, then fallbacks if needed."""
        models = [self.config.model] + (self.config.fallback_models or [])
        
        for i, model in enumerate(models):
            try:
                return self._chat_single(model, messages, **kwargs)
            except Exception as e:
                logger.warning(f"Model {model} failed: {str(e)}")
                
                if i == len(models) - 1:  # Last model
                    logger.error("All models failed")
                    raise
                else:
                    logger.info(f"Trying fallback model {models[i+1]}...")
                    continue

    def _track_completion(self, model: str, response, start_time: float):
        """Track successful completion metrics."""
        duration = time.time() - start_time
        
        # Extract usage metrics
        usage = getattr(response, 'usage', None)
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
        total_tokens = getattr(usage, 'total_tokens', 0) if usage else 0
        
        # Calculate cost using LiteLLM's completion_cost function
        cost = 0.0
        try:
            cost = self._client.completion_cost(completion_response=response)
        except Exception as e:
            logger.debug(f"Could not calculate cost: {e}")
        
        # Update tracking counters
        self._total_calls += 1
        self._total_cost += cost
        self._total_tokens += total_tokens
        
        # Log at debug level to avoid cluttering
        logger.debug(
            f"LLM Call - Model: {model}, Duration: {duration:.3f}s, "
            f"Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}), "
            f"Cost: ${cost:.6f}, Total Cost: ${self._total_cost:.6f}"
        )

    def _track_error(self, model: str, error: Exception, start_time: float):
        """Track failed completion metrics."""
        duration = time.time() - start_time
        self._total_calls += 1
        
        logger.error(
            f"LLM Call Failed - Model: {model}, Duration: {duration:.3f}s, "
            f"Error: {str(error)}, Total Calls: {self._total_calls}"
        )

    def get_cost_stats(self) -> Dict[str, float]:
        """Get current cost tracking statistics."""
        return {
            "total_calls": self._total_calls,
            "total_cost": self._total_cost,
            "total_tokens": self._total_tokens,
            "average_cost_per_call": self._total_cost / max(self._total_calls, 1),
            "average_tokens_per_call": self._total_tokens / max(self._total_calls, 1),
        }

    def reset_cost_tracking(self):
        """Reset cost tracking counters."""
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_tokens = 0


# Convenience function for backward compatibility
def create_llm(model: Optional[str] = None) -> BaseLLM:
    """Factory function to create LLM instances."""
    return LiteLLMChatLLM(model=model)