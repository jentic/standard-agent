"""Lightweight LLM wrapper interfaces used by reasoners."""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from .config import get_config_value

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Minimal synchronous chat‑LLM interface.

    • Accepts a list[dict] *messages* like the OpenAI Chat format.
    • Returns *content* (str) of the assistant reply.
    • Implementations SHOULD be stateless; auth + model name given at init.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...


class LiteLLMChatLLM(BaseLLM):
    """Enhanced wrapper around litellm.completion with cost tracking and fallback support."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_cost_tracking: bool = True,
    ) -> None:
        import litellm
        
        # Load configuration from config.toml
        if model is None:
            model = get_config_value("llm", "model", default="gpt-4o")
        if temperature is None:
            temperature = get_config_value("llm", "temperature", default=0.2)
        if max_tokens is None:
            max_tokens = get_config_value("llm", "max_tokens", default=None)
        
        
        # Configuration validation
        if not model or model.strip() == "":
            logger.error("No LLM model configured! Please set [tool.actbots.llm.model] in config.toml")
            logger.info("Example models: 'gpt-4o', 'claude-3-opus-20240229', 'gemini/gemini-2.0-flash'")
            raise ValueError("No LLM model configured. Set model in config.toml")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_cost_tracking = enable_cost_tracking
        self._client = litellm
        
        # Set timeout from config
        timeout = get_config_value("llm", "timeout", default=60)
        litellm.request_timeout = timeout
        
        # Cost tracking state
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_tokens = 0
        
        logger.info(f"Initialized LLM with model: {self.model}")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send chat messages to the LLM and return the response."""
        return self._chat_single(self.model, messages, **kwargs)

    def _chat_single(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Execute a single chat call."""
        start_time = time.time()
        
        try:
            resp = self._client.completion(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            
            content = resp.choices[0].message.content or ""
            
            # Track cost if enabled
            if self.enable_cost_tracking:
                self._track_completion(model, resp, start_time)
            
            return content
            
        except Exception as e:
            if self.enable_cost_tracking:
                self._track_error(model, e, start_time)
            raise


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