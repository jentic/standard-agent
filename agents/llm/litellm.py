from agents.llm.base_llm import BaseLLM, JSON_CORRECTION_PROMPT
from typing import List, Dict, Any
import json
import litellm

from utils.logger import get_logger
logger = get_logger(__name__)

class LiteLLM(BaseLLM):
    """Wrapper around litellm.completion."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens

    def completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Merge default parameters with provided kwargs
        effective_temperature = kwargs.get("temperature", self.temperature)
        effective_max_tokens = kwargs.get("max_tokens", self.max_tokens)

        completion_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if effective_temperature is not None:
            completion_kwargs["temperature"] = effective_temperature
        if effective_max_tokens is not None:
            completion_kwargs["max_tokens"] = effective_max_tokens

        # Add any additional kwargs (like response_format)
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens"]:
                completion_kwargs[key] = value

        resp = litellm.completion(**completion_kwargs)

        # Notify eval layer about per-call token usage if available
        # Expected shape: resp.usage.prompt_tokens / completion_tokens (provider-dependent)
        try:
            usage = getattr(resp, "usage", None)
            prompt_tokens = None
            completion_tokens = None
            if usage is not None:
                # Attribute style
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
            # Dict-style fallback if provider returns dict-like usage
            if prompt_tokens is None and isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            # Call optional usage callback if provided on this instance
            usage_cb = getattr(self, "usage_callback", None)
            if callable(usage_cb):
                usage_cb(prompt_tokens if isinstance(prompt_tokens, int) else None,
                         completion_tokens if isinstance(completion_tokens, int) else None)
        except Exception:
            # Never fail the main call due to usage accounting
            pass

        try:
            return resp.choices[0].message.content.strip()
        except (IndexError, AttributeError):
            return ""

    def prompt_to_json(self, content: str, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Enhanced JSON prompting with automatic retry logic.

        First attempts to use the base class implementation (JSON mode).
        If that fails, retries with correction prompts up to max_retries times.

        Args:
            content: The prompt content
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Additional arguments passed to completion()

        Returns:
            Parsed JSON object as a dictionary

        Raises:
            json.JSONDecodeError: If all retry attempts fail
        """

        original_prompt = content
        current_prompt = content

        for attempt in range(max_retries + 1):
            try:
                return super().prompt_to_json(current_prompt, **kwargs)

            except json.JSONDecodeError as e:
                # Update prompt for next iteration
                current_prompt = JSON_CORRECTION_PROMPT.format(
                    original_prompt=original_prompt,
                    bad_json="The previous response was not valid JSON"
                )

                logger.warning("json_decode_failed", attempt=attempt, error=str(e))

        # This should never be reached, but mypy requires it
        raise json.JSONDecodeError("Unexpected end of function", "", 0)
