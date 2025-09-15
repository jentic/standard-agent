from agents.llm.base_llm import BaseLLM, JSON_CORRECTION_PROMPT
from typing import List, Dict, Any
import json
import litellm

from utils.logger import get_logger
from utils.observability import observe
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

    @observe(llm=True)
    def completion(self, messages: List[Dict[str, str]], **kwargs) -> BaseLLM.LLMResponse:
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

        text = ""
        try:
            text = resp.choices[0].message.content.strip()
        except (IndexError, AttributeError):
            pass

        prompt_tokens, completion_tokens, total_tokens = self._extract_token_usage(resp)

        return BaseLLM.LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        )

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


    def _extract_token_usage(self, resp: Any) -> tuple[int | None, int | None, int | None]:
        """Extract token usage from provider response with fallbacks for different providers."""
        def _get_token(obj: Any, *keys: str) -> int | None:
            for key in keys:
                if hasattr(obj, key):
                    val = getattr(obj, key, None)

                elif isinstance(obj, dict):
                    val = obj.get(key)
                else:
                    continue
                if isinstance(val, int):
                    return val
            return None

        try:
            usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
            if usage is None:
                return None, None, None

            prompt_tokens = _get_token(usage, "prompt_tokens", "input_tokens")
            completion_tokens = _get_token(usage, "completion_tokens", "output_tokens")
            total_tokens = _get_token(usage, "total_tokens")

            # Compute total if missing but components available
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens

            return prompt_tokens, completion_tokens, total_tokens
        except Exception:
            return None, None, None
