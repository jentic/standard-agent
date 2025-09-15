from agents.llm.base_llm import BaseLLM, JSON_CORRECTION_PROMPT
from typing import List, Dict, Any
import json
import litellm

from utils.logger import get_logger
from utils.observe import observe
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

        # Extract usage/token counts across providers
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        try:
            usage = getattr(resp, "usage", None)
            if usage is None and isinstance(resp, dict):
                usage = resp.get("usage")
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                if prompt_tokens is None:
                    prompt_tokens = getattr(usage, "input_tokens", None)
                if completion_tokens is None:
                    completion_tokens = getattr(usage, "output_tokens", None)
            if isinstance(usage, dict):
                if prompt_tokens is None:
                    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                if completion_tokens is None:
                    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                if total_tokens is None:
                    total_tokens = usage.get("total_tokens")
        except Exception:
            pass

        try:
            text = resp.choices[0].message.content.strip()
        except (IndexError, AttributeError):
            text = ""

        # Compute total if missing
        if not isinstance(total_tokens, int):
            total = 0
            if isinstance(prompt_tokens, int):
                total += prompt_tokens
            if isinstance(completion_tokens, int):
                total += completion_tokens
            total_tokens = total if total > 0 else None

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
