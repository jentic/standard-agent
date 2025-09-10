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
        try:
            content= resp.choices[0].message.content.strip()
            if not content: 
                logger.error("empty_llm_response",msg = "LLM returned an empty response")
                raise ValueError("LLM returned an empty response")
            return content
        except (IndexError, AttributeError) as e:
            logger.error("malformed_llm_response",msg="LLM response was malformed.",error=str(e))
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
        raw_response = ""

        for attempt in range(max_retries + 1):
            try:
                raw_response = self.prompt(current_prompt, **kwargs)
                
                return super().prompt_to_json(current_prompt, **kwargs)

            except (json.JSONDecodeError, ValueError) as e:
                if attempt >= max_retries:
                    logger.error("json_decode_failed", attempt=attempt, error=str(e), msg="Exceeded max retries for JSON parsing")
                    raise json.JSONDecodeError("Failed to get valid JSON after multiple retries.", raw_response, 0) from e
                
                try:
                    bad_json_content = self.prompt(current_prompt, **kwargs)
                except (json.JSONDecodeError, ValueError) as inner_e:
                    bad_json_content = f"An error occurred: {str(inner_e)}"

                current_prompt = JSON_CORRECTION_PROMPT.format(
                    original_prompt=original_prompt,
                    bad_json=bad_json_content
                )

                logger.warning("json_decode_failed", attempt=attempt, error=str(e))

        # This should never be reached, but mypy requires it
        raise json.JSONDecodeError("Unexpected end of function", "", 0)
