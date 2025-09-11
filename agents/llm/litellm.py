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
            content = resp.choices[0].message.content.strip()
            if not content:
                logger.error("empty_llm_response", msg="LLM returned an empty response")
                raise ValueError("LLM returned an empty response")
            return content
        except (IndexError, AttributeError) as e:
            logger.error("malformed_llm_response", msg="LLM response was malformed", error=str(e))
            raise ValueError("LLM returned malformed response") from e

    def prompt_to_json(self, content: str, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Enhanced JSON prompting with automatic retry logic.

        Handles two types of failures differently:
        - ValueError (empty/malformed responses): Retry same prompt for transient issues
        - JSONDecodeError (bad JSON syntax): Retry with correction prompt using actual failed content

        Args:
            content: The prompt content
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Additional arguments passed to completion()

        Returns:
            Parsed JSON object as a dictionary

        Raises:
            json.JSONDecodeError: If all retry attempts fail
            ValueError: If LLM consistently returns empty/malformed responses
        """

        original_prompt = content
        current_prompt = content

        for attempt in range(max_retries + 1):
            try:
                return super().prompt_to_json(current_prompt, **kwargs)  # Single call!
            except json.JSONDecodeError as e:
                logger.warning("json_parse_failed", attempt=attempt, error=str(e))
                if attempt >= max_retries:
                    logger.error("json_decode_failed", attempt=attempt, error=str(e), msg="Exceeded max retries for JSON parsing")
                    raise

                if hasattr(e, 'raw_content'): #Use raw content if available, else fallback to generic
                    bad_json_content = e.raw_content  # Exact failed content
                else:
                    bad_json_content = "The previous response was not valid JSON"  # Fallback

                current_prompt = JSON_CORRECTION_PROMPT.format(
                    original_prompt=original_prompt,
                    bad_json=bad_json_content
                )

            except ValueError as e:
                # Handle empty/malformed responses - retry same prompt for transient issues
                logger.warning("empty_response_retry", attempt=attempt, error=str(e))
                if attempt >= max_retries:
                    raise

        # This should never be reached, but mypy requires it
        raise json.JSONDecodeError("Unexpected end of function", "", 0)
