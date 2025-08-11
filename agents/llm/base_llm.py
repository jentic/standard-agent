"""Lightweight LLM wrapper interfaces used by reasoner implementations."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
import os
from textwrap import dedent
from typing import List, Dict, Any

from utils.logger import get_logger
logger = get_logger(__name__)


# JSON correction prompt for retry attempts
JSON_CORRECTION_PROMPT = dedent("""
    <role>
    You are a meticulous JSON‑syntax corrector. Your sole mission is to turn an invalid JSON string into a valid one **without altering any data values or keys**.
    </role>
    
    <input>
    original_prompt: {original_prompt}
    bad_json: {bad_json}
    </input>
    
    <output_format>
     **A single, raw, valid JSON object that contains the same data as bad_json but with valid syntax**
    </output_format>
    
    STRICT RULES:
    1. Respond with exactly one JSON object that can be successfully parsed by the json.loads() in Python — no markdown, comments, or code fences.
    2. Preserve every key–value pair from the faulty input; fix syntax only.
    3. Do **not** add explanations or extra fields.
    
    <self_check>
    After drafting your answer:
    - Parse it internally to ensure it is valid JSON.
    - Verify all data values match the original.
    If any check fails, silently regenerate until both checks pass.
    </self_check>
""").strip()


class BaseLLM(ABC):
    """Minimal synchronous chat‑LLM interface.

    • Accepts a list[dict] *messages* like the OpenAI Chat format.
    • Returns *content* (str) of the assistant reply.
    • Implementations SHOULD be stateless; auth + model name given at init.

    Raises
    ------
    ValueError : If neither `model` is passed nor `LLM_MODEL` is set in .env during initialization.
    """

    # Shared regex pattern for extracting JSON from markdown code fences
    _fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")

    def __init__(self, model: str | None = None, *, temperature: float | None = None) -> None:
        resolved_model = model or os.getenv("LLM_MODEL")
        if not resolved_model:
            logger.error( "llm_model_missing", msg="No LLM model configured in .env")
            raise ValueError( "Missing LLM model. Provide model='your-model' when constructing the LLM or set LLM_MODEL in the environment.")

        self.model: str = resolved_model
        self.temperature: float = self._load_env_temperature() if temperature is None else temperature

    @staticmethod
    def _load_env_temperature() -> float | None:
        env_temp = os.getenv("LLM_TEMPERATURE")
        if env_temp is None:
            return None
        try:
            return float(env_temp)
        except ValueError:
            logger.warning("invalid_env_temperature", value=env_temp)
            return None

    @abstractmethod
    def completion(self, messages: List[Dict[str, str]], **kwargs) -> str: ...

    def prompt(self, content: str, **kwargs) -> str:
        """Convenience method for single user prompts."""
        return self.completion([{"role": "user", "content": content}], **kwargs)

    def prompt_to_json(self, content: str, **kwargs) -> Dict[str, Any]:
        """
        Prompt the LLM and ensure the response is valid JSON.

        Basic implementation uses JSON mode if supported by the underlying LLM.
        Subclasses can override this method to add retry logic or other enhancements.

        Args:
            content: The prompt content
            max_retries: Maximum number of retry attempts (ignored in base implementation)
            **kwargs: Additional arguments passed to completion()

        Returns:
            Parsed JSON object as a dictionary

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        # Use JSON mode if supported by the LLM
        kwargs_with_json = kwargs.copy()
        kwargs_with_json.setdefault("response_format", {"type": "json_object"})
        raw_response = self.prompt(content, **kwargs_with_json)
        cleaned_response = self._fence_pattern.sub(lambda m: m.group(1).strip(), raw_response)
        return json.loads(cleaned_response)
