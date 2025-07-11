"""Lightweight LLM wrapper interfaces used by reasoners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """Minimal synchronous chat‑LLM interface.

    • Accepts a list[dict] *messages* like the OpenAI Chat format.
    • Returns *content* (str) of the assistant reply.
    • Implementations SHOULD be stateless; auth + model name given at init.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...


class LiteLLMChatLLM(BaseLLM):
    """Wrapper around litellm.completion."""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        import litellm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = litellm

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = self._client.completion(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        content = resp.choices[0].message.content
        return content or ""  # Avoid None propagating 