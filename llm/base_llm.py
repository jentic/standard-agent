"""Lightweight LLM wrapper interfaces used by reasoners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLM(ABC):
    """Minimal synchronous chat‑LLM interface.

    • Accepts a list[dict] *messages* like the OpenAI Chat format.
    • Returns *content* (str) of the assistant reply.
    • Implementations SHOULD be stateless; auth + model name given at init.
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
