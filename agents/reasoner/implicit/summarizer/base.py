from __future__ import annotations

from abc import ABC, abstractmethod
from agents.llm.base_llm import BaseLLM


class Summarizer(ABC):

    def __init__(self, *, llm: BaseLLM) -> None:
        self.llm = llm

    @abstractmethod
    def __call__(self, state: "ImplicitState") -> str:
        ...
