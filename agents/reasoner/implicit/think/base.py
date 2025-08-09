from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from agents.llm.base_llm import BaseLLM

class Think(ABC):

    def __init__(self, *, llm: BaseLLM) -> None:
        self.llm = llm

    @abstractmethod
    def __call__(self, state: "ImplicitState", memory: MutableMapping):
        ...
