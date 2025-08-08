from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from agents.llm.base_llm import BaseLLM
if TYPE_CHECKING:
    from agents.reasoner.implicit.reasoner import ImplicitState


class Summarizer(ABC):

    def __init__(self, *, llm: BaseLLM) -> None:
        self.llm = llm

    @abstractmethod
    def __call__(self, state: "ImplicitState") -> str:
        ...
