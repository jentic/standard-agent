from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping

from agents.reasoner.sequential.reasoner import ReasonerState, Step
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase


class Reflect(ABC):
    def __init__(self, *, llm: BaseLLM, tools: JustInTimeToolingBase, memory: MutableMapping):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    @abstractmethod
    def __call__(self, error: Exception, step: Step, state: ReasonerState) -> None:
        ...
