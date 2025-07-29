from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping

from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase


class ExecuteStep(ABC):
    def __init__(self, *, llm: BaseLLM, tools: JustInTimeToolingBase, memory: MutableMapping):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    @abstractmethod
    def __call__(self, step, state):
        ...