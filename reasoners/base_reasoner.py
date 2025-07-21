from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

from memory.base_memory import BaseMemory
from tools.interface import ToolInterface
from utils.llm import BaseLLM
from reasoners.models import ReasoningResult
from typing import Any


class BaseReasoner(ABC):
    """Abstract contract for a reasoning loop implementation."""

    llm: BaseLLM
    tools: ToolInterface
    memory: BaseMemory

    def attach_services(self, *, llm: BaseLLM, tools: ToolInterface, memory: BaseMemory) -> None:
        """Explicitly attaches shared services to the reasoner instance."""
        self.llm = llm
        self.tools = tools
        self.memory = memory

    @abstractmethod
    def run(self, goal: str, *, meta: Dict[str, Any] | None = None) -> ReasoningResult:
        """The main entry point to execute the reasoning loop."""
        raise NotImplementedError
