from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Deque, Dict

from memory.base_memory import BaseMemory
from tools.interface import ToolInterface
from utils.llm import BaseLLM
from reasoners.models import ReasonerState, Step


class BaseComponent(ABC):
    """Super-small DI hook shared by all components."""

    llm: BaseLLM | None = None
    tools: ToolInterface | None = None
    memory: BaseMemory | None = None

    def attach_services(
        self, *, llm: BaseLLM, tools: ToolInterface, memory: BaseMemory
    ) -> None:
        self.llm, self.tools, self.memory = llm, tools, memory


class Planner(BaseComponent):
    @abstractmethod
    def plan(self, goal: str) -> Deque[Step]:
        ...


class StepExecutor(BaseComponent):
    @abstractmethod
    def execute(self, step: Step, state: ReasonerState) -> Dict[str, Any] | None:
        ...


class Reflector(BaseComponent):
    @abstractmethod
    def handle(self, error: Exception, step: Step, state: ReasonerState) -> None:
        ...


class AnswerBuilder(BaseComponent):
    @abstractmethod
    def build(self, state: ReasonerState) -> str:
        ...
