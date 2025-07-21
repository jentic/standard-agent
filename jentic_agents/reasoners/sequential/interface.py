from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Deque, Dict

from jentic_agents.memory.base_memory import BaseMemory
from jentic_agents.tools.interface import ToolInterface
from jentic_agents.utils.llm import BaseLLM
from jentic_agents.reasoners.models import ReasonerState, Step


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
    def handle(self, error: Exception, step: Step, state: ReasonerState, failed_tool_id: str) -> None:
        ...


class Synthesizer(BaseComponent):
    @abstractmethod
    def synthesize(self, state: ReasonerState) -> str:
        ...
