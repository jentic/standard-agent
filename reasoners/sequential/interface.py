from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Deque, Dict

from llm.base_llm import BaseLLM
from tools.interface import ToolInterface
from memory.base_memory import BaseMemory
from reasoners.models import ReasonerState, Step


class BaseComponent(ABC):
    def __init__(
            self,
            *,
            llm: BaseLLM | None = None,
            tools: ToolInterface | None = None,
            memory: BaseMemory | None = None,
    ) -> None:
        self._llm: BaseLLM | None = None
        self._tools: ToolInterface | None = None
        self._memory: BaseMemory | None = None

        if llm or tools or memory:
            self.set_services(llm=llm, tools=tools, memory=memory)

    # --------------------------- wiring ------------------------------ #
    def set_services(
            self,
            *,
            llm: BaseLLM | None = None,
            tools: ToolInterface | None = None,
            memory: BaseMemory | None = None,
    ) -> None:
        if llm is not None:
            self._llm = llm
        if tools is not None:
            self._tools = tools
        if memory is not None:
            self._memory = memory

    # ------------------------ convenience --------------------------- #
    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            raise RuntimeError(f"{self.__class__.__name__} has no LLM wired in")
        return self._llm

    @property
    def tools(self) -> ToolInterface:
        if self._tools is None:
            raise RuntimeError(f"{self.__class__.__name__} has no Tools interface wired in")
        return self._tools

    @property
    def memory(self) -> BaseMemory:
        if self._memory is None:
            raise RuntimeError(f"{self.__class__.__name__} has no Memory wired in")
        return self._memory


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
