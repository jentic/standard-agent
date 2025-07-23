from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Deque, Dict

from reasoners.models import RuntimeContext
from reasoners.models import ReasonerState, Step


class BaseComponent(ABC):
    """All components keep a reference to the shared RuntimeContext."""
    _ctx: RuntimeContext | None = None

    def set_context(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx

    # convenience shorthands
    @property
    def llm(self):    return self._ctx.llm
    @property
    def tools(self):  return self._ctx.tools
    @property
    def memory(self): return self._ctx.memory


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
