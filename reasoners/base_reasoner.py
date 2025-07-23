from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

from reasoners.models import ReasoningResult, RuntimeContext
from typing import Any


class BaseReasoner(ABC):
    """Abstract contract for a reasoning loop implementation."""

    _ctx: RuntimeContext | None = None

    # -------------- runtime wiring -----------------------------------------
    def set_context(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._pass_context_to_components(ctx)

    def _pass_context_to_components(self, ctx: RuntimeContext) -> None:
        """Hook for subclasses to broadcast the context to their components."""
        pass  # default: nothing to do

    # -------------- convenient shorthand props -----------------------------
    @property
    def llm(self):    return self._ctx.llm

    @property
    def tools(self):  return self._ctx.tools

    @property
    def memory(self): return self._ctx.memory

    @abstractmethod
    def run(self, goal: str, *, meta: Dict[str, Any] | None = None) -> ReasoningResult:
        """The main entry point to execute the reasoning loop."""
        raise NotImplementedError
