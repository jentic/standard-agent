from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from agents.reasoner.implicit.reasoner import ImplicitState


class DecidePolicy(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> str:
        """Return one of: "REASON" | "TOOL" | "HALT"."""
        ...


class SimpleDecidePolicy(DecidePolicy):
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> str:
        if not state.turns:
            return "REASON"
        last = state.turns[-1]
        return "REASON" if last.observation is not None else "TOOL"


