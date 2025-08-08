from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Optional

from agents.reasoner.implicit.reasoner import ImplicitState


class StopCondition(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> Optional[str]:
        ...


class SimpleStopCondition(StopCondition):
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> Optional[str]:
        if not state.turns:
            return None
        last = state.turns[-1]
        if last.thought and last.thought.strip().upper().startswith("FINAL:"):
            return last.thought.split(":", 1)[1].strip() or None
        return None


