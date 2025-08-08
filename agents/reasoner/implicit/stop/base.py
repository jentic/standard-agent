from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Optional

from agents.reasoner.implicit.reasoner import ImplicitState


class StopCondition(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> Optional[str]:
        ...
