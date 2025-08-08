from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping


class DecidePolicy(ABC):
    @abstractmethod
    def __call__(self, state: "ImplicitState", memory: MutableMapping) -> str:
        """Return one of: "REASON" | "TOOL" | "HALT"."""
        ...