from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Dict, Tuple

from agents.reasoner.implicit.reasoner import ImplicitState


class Act(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> Tuple[str, Dict[str, Any], Any]:
        """Return (tool_id, params, observation)."""
        ...