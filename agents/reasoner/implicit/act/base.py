from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Dict, Tuple

from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase

class Act(ABC):
    def __init__(self, *, llm: BaseLLM, tools: JustInTimeToolingBase, top_k: int = 15) -> None:
        self.llm = llm
        self.tools = tools
        self.top_k = top_k

    @abstractmethod
    def __call__(self, state: "ImplicitState", memory: MutableMapping) -> Tuple[str, Dict[str, Any], Any]:
        """Return (tool_id, params, observation)."""
        ...