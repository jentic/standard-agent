from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase
from agents.reasoner.implicit.reasoner import ImplicitState

class Act(ABC):
    """Select and execute a tool for the current ACTION step.

    Responsibilities
    - Interpret the latest ACTION text from `ImplicitState` and select the most
      appropriate tool (via search or rule) to fulfill the instruction and execute it.

    Contract
    - Called only when the latest `ReasonNode.kind == ACTION`
    - Must return a tuple: (tool_id: str, params: dict, observation: Any)
    - Should raise a domain error (e.g., ToolSelectionError) when no suitable
      tool exists. The reasoner catches and writes it into the transcript so
      the next THOUGHT can adapt.

    """
    def __init__(self, *, llm: BaseLLM, tools: JustInTimeToolingBase, top_k: int = 15) -> None:
        self.llm = llm
        self.tools = tools
        self.top_k = top_k

    @abstractmethod
    def __call__(self, state: ImplicitState) -> Tuple[str, Dict[str, Any], Any]:
        """Return (tool_id, params, observation)."""
        ...