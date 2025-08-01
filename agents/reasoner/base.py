from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

from collections.abc import MutableMapping
from agents.tools.base import JustInTimeToolingBase
from agents.llm.base_llm import BaseLLM


@dataclass
class ReasoningResult:
    """Lightweight summary returned by a Reasoner run."""

    final_answer: str
    iterations: int
    tool_calls: List[dict[str, Any]]
    success: bool
    error_message: str | None = None


class BaseReasoner(ABC):
    """Abstract contract for a reasoning loop implementation."""

    def __init__(self, *, llm: BaseLLM, tools: JustInTimeToolingBase, memory: MutableMapping):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    @abstractmethod
    def run(self, goal: str) -> ReasoningResult:
        """The main entry point to execute the reasoning loop."""
        raise NotImplementedError
