from __future__ import annotations

from abc import ABC, abstractmethod

from agents.reasoner.sequential.reasoner import ReasonerState
from agents.llm.base_llm import BaseLLM


class SummarizeResult(ABC):
    """Abstract base class for summarizing reasoning results into final answers."""

    def __init__(self, *, llm: BaseLLM):
        self.llm = llm

    @abstractmethod
    def __call__(self, state: ReasonerState) -> str:
        """Generate a final answer from the reasoning state.

        Args:
            state: The current reasoning state containing goal, history, and steps.

        Returns:
            A string containing the final answer for the user.
        """
        ...
