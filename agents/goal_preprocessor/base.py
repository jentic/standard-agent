from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any, Tuple

from agents.llm.base_llm import BaseLLM


class BaseGoalPreprocessor(ABC):
    """
    Component that preprocess a raw user goal.
    """

    def __init__(self, *, llm: BaseLLM):
        self.llm = llm

    @abstractmethod
    def process(self, goal: str, history: Sequence[Dict[str, Any]]) -> Tuple[str, str | None]:
        """
        Preprocess a raw user goal.
        
        Args:
            goal: The raw goal from the user.
            history: A sequence of previous goal/result dictionaries.

        Returns:
            A tuple of (revised_goal, intervention_message).
            - If intervention_message is None, use the revised_goal.
            - If intervention_message is present, ask the user that question.
        """
        ...
