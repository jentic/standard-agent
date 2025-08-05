from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class ClarificationNeededError(Exception):
    """
    Raised by a GoalProcessor when the user's request is too ambiguous to
    continue automatically.
    """

    def __init__(self, question: str):
        self.question = question
        super().__init__(f"Clarification needed from user: {question}")

        logger.warning(f"Clarification needed from user: {question}")


class BaseGoalResolver(ABC):
    """
    Takes a user goal and conversation history, returning a revised goal if the goal is ambiguous.

        Args:
            goal: The raw goal from the user.
            history: A sequence of previous goal/result dictionaries.

        Returns:
            A revised, self-contained goal string ready for the reasoner.

        Raises:
            ClarificationNeededError: If ambiguity cannot be resolved.
    """

    @abstractmethod
    def process(self, goal: str, history: Sequence[Dict[str, Any]]) -> str: ...
