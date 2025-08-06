from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseGoalResolver(ABC):
    """
    Takes a user goal and conversation history, returning a revised goal if the goal is ambiguous.

        Args:
            goal: The raw goal from the user.
            history: A sequence of previous goal/result dictionaries.

        Returns:
            A tuple of (revised_goal, clarification_question).
            - If clarification_question is None, use the revised_goal.
            - If clarification_question is present, ask the user that question.
    """

    @abstractmethod
    def process(self, goal: str, history: Sequence[Dict[str, Any]]) -> Tuple[str, str | None]: ...
