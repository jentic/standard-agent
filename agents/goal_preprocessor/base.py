from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseGoalPreprocessor(ABC):
    """
    Takes a user goal and conversation history, returning a revised goal if the goal is ambiguous.

        Args:
            goal: The raw goal from the user.
            history: A sequence of previous goal/result dictionaries.

        Returns:
            A tuple of (revised_goal, intervention_message).
            - If intervention_message is None, use the revised_goal.
            - If intervention_message is present, surface the intervention_message to the user.
    """

    @abstractmethod
    def process(self, goal: str, history: Sequence[Dict[str, Any]]) -> Tuple[str, str | None]: ...
