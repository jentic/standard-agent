from __future__ import annotations

from typing import Sequence, Dict, Any, Tuple
from agents.goal_preprocessor.base import BaseGoalPreprocessor
from agents.prompts import load_prompts

from utils.logger import get_logger
logger = get_logger(__name__)

_PROMPTS = load_prompts("goal_preprocessors/conversational", required_prompts=["clarify_goal"])


class ConversationalGoalPreprocessor(BaseGoalPreprocessor):
    """
    Resolves ambiguous user goals by leveraging conversation history.

    This preprocessor analyzes goals that contain unclear references (like "do it again",
    "send it again", or "fix that") and attempts to resolve them using recent
    conversation context.

    Returns:
        Tuple[str, str | None]: (revised_goal, clarification_question)
        - revised_goal: The goal to execute (original or improved)
        - clarification_question: Question for user if goal is unclear, None otherwise
    """

    def process(self, goal: str, history: Sequence[Dict[str, Any]]) -> Tuple[str, str | None]:

        history_str = "\n".join(f"Goal: {item['goal']}\nResult: {item['result']}" for item in history)
        prompt = _PROMPTS["clarify_goal"].format(history_str=history_str, goal=goal)
        response = self.llm.prompt_to_json(prompt)

        if response.get("revised_goal"):
            logger.info("revised_goal", original_goal=goal, revised_goal=response["revised_goal"])
            return response["revised_goal"], None
        
        if response.get("clarification_question"):
            logger.warning('clarification_question', clarification_question=response["clarification_question"])
            return goal, response["clarification_question"]

        return goal, None
