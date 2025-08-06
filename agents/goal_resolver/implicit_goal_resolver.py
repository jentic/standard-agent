from __future__ import annotations

from typing import Sequence, Dict, Any, Tuple
from textwrap import dedent
from agents.llm.base_llm import BaseLLM
from agents.goal_resolver.base import BaseGoalResolver

from utils.logger import get_logger
logger = get_logger(__name__)

IMPLICIT_GOAL_RESOLVER_PROMPT = dedent("""
    <role>
    You are a Goal Disambiguator working within the Agent ecosystem.
    Your responsibility is to analyze a user’s new goal in the context of the recent conversation history and determine whether the goal is ambiguous or underspecified. 
    If the goal is ambiguous, your job is to resolve it using prior context — or, if resolution is not possible, ask the user a precise follow-up question.
    </role>
    
    <instructions>
    Follow these steps carefully:
    
    1. Analyze the conversation history, prioritizing the most recent goals and results.
    2. Identify ambiguity in the new goal — especially vague references, missing objects, or unclear intent (e.g., pronouns like "it", phrases like "do it again", or unclear verbs like "fix this").
    3. If the goal is ambiguous:
       - Determine whether the ambiguity can be fully resolved **only using the conversation history**.
       - If it *can* be resolved, rewrite the goal to be explicit, complete, and self-contained.
       - If it *cannot* be resolved, generate **a single clear clarification question** for the user.
    4. If the goal is **not** ambiguous, return it unchanged.
    
    Common ambiguity signals include:
    - Pronouns without clear referents (e.g., “it”, “that”, “this”)
    - Phrases referring to prior actions vaguely (e.g., “do that again”, “fix it”, “summarize this”)
    - Missing targets (e.g., “send report” without saying which report)
    - Context-dependent tasks (e.g., “follow up on the last thing”)
    
    You must weigh recent conversation turns more heavily when resolving references.
    </instructions>
    
    <input>
    Conversation History:
    {history_str}
    
    New Goal: "{goal}"
    </input>
    
    <output_format>
    Respond with a valid JSON object in the following format:
    
    {{
      "is_ambiguous": boolean,                // True if the goal is ambiguous
      "can_be_resolved": boolean,            // True only if ambiguity can be resolved using history
      "revised_goal": string,                // If resolvable, return rewritten explicit goal; else empty string
      "clarification_question": string       // If not resolvable, return a user-facing clarification question; else empty string
    }}
    
    Validation Rules:
    - If "is_ambiguous" is False, "can_be_resolved" must be False, and both "revised_goal" and "clarification_question" should be empty strings.
    - If "is_ambiguous" is True and "can_be_resolved" is True, return a meaningful "revised_goal" and an empty "clarification_question".
    - If "is_ambiguous" is True and "can_be_resolved" is False, return a meaningful "clarification_question" and an empty "revised_goal".
    </output_format>
""")


class ImplicitGoalResolver(BaseGoalResolver):
    """
    LLM-based processor that tries to resolve ambiguous references in a goal
    using recent conversation history. If it cannot, it raises
    ClarificationNeededError with a follow-up question.
    """
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def process(self, goal: str, history: Sequence[Dict[str, Any]]) -> Tuple[str, str | None]:
        if not history:
            return goal, None

        prompt = self._build_prompt(goal, history)
        response = self.llm.prompt_to_json(prompt)

        if response.get("is_ambiguous", False):
            if response.get("can_be_resolved", False) and response.get("revised_goal"):
                logger.info("revised_goal", original_goal=goal, revised_goal=response["revised_goal"])
                return response["revised_goal"], None
            else:
                logger.warning('clarification_question', clarification_question=response["clarification_question"])
                return goal, response.get("clarification_question", "Could you clarify your request?")

        return goal, None

    @staticmethod
    def _build_prompt(goal: str, history: Sequence[Dict[str, Any]]) -> str:
        history_str = "\n".join(
            f"Goal: {item['goal']}\nResult: {item['result']}" for item in history
        )
        return IMPLICIT_GOAL_RESOLVER_PROMPT.format(
            history_str=history_str,
            goal=goal,
        )
