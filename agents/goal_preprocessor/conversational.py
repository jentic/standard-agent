from __future__ import annotations

from typing import Sequence, Dict, Any, Tuple
from textwrap import dedent
from agents.llm.base_llm import BaseLLM
from agents.goal_preprocessor.base import BaseGoalPreprocessor

from utils.logger import get_logger
logger = get_logger(__name__)

CONVERSATIONAL_GOAL_RESOLVER_PROMPT = dedent("""
    <role>
    You are a Goal Disambiguator working within the Agent ecosystem.
    Your responsibility is to analyze a user's new goal in the context of the recent conversation history and determine whether the goal contains critical ambiguities that prevent execution. 
    If the goal has unresolvable ambiguities, your job is to resolve them using prior context — or, if resolution is not possible, ask the user a precise follow-up question.
    </role>
    
    <instructions>
    Follow these steps carefully:
    
    1. Analyze the conversation history, prioritizing the most recent goals and results.
    2. Identify **critical ambiguity** in the new goal — references that make the goal impossible to execute (e.g., pronouns like "it" without clear referents, phrases like "do it again" without knowing what "it" is, or unclear targets like "send it to him" without knowing what or who).
    3. **Important**: Goals that are actionable but could be more specific are NOT ambiguous. An agent can make reasonable assumptions and proceed with execution.
    4. If the goal contains critical ambiguity:
       - Determine whether the ambiguity can be fully resolved **only using the conversation history**.
       - If it *can* be resolved, rewrite the goal to be explicit, complete, and self-contained.
       - If it *cannot* be resolved, generate **a single clear clarification question** for the user.
    5. If the goal is actionable (even if it could be more detailed), return it unchanged.
    
    Critical ambiguity signals (these make execution impossible):
    - Pronouns without clear referents that prevent action (e.g., "send it" - what is "it"?)
    - References to prior actions without context (e.g., "do that again" - what is "that"?)
    - Missing essential targets (e.g., "call him back" - who is "him"?)
    - Context-dependent tasks with no context (e.g., "fix the issue" with no prior issue mentioned)
    
    NOT critical ambiguity (these are actionable):
    - General requests that can proceed with reasonable assumptions
    - Requests missing preferences but not essential info (e.g., "book a flight to Paris" - agent can ask for dates during execution)
    - Broad requests where agent can provide comprehensive results (e.g., "show me news about Tesla" - agent can show recent general news)
    
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
      "is_ambiguous": boolean,                // True if the goal contains critical ambiguity that prevents execution
      "can_be_resolved": boolean,            // True only if critical ambiguity can be resolved using history
      "revised_goal": string,                // If resolvable, return rewritten explicit goal; else empty string
      "clarification_question": string       // If not resolvable, return a user-facing clarification question; else empty string
    }}
    
    Validation Rules:
    - If "is_ambiguous" is False, "can_be_resolved" must be False, and both "revised_goal" and "clarification_question" should be empty strings.
    - If "is_ambiguous" is True and "can_be_resolved" is True, return a meaningful "revised_goal" and an empty "clarification_question".
    - If "is_ambiguous" is True and "can_be_resolved" is False, return a meaningful "clarification_question" and an empty "revised_goal".
    </output_format>
""")


class ConversationalGoalPreprocessor(BaseGoalPreprocessor):
    """
    LLM-based processor that tries to resolve ambiguous references in a goal
    using recent conversation history. If it cannot, it raises
    ClarificationNeededError with a follow-up question.
    """
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def process(self, goal: str, history: Sequence[Dict[str, Any]]) -> Tuple[str, str | None]:

        history_str = "\n".join(f"Goal: {item['goal']}\nResult: {item['result']}" for item in history)
        prompt = CONVERSATIONAL_GOAL_RESOLVER_PROMPT.format(history_str=history_str, goal=goal)
        response = self.llm.prompt_to_json(prompt)

        if response.get("is_ambiguous", False):
            if response.get("can_be_resolved", False) and response.get("revised_goal"):
                logger.info("revised_goal", original_goal=goal, revised_goal=response["revised_goal"])
                return response["revised_goal"], None
            else:
                logger.warning('clarification_question', clarification_question=response["clarification_question"])
                return goal, response.get("clarification_question")

        return goal, None
