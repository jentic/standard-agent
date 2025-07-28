from __future__ import annotations

from reasoners.models import ReasonerState
from reasoners.sequential.interface import AnswerBuilder

from utils.logger import get_logger
logger = get_logger(__name__)

FINAL_ANSWER_BUILDER_PROMPT: str = (
    """
    <role>
    You are the Final Answer Synthesizer for autonomous agents within the Jentic ecosystem. Your mission is to transform raw execution logs into clear, user-friendly responses that demonstrate successful goal achievement. You specialize in data interpretation, content formatting, and user communication.

    Your core responsibilities:
    - Analyze execution logs to extract meaningful results
    - Assess data sufficiency for reliable answers
    - Format responses using clear markdown presentation
    - Maintain professional, helpful tone in all communications
    </role>

    <goal>
    Generate a comprehensive final answer based on the execution log that directly addresses the user's original goal.
    </goal>

    <input>
    User's Goal: {goal}
    Execution Log: {history}
    </input>

    <instructions>
    1. Review the execution log to understand what actions were taken
    2. Assess if the collected data is sufficient to achieve the user's goal
    3. If insufficient data, respond with: "ERROR: insufficient data for a reliable answer."
    4. If sufficient, synthesize a comprehensive answer that:
       - Directly addresses the user's goal
       - Uses only information from the execution log
       - Presents content clearly with markdown formatting
       - Maintains helpful, professional tone
       - Avoids revealing internal technical details
    </instructions>

    <constraints>
    - Use only information from the execution log
    - Do not add external knowledge or assumptions
    - Do not reveal internal monologue or technical failures
    - Present results as if from a helpful expert assistant
    </constraints>

    <output_format>
    Clear, user-friendly response using markdown formatting (headings, lists, bold text as appropriate)
    </output_format>
    """
)

class FinalAnswerBuilder(AnswerBuilder):
    """
    Generates the final answer with the LLM; falls back to
    last-successful-result if the LLM fails.
    """

    def build(self, state: ReasonerState) -> str:
        logger.info("phase=SYNTHESIZE_START")
        if not self.llm:
            logger.warning("phase=SYNTHESIZE_FALLBACK reason='No LLM attached'")
            return self._heuristic(state)

        prompt = FINAL_ANSWER_BUILDER_PROMPT.format(
            goal=state.goal,
            history="\n".join(state.history),
        )
        logger.debug(f"phase=SYNTHESIZE_PROMPT prompt={prompt}")

        try:
            reply = (self.llm.chat([{"role": "user", "content": prompt}],).strip())
            logger.info(f"FINAL ANSWER: {reply}")
            logger.debug(f"phase=SYNTHESIZE_LLM_REPLY reply={reply}")

            if not reply:
                logger.error("phase=SYNTHESIZE_FAILED reason='LLM returned empty content'")
                raise ValueError("LLM returned empty content")

            state.is_complete = True
            logger.info("phase=SYNTHESIZE_SUCCESS")
            return reply

        except Exception as exc:
            logger.error("phase=SYNTHESIZE_FAILED error='%s'", exc, exc_info=True)
            return self._heuristic(state)

    @staticmethod
    def _heuristic(state: ReasonerState) -> str:
        """
        Fallback logic when LLM is unavailable:
        1. Return the `result` of the last successful Step, if any.
        2. Otherwise, dump the last history line.
        """
        logger.info("phase=HEURISTIC_FALLBACK_START")
        for step in reversed(state.completed_steps or []):
            if step.result:
                logger.info(
                    "phase=HEURISTIC_SUCCESS reason='Found last successful step'"
                )
                state.is_complete = True
                return str(step.result)

        if state.history:
            logger.info("phase=HEURISTIC_SUCCESS reason='Found last history item'")
            return state.history[-1]

        logger.warning("phase=HEURISTIC_FAILED reason='No answer found'")
        return "No definitive answer was produced."
