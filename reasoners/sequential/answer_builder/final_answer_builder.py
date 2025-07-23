from __future__ import annotations

from reasoners.models import ReasonerState
from reasoners.sequential.interface import AnswerBuilder
from reasoners.prompts import FINAL_ANSWER_SYNTHESIS_PROMPT

from utils.logger import get_logger
logger = get_logger(__name__)

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

        prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.format(
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
