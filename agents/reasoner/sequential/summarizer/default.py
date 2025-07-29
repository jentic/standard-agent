from __future__ import annotations

import textwrap

from agents.reasoner.sequential.reasoner import ReasonerState
from agents.reasoner.sequential.summarizer.base import SummarizeResult

from utils.logger import get_logger, trace_method
logger = get_logger(__name__)

SUMMARIZE_RESULT_PROMPT = textwrap.dedent("""
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
""").strip()

class DefaultSummarizeResult(SummarizeResult):
    """
    Generates the final answer with the LLM; falls back to
    last-successful-result if the LLM fails.
    """

    @trace_method
    def __call__(self, state: ReasonerState) -> str:
        prompt = SUMMARIZE_RESULT_PROMPT.format(
            goal=state.goal,
            history="\n".join(state.history),
        )

        try:
            reply = self.llm.prompt(prompt)
            logger.info("summary_generated", goal=state.goal, result_length=len(reply) if reply else 0)

            if not reply:
                logger.error("llm_empty_response", goal=state.goal)
                raise ValueError("LLM returned empty content")

            state.is_complete = True
            return reply

        except Exception as exc:
            logger.error("summarization_failed", goal=state.goal, error=str(exc), exc_info=True)
            if state.history:
                logger.info("using_fallback_answer", goal=state.goal, fallback_source="last_history_item")
                state.is_complete = True
                return state.history[-1]
            else:
                logger.warning("no_fallback_available", goal=state.goal)
                return "No definitive answer was produced."
