from __future__ import annotations

import textwrap

from agents.reasoner.sequential.reasoner import ReasonerState
from agents.reasoner.sequential.summarizer.base import SummarizeResult

from utils.logger import get_logger, trace_method
logger = get_logger(__name__)

SUMMARIZE_RESULT_PROMPT = textwrap.dedent("""
    You are the Final Answer Synthesizer for an autonomous agent. Your sole responsibility is to generate a clear, concise, and user-friendly final answer based on the provided information.

    **User's Goal:**
    {goal}

    **Chronological Log of Actions and Available Data:**
    ```
    {history}
    ```

    **Your Task:**
    1.  **Analyze the Log:** Carefully review the log to understand what actions were taken and what data was collected.
    2.  **Assess Sufficiency:** Determine if the data in the log is sufficient to fully and accurately achieve the User's Goal.
        -   If NOT sufficient, you MUST reply with the single line: `ERROR: insufficient data for a reliable answer.`
    3.  **Synthesize the Final Answer:** If the data is sufficient, synthesize a comprehensive answer.
        -   Directly address the User's Goal.
        -   Use only the information from the log. Do NOT use outside knowledge.
        -   Present the answer clearly using Markdown for formatting (e.g., headings, lists, bold text).
        -   Do NOT reveal the internal monologue, failed steps, or raw data snippets. Your tone should be that of a helpful, expert assistant.

    **Final Answer:**
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
