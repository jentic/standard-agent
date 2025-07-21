from __future__ import annotations

from jentic_agents.reasoners.models import ReasonerState
from jentic_agents.reasoners.sequential.interface import Synthesizer
import logging
log = logging.getLogger("FinalAnswerSynthesizer")

FINAL_ANSWER_SYNTHESIS_PROMPT: str = (
    """
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
    """
)

class FinalAnswerSynthesizer(Synthesizer):
    """
    Generates the final answer with the LLM; falls back to
    last-successful-result if the LLM fails.
    """

    def synthesize(self, state: ReasonerState) -> str:
        if not self.llm:
            log.warning("No LLM attached – returning heuristic answer")
            return self._heuristic(state)

        prompt = FINAL_ANSWER_SYNTHESIS_PROMPT.format(
            goal=state.goal,
            history="\n".join(state.history),
        )

        try:
            reply = self.llm.chat([{"role": "user", "content": prompt}],).strip()

            if not reply:
                raise ValueError("LLM returned empty content")

            state.is_complete = True
            return reply

        except Exception as exc:
            log.error("LLM synthesis failed: %s – using heuristic", exc)
            return self._heuristic(state)

    @staticmethod
    def _heuristic(state: ReasonerState) -> str:
        """
        Fallback logic when LLM is unavailable:
        1. Return the `result` of the last successful Step, if any.
        2. Otherwise, dump the last history line.
        """
        for step in reversed(state.completed_steps or []):
            if step.result:
                state.is_complete = True
                return str(step.result)

        if state.history:
            return state.history[-1]

        return "No definitive answer was produced."
