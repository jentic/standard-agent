from __future__ import annotations

from typing import List
from textwrap import dedent

from typing import TYPE_CHECKING
from agents.reasoner.implicit.summarizer.base import Summarizer
if TYPE_CHECKING:
    from agents.reasoner.implicit.reasoner import ImplicitState

SUMMARY_PROMPT = dedent(
    """
    <role>
    You are the Final Answer Synthesizer for an agent. Your job is to produce a clear, correct, and helpful final answer using only the transcript.
    </role>

    <goal>
    Provide the final answer to the user's goal based solely on Thoughts, Actions, and Observations below.
    </goal>

    <transcript>
    {transcript}
    </transcript>

    <instructions>
    1. If the transcript contains a Thought starting with 'FINAL:', use its text after 'FINAL:' exactly as the answer (lightly polish grammar only).
    2. Otherwise, synthesize a concise, accurate answer strictly from the Observations and relevant Thoughts.
    3. Do not expose internal mechanics (e.g., tool errors, scoring rules) unless necessary to explain limitations.
    4. If evidence is clearly insufficient to answer, return exactly: "ERROR: insufficient data for a reliable answer."
    </instructions>

    <output_format>
    A short, user-facing answer in plain text. No markdown, no code fences.
    </output_format>
    """
).strip()


class DefaultImplicitSummarizer(Summarizer):
    def __call__(self, state: "ImplicitState") -> str:
        # If loop already set a final answer, just return it
        if state.final_answer:
            return state.final_answer

        prompt = SUMMARY_PROMPT.format(transcript=state.get_reasoning_transcript())
        reply = self.llm.prompt(prompt)
        return reply or "ERROR: insufficient data for a reliable answer."


