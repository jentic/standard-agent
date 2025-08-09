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
    You are the Final Answer Synthesizer for an agent. Produce a clear, correct, and helpful final answer using only the transcript.
    </role>

    <goal>
    Provide the final answer to the user's goal based solely on the transcript of reasoning and tool use.
    </goal>

    <transcript>
    {transcript}
    </transcript>

    <instructions>
    1. The transcript consists of lines like:
       - THOUGHT: <text>
       - ACTION: <text>              (an intended action produced by reasoning)
       - ACTION_EXECUTED: tool_id=…  (the tool that actually ran)
       - OBSERVATION: <text/json>    (results returned by tools)
       - FINAL: <text>               (a ready final answer)
    2. If a FINAL line exists, use its text (lightly polish grammar only) as the final answer.
    3. Otherwise, synthesize a concise, accurate answer strictly from OBSERVATION lines, using THOUGHT/ACTION for context when needed.
    4. Ignore internal mechanics (IDs, prompts, scoring). Do not mention implementation details.
    5. If evidence is clearly insufficient to answer, return exactly: "ERROR: insufficient data for a reliable answer."
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


