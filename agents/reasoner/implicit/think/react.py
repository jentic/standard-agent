from __future__ import annotations

from collections.abc import MutableMapping
from typing import List
from textwrap import dedent

from agents.reasoner.implicit.reasoner import ImplicitState
from agents.reasoner.implicit.think.base import Think


THINK_PROMPT = dedent(
    """
    <role>
    You are the Reasoning Engine within an agent. Your job is to think step-by-step to progress the goal.
    You do not call tools here; you produce either:
    - a next Thought (analysis), or
    - an ACTION: {{json}} when the next move requires a tool, or
    - FINAL: <answer> when sufficient.
    </role>

    <goal>
    Achieve the user's goal using only the information in the transcript below.
    </goal>

    <transcript>
    {transcript}
    </transcript>

    <instructions>
    1. If sufficient for a user-facing answer, start your output with EXACTLY 'FINAL: ' then the answer.
    2. If a single tool action is the correct next move, output EXACTLY one line: 'ACTION: {{json}}'
       The JSON must be a single object with these keys:
       - domain: string (e.g., "nytimes", "discord", "slack", "gmail")
       - intent: string (e.g., "search_articles", "send_message")
       - inputs_ref: optional string name of data to send (e.g., "summary")
       - args: optional object of human-level arguments (example: channel_id = "123")
       Do not include API parameters; keep it provider-agnostic.
    3. Otherwise, output a single Thought (no prefix) that advances toward an actionable step. Be specific and build on the latest Observation if present.
    4. Do not repeat prior Thoughts verbatim.
    </instructions>

    <output_format>
    - FINAL: <answer>
    - ACTION: {{json}}
    - <Thought sentence>
    - No markdown, no code fences, no extra labels
    </output_format>
    """
).strip()


class ReACTThink(Think):
    def __init__(self, *, llm) -> None:
        super().__init__(llm=llm)

    def __call__(self, state: "ImplicitState", memory: MutableMapping) -> str:
        lines: List[str] = [f"Goal: {state.goal}"]
        for t in state.turns[-6:]:
            if t.thought:
                lines.append(f"Thought: {t.thought}")
            if t.action:
                tool = t.action.get("tool_id") if isinstance(t.action, dict) else str(t.action)
                lines.append(f"Action: tool_id={tool}")
            if t.observation is not None:
                lines.append(f"Observation: {str(t.observation)}")

        prompt = THINK_PROMPT.format(transcript="\n".join(lines))
        return self.llm.prompt(prompt)


