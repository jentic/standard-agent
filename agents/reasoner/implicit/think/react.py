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
    You do not call tools here; you only produce the next Thought or, if sufficient, the final answer.
    </role>

    <goal>
    Achieve the user's goal using only the information in the transcript below.
    </goal>

    <transcript>
    {transcript}
    </transcript>

    <instructions>
    1. If the transcript clearly contains enough information to answer the goal, output a final answer.
    2. Otherwise, produce the single next Thought that advances the plan toward an actionable step.
    3. Be specific and testable (what to verify or clarify next), but do NOT mention tool names or parameters.
    4. Do not repeat prior Thoughts verbatim; build on the latest Observation if present.
    </instructions>

    <output_format>
    - If final: start with EXACTLY 'FINAL: ' followed by the answer
    - Else: a single concise sentence describing the next Thought
    - No markdown, no code fences, no additional labels
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
                obs = str(t.observation)
                lines.append(f"Observation: {obs[:500] + ('…' if len(obs) > 500 else '')}")

        prompt = THINK_PROMPT.format(transcript="\n".join(lines))
        return self.llm.prompt(prompt)


