from __future__ import annotations

from collections.abc import MutableMapping
from typing import List
from textwrap import dedent

from agents.reasoner.implicit.reasoner import ImplicitState
from agents.reasoner.implicit.models import ReasonNode, ReasonKind
from agents.reasoner.implicit.exceptions import ThinkFormatError
from agents.reasoner.implicit.think.base import Think

from utils.logger import get_logger
logger = get_logger(__name__)

THINK_PROMPT = dedent(
    """
    <role>
    You are the Reasoning Engine within an agent. Decide the immediate next step to progress the goal.
    Return exactly ONE JSON object with fields: kind and text.
    </role>

    <goal>
    Achieve the user's goal using only the transcript below.
    </goal>

    <transcript>
    {transcript}
    </transcript>

    <instructions>
    1. kind MUST be one of: "THOUGHT", "ACTION", "FINAL".
       - If kind == "FINAL":
         • text = the final user-facing answer. Concise, factual, no internal details.
       - If kind == "ACTION":
         • text = a single, clear, executable instruction in plain language (e.g., "send hi to discord channel 1234", "search nytimes for articles about Artificial Intelligence").
         • Only include ONE action; no multi-step plans.
       - If kind == "THOUGHT":
         • text = a brief reasoning step describing what to figure out next; no tool names or API parameters.
    2. Be specific and build on the latest Observation if present. Do not repeat earlier Thoughts verbatim.
    3. Output ONLY the JSON object. No markdown, no commentary.
    </instructions>

    <output_format>
    {{"kind": "THOUGHT|ACTION|FINAL", "text": "..."}}
    </output_format>
    """
).strip()


class ReACTThink(Think):
    def __init__(self, *, llm) -> None:
        super().__init__(llm=llm)

    def __call__(self, state: "ImplicitState", memory: MutableMapping):
        lines: List[str] = [f"Goal: {state.goal}"]
        for t in state.turns:
            if t.thought:
                lines.append(f"{t.thought.kind.name}: {t.thought.text}")
            if t.action:
                tool = t.action.get("tool_id") if isinstance(t.action, dict) else str(t.action)
                lines.append(f"ACTION_EXECUTED: tool_id={tool}")
            if t.observation is not None:
                lines.append(f"OBSERVATION: {str(t.observation)}")

        prompt = THINK_PROMPT.format(transcript="\n".join(lines))

        try:
            obj = self.llm.prompt_to_json(prompt, max_retries=0)
            kind = ((obj or {}).get("kind") or "").strip().upper()
            text = ((obj or {}).get("text") or "").strip()
            if kind in {"FINAL", "ACTION", "THOUGHT"} and text:
                return ReasonNode(kind=ReasonKind[kind], text=text)
            else:
                logger.error("Invalid think output", kind=kind, text_present=bool(text))
                raise ThinkFormatError(f"Invalid think output: kind='{kind}', text_present={bool(text)}")

        except Exception:
            # No second LLM call; avoid hidden retries
            return ReasonNode(kind=ReasonKind.THOUGHT, text="Continuing reasoning to determine next step.")


