from __future__ import annotations

from collections.abc import MutableMapping
from textwrap import dedent
from typing import List

from agents.llm.base_llm import BaseLLM
from agents.reasoner.implicit.reasoner import ImplicitState
from .base import DecidePolicy


class ReACTPolicy(DecidePolicy):
    """ReACT policy powered by an LLM that decides REASON|TOOL|HALT from a short transcript.

    - Strict output: exactly one token among {REASON, TOOL, HALT}
    - Low temperature via llm config; caller controls model
    """

    def __init__(self, *, llm: BaseLLM, transcript_window: int = 6) -> None:
        self.llm = llm
        self.transcript_window = max(1, transcript_window)

    def __call__(self, state: ImplicitState, memory: MutableMapping) -> str:
        # Build compact transcript
        lines: List[str] = [f"Goal: {state.goal}"]
        for i, t in enumerate(state.turns[-self.transcript_window :], 1):
            if t.thought:
                lines.append(f"Thought: {t.thought}")
            if t.action:
                lines.append(f"Action: {t.action}")
            if t.observation is not None:
                lines.append(f"Observation: {t.observation}")
        transcript = "\n".join(lines)

        prompt = dedent(
            f"""
            You are a controller for a ReACT-style agent. Decide the next mode.

            Rules:
            - Output MUST be exactly one of: REASON, TOOL, HALT
            - An Action should be preceded by a Thought explaining why
            - After an Observation, prefer REASON to process it
            - If the latest Thought clearly contains a final answer (e.g., starts with 'FINAL:'), choose HALT

            Transcript:
            {transcript}

            Respond with one token: REASON or TOOL or HALT
            """
        ).strip()

        decision = self.llm.prompt(prompt).strip().upper()
        if decision not in {"REASON", "TOOL", "HALT"}:
            return "REASON"
        return decision


