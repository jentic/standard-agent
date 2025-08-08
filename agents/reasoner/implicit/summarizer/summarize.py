from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from agents.reasoner.implicit.reasoner import ImplicitState
from agents.reasoner.implicit.summarizer import Summarizer

class DefaultImplicitSummarizer(Summarizer):


    def __call__(self, state: ImplicitState) -> str:
        lines: List[str] = [f"Goal: {state.goal}"]
        for i, t in enumerate(state.turns, 1):
            if t.thought:
                lines.append(f"Thought {i}: {t.thought}")
            if t.action:
                lines.append(f"Action {i}: {t.action}")
            if t.observation is not None:
                lines.append(f"Observation {i}: {t.observation}")
        prompt = (
            "Synthesize a final answer to the goal using only the transcript of thoughts, actions, and observations.\n\n"
            + "\n".join(lines)
        )
        return self.llm.prompt(prompt)


