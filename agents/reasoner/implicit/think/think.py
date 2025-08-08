from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import List

from agents.llm.base_llm import BaseLLM
from agents.reasoner.implicit.reasoner import ImplicitState


class Think(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> str:
        ...


class LLMThink(Think):
    def __init__(self, *, llm: BaseLLM) -> None:
        self.llm = llm

    def __call__(self, state: ImplicitState, memory: MutableMapping) -> str:
        lines: List[str] = [f"Goal: {state.goal}"]
        for i, t in enumerate(state.turns, 1):
            if t.thought:
                lines.append(f"Thought {i}: {t.thought}")
            if t.action:
                lines.append(f"Action {i}: {t.action}")
            if t.observation is not None:
                lines.append(f"Observation {i}: {t.observation}")
        prompt = (
            "You are reasoning step-by-step. Based on the goal and prior observations, "
            "produce the next concise thought (no tools).\n\n" + "\n".join(lines)
        )
        return self.llm.prompt(prompt)


