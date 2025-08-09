from __future__ import annotations

from abc import ABC, abstractmethod
from agents.llm.base_llm import BaseLLM
from agents.reasoner.implicit.reasoner import ReasonNode, ImplicitState

class Think(ABC):
    """Generate the next reasoning step for the implicit agent.

    Responsibilities
    - Read the current `ImplicitState` (goal, transcript of Thoughts/Actions/Observations)
    - Decide the immediate next step and return a `ReasonNode` with:
      kind ∈ {THOUGHT, ACTION, FINAL} and text describing the content

    Contract
    - Must return a well-formed `ReasonNode` on every call; do not raise on minor
      formatting issues. Prefer internal fallbacks to THOUGHT when output is
      ambiguous so the loop can continue.
    - THOUGHT.text: brief reasoning that advances toward an actionable step.
    - ACTION.text: a single, clear instruction in plain language (no API params).
    - FINAL.text: concise, user-facing answer.

    """

    def __init__(self, *, llm: BaseLLM) -> None:
        self.llm = llm

    @abstractmethod
    def __call__(self, state: ImplicitState) -> ReasonNode:
        ...
