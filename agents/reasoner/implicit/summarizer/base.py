from __future__ import annotations

from abc import ABC, abstractmethod
from agents.llm.base_llm import BaseLLM
from agents.reasoner.implicit.reasoner import ImplicitState


class Summarizer(ABC):
    """Produce the final user-facing answer from the implicit transcript.

    Responsibilities
    - Read the full `ImplicitState` transcript (THOUGHT/ACTION/ACTION_EXECUTED/
      OBSERVATION/FINAL) and return a concise final answer string
    - Prefer a FINAL line when present; otherwise synthesize from observations

    Contract
    - Must return a string; do not raise for ordinary insufficiency. If the
      evidence is inadequate, return a clear error message (e.g., "ERROR:
      insufficient data for a reliable answer.")

    """

    def __init__(self, *, llm: BaseLLM) -> None:
        self.llm = llm

    @abstractmethod
    def __call__(self, state: ImplicitState) -> str:
        ...
