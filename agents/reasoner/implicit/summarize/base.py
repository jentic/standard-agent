from __future__ import annotations

from abc import ABC, abstractmethod
from agents.reasoner.implicit.reasoner import ImplicitState


class Summarizer(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState) -> str:
        ...
