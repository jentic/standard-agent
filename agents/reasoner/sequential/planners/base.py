from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Deque

from agents.llm.base_llm import BaseLLM
from agents.reasoner.sequential.reasoner import Step


class Plan(ABC):
    def __init__(self, *, llm: BaseLLM):
        self.llm = llm

    @abstractmethod
    def __call__(self, goal: str) -> Deque[Step]:
        ...
