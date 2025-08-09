from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from agents.reasoner.implicit.policy.decision import Decision


class DecidePolicy(ABC):
    """Policy component that decides the next mode (REASON | TOOL | HALT).

    Why this exists (even when the default can be simple):
    - Separation of concerns: keeps the core loop lean; decision logic lives here.
    - Composability: OSS users can swap in custom policies (safety gates, anti-stall,
      budget/latency caps, domain heuristics) without touching the reasoner loop.
    - Testability: decision behavior can be unit-tested in isolation.
    - Zero-cost default: a rule-based policy requires no LLM calls, so keeping this
      extension point does not impose runtime overhead.
    """

    @abstractmethod
    def __call__(self, state: "ImplicitState", memory: MutableMapping) -> Decision:
        ...