from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from agents.reasoner.implicit.policy.decision import Decision


class DecidePolicy(ABC):
    """Arbitrates the agent's next mode of operation.

    Role
    - Given the current implicit state and memory, decide one of the modes:
      REASON (think again), TOOL (execute an external tool), or HALT (stop).

    Why a standalone component
    - Separation of concerns: the reasoner loop remains a simple orchestrator;
      switching logic lives here and can evolve independently.
    - Composability: downstream users can swap policies without touching the core
      loop to inject domain rules, safety checks, or cost/latency preferences.
    - Testability: decision strategies can be unit-tested in isolation.

    Typical strategies (non-exhaustive)
    - Rule-based (default): if latest node is FINAL → HALT; if ACTION → TOOL; else → REASON.
    - Safety/compliance gate: block TOOL for risky actions; force REASON to reassess.
    - Anti-stall/quality: detect repetition or non-progress and choose HALT or force REASON.
    - Budget/latency aware: prefer REASON or HALT under token/time constraints.
    - Domain heuristics: e.g., always REASON after an Observation before any TOOL.
    - Human-in-the-loop: require approval for certain ACTION texts before TOOL.

    Return value
    - Must be a Decision enum value (Decision.REASON | Decision.TOOL | Decision.HALT).
    """

    @abstractmethod
    def __call__(self, state: "ImplicitState") -> Decision:
        ...