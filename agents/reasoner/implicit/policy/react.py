from __future__ import annotations

from collections.abc import MutableMapping

from agents.reasoner.implicit.reasoner import ImplicitState
from agents.reasoner.implicit.models import ReasonNode, ReasonKind
from agents.reasoner.implicit.reasoner import Decision
from agents.reasoner.implicit.policy.base import DecidePolicy

class ReACTPolicy(DecidePolicy):
    """Deterministic rule-based policy for ReACT-style agents.

    Rules:
    - If latest node kind == FINAL → HALT
    - If latest node kind == ACTION → TOOL
    - Else → REASON
    """


    def __call__(self, state: ImplicitState, memory: MutableMapping) -> Decision:

        last = state.turns[-1] if state.turns else None
        if last and isinstance(last.thought, ReasonNode):
            if last.thought.kind == ReasonKind.FINAL:
                return Decision.HALT
            if last.thought.kind == ReasonKind.ACTION:
                return Decision.TOOL

        return Decision.REASON


