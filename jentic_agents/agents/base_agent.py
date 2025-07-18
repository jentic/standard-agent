"""BaseAgent
================

Lightweight façade that wires together the core runtime services (LLM, memory,
external tools) with a pluggable *reasoner* implementation.  The
agent owns the services; the reasoner simply uses what the agent provides.
"""
from __future__ import annotations

# Local imports use the new, refactored reasoner package
from .models import Goal
from ..inbox.base_inbox import BaseInbox
from ..memory.base_memory import BaseMemory
from ..outbox.base_outbox import BaseOutbox
from ..reasoners.base_reasoner import BaseReasoner  # type: ignore
from ..reasoners.models import ReasoningResult
from ..tools.interface import ToolInterface
from ..utils.llm import BaseLLM


class BaseAgent:
    """Wires together a reasoner with shared services (LLM, memory, tools)."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        memory: BaseMemory,
        tool_interface: ToolInterface,
        reasoner: BaseReasoner,
    ):
        """Initializes the agent and injects services into the reasoner.

        Args:
            llm: The language model instance.
            memory: The memory backend.
            tool_interface: The interface for accessing external tools.
            reasoner: The reasoning engine that will use the services.
        """
        self.llm = llm
        self.memory = memory
        self.tool_interface = tool_interface
        self.reasoner = reasoner

        # Inject shared services into the reasoner if it expects them
        for attr, value in (
            ("llm", llm),
            ("memory", memory),
            ("tool_interface", tool_interface),
        ):
            if hasattr(self.reasoner, attr):
                setattr(self.reasoner, attr, value)

    def solve(self, goal: Goal) -> ReasoningResult:
        """Solves a goal synchronously (library-style API)."""
        # Persist goal for traceability if memory supports it
        if hasattr(self.memory, "store"):
            self.memory.store("current_goal", goal)

        result = self.reasoner.run(goal.text, **goal.metadata)

        if hasattr(self.memory, "store"):
            self.memory.store("last_result", result.model_dump())

        return result

    def tick(self, inbox: BaseInbox, outbox: BaseOutbox) -> bool:
        """Processes one item from an inbox (service-style API).

        Returns:
            True if a goal was processed, False otherwise.
        """
        if not (goal_text := inbox.get_next_goal()):
            return False

        goal = Goal(text=goal_text)
        result = self.solve(goal)
        outbox.send(result)
        inbox.acknowledge_goal(goal_text)
        return True
