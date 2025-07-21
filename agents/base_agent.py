"""BaseAgent
================

Lightweight façade that wires together the core runtime services (LLM, memory,
external tools) with a pluggable *reasoner* implementation.  The
agent owns the services; the reasoner simply uses what the agent provides.
"""
from __future__ import annotations

from  agents.models import Goal
from  inbox.base_inbox import BaseInbox
from  memory.base_memory import BaseMemory
from  outbox.base_outbox import BaseOutbox
from  reasoners.base_reasoner import BaseReasoner  # type: ignore
from  reasoners.models import ReasoningResult
from  tools.interface import ToolInterface
from  utils.llm import BaseLLM
from  uuid import uuid4

class BaseAgent:
    """Wires together a reasoner with shared services (LLM, memory, tools)."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: ToolInterface,
        memory: BaseMemory,
        reasoner: BaseReasoner,
    ):
        """Initializes the agent and injects services into the reasoner.

        Args:
            llm: The language model instance.
            tools: The interface for accessing external tools.
            memory: The memory backend.
            reasoner: The reasoning engine that will use the services.
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.reasoner = reasoner

        # Explicit handshake to wire services into the reasoner
        self.reasoner.attach_services(llm=llm, tools=tools, memory=memory)

    def solve(self, goal: Goal) -> ReasoningResult:
        """Solves a goal synchronously (library-style API)."""
        run_id = uuid4().hex

        if hasattr(self.memory, "store"):
            self.memory.store(f"goal:{run_id}", goal)

        result = self.reasoner.run(goal.text, meta=goal.metadata)

        if hasattr(self.memory, "store"):
            self.memory.store(f"result:{run_id}", result.model_dump())

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
