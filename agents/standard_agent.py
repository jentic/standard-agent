"""
StandardAgent

Lightweight façade that wires together the core runtime services (LLM, memory,
external tools) with a pluggable *reasoner* implementation.  The
agent owns the services; the reasoner simply uses what the agent provides.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from  agents.reasoner.base import BaseReasoner
from  agents.llm.base_llm import BaseLLM
from  agents.reasoner.base import ReasoningResult
from  agents.tools.base import JustInTimeToolingBase

from  uuid import uuid4
from  enum import Enum

class AgentState(str, Enum):
    READY               = "READY"
    BUSY                = "BUSY"
    NEEDS_ATTENTION     = "NEEDS_ATTENTION"

class StandardAgent:
    """Wires together a reasoner with shared services (LLM, memory, tools)."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
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


        self._state: AgentState = AgentState.READY

    @property
    def state(self) -> AgentState:
        return self._state

    def solve(self, goal: str) -> ReasoningResult:
        """Solves a goal synchronously (library-style API)."""
        run_id = uuid4().hex

        if hasattr(self.memory, "__setitem__"):
            self.memory[f"goal:{run_id}"] = goal

        self._state = AgentState.BUSY

        try:
            result = self.reasoner.run(goal)
            self.memory[f"result:{run_id}"] = result
            self._state = AgentState.READY
            return result

        except Exception:
            self._state = AgentState.NEEDS_ATTENTION
            raise
