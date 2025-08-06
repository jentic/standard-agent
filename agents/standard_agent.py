"""
StandardAgent

Lightweight façade that wires together the core runtime services (LLM, memory,
external tools) with a pluggable *reasoner* implementation.  The
agent owns the services; the reasoner simply uses what the agent provides.
"""
from __future__ import annotations

from  collections.abc import MutableMapping
from  collections import deque
from  agents.reasoner.base import BaseReasoner, ReasoningResult
from  agents.llm.base_llm import BaseLLM
from  agents.tools.base import JustInTimeToolingBase
from  agents.goal_resolver.base import BaseGoalResolver

from  uuid import uuid4
from  enum import Enum

class AgentState(str, Enum):
    READY               = "READY"
    BUSY                = "BUSY"
    NEEDS_ATTENTION     = "NEEDS_ATTENTION"

class StandardAgent:
    """Top-level class that orchestrates the main components of the agent framework."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        reasoner: BaseReasoner,

        # Optionals
        goal_resolver: BaseGoalResolver = None,
        conversation_history_window: int = 5
    ):
        """Initializes the agent.

        Args:
            llm: The language model instance.
            tools: The interface for accessing external tools.
            memory: The memory backend.
            reasoner: The reasoning engine that will use the services.

            goal_resolver: A component to preprocess the user's goal.
            conversation_history_window: The number of past interactions to keep in memory.
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.reasoner = reasoner

        self.goal_resolver = goal_resolver
        self.memory.setdefault("conversation_history", deque(maxlen=conversation_history_window))

        self._state: AgentState = AgentState.READY

    @property
    def state(self) -> AgentState:
        return self._state

    def solve(self, goal: str) -> ReasoningResult:
        """Solves a goal synchronously (library-style API)."""
        run_id = uuid4().hex

        if self.goal_resolver:
            revised_goal, user_message = self.goal_resolver.process(goal, self.memory.get("conversation_history"))
            if user_message:
                self.memory["conversation_history"].append({ "goal": goal, "result": f"Asked for clarification: {user_message}"})
                return ReasoningResult(success=False, final_answer=user_message)
            goal = revised_goal

        self.memory[f"goal:{run_id}"] = goal
        self._state = AgentState.BUSY

        try:
            result = self.reasoner.run(goal)
            self.memory[f"result:{run_id}"] = result
            self.memory["conversation_history"].append({"goal": goal, "result": result.final_answer})
            self._state = AgentState.READY
            return result

        except Exception:
            self._state = AgentState.NEEDS_ATTENTION
            raise
