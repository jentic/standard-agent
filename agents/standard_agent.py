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
from  agents.goal_preprocessor.base import BaseGoalPreprocessor
from  agents.prompts import load_prompts

from  uuid import uuid4
import time
from  enum import Enum
from utils.logger import get_logger

logger = get_logger(__name__)

_PROMPTS = load_prompts("agent", required_prompts=["summarize"])

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
        goal_preprocessor: BaseGoalPreprocessor = None,
        conversation_history_window: int = 5
    ):
        """Initializes the agent.

        Args:
            llm: The language model instance.
            tools: The interface for accessing external tools.
            memory: The memory backend.
            reasoner: The reasoning engine that will use the services.

            goal_preprocessor: An OPTIONAL component to preprocess the user's goal.
            conversation_history_window: The number of past interactions to keep in memory.
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.reasoner = reasoner

        self.goal_preprocessor = goal_preprocessor
        self.memory.setdefault("conversation_history", deque(maxlen=conversation_history_window))

        self._state: AgentState = AgentState.READY

    @property
    def state(self) -> AgentState:
        return self._state

    def solve(self, goal: str) -> ReasoningResult:
        """Solves a goal synchronously (library-style API)."""
        run_id = uuid4().hex
        start_time = time.perf_counter()

        if self.goal_preprocessor:
            revised_goal, intervention_message = self.goal_preprocessor.process(goal, self.memory.get("conversation_history"))
            if intervention_message:
                self.memory["conversation_history"].append({ "goal": goal, "result": f"user intervention message: {intervention_message}"})
                return ReasoningResult(success=False, final_answer=intervention_message)
            goal = revised_goal

        self.memory[f"goal:{run_id}"] = goal
        self._state = AgentState.BUSY

        try:
            result = self.reasoner.run(goal)
            result.final_answer = self.llm.prompt(_PROMPTS["summarize"].format(goal=goal, history=getattr(result, "transcript", "")))

            self.memory[f"result:{run_id}"] = result
            self.memory["conversation_history"].append({"goal": goal, "result": result.final_answer})
            self._state = AgentState.READY

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "final_result",
                run_id=run_id,
                success=result.success,
                iterations=result.iterations,
                tool_calls_count=len(result.tool_calls or []),
                duration_ms=duration_ms,
                goal=goal,
                final_answer_preview=((result.final_answer[:200] + "…") if (result.final_answer and len(result.final_answer) > 200) else result.final_answer),
                model=getattr(self.llm, "model", None),
            )
            return result

        except Exception:
            self._state = AgentState.NEEDS_ATTENTION
            raise
