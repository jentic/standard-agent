"""
StandardAgent

Lightweight façade that wires together the core runtime services (LLM, memory,
external tools) with a pluggable *reasoner* implementation.  The
agent owns the services; the reasoner simply uses what the agent provides.
"""
from __future__ import annotations

from  uuid import uuid4
import time
import os
from  enum import Enum
from  zoneinfo import ZoneInfo
from  collections.abc import MutableMapping

from  agents.reasoner.base import BaseReasoner, ReasoningResult
from  agents.llm.base_llm import BaseLLM
from  agents.tools.base import JustInTimeToolingBase
from  agents.goal_preprocessor.base import BaseGoalPreprocessor
from  agents.prompts import load_prompts

from  utils.logger import get_logger
from  utils.observability import observe

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
        conversation_history_window: int = 5,

        # Runtime Context
        timezone: str | None = None,
    ):
        """Initializes the agent.

        Args:
            llm: The language model instance.
            tools: The interface for accessing external tools.
            memory: The memory backend.
            reasoner: The reasoning engine that will use the services.

            goal_preprocessor: An OPTIONAL component to preprocess the user's goal.
            conversation_history_window: The number of past interactions to keep in memory.

            Session Context
            timezone: Session timezone as IANA string like "America/New_York", "Europe/London" [https://www.iana.org/time-zones, https://en.wikipedia.org/wiki/List_of_tz_database_time_zones]

        """
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.reasoner = reasoner

        self.goal_preprocessor = goal_preprocessor
        self.conversation_history_window = self._normalize_history_window(
            conversation_history_window
        )
        self.memory.setdefault("conversation_history", [])

        self.memory["context"] = {}
        self.memory["context"]["timezone"] = self._resolve_timezone(timezone)

        self._state: AgentState = AgentState.READY

    @property
    def state(self) -> AgentState:
        return self._state

    @observe(root=True)
    def solve(self, goal: str) -> ReasoningResult:
        """Solves a goal synchronously (library-style API)."""
        run_id = uuid4().hex
        start_time = time.perf_counter()

        if self.goal_preprocessor:
            revised_goal, intervention_message = self.goal_preprocessor.process(goal, self.memory.get("conversation_history"))
            if intervention_message:
                self._record_interaction({"goal": goal, "result": f"user intervention message: {intervention_message}"})
                return ReasoningResult(success=False, final_answer=intervention_message)
            goal = revised_goal

        self._state = AgentState.BUSY

        try:
            result = self.reasoner.run(goal)
            # Truncate transcript to the last ~12KB to limit context size and avoid context-window errors
            result.final_answer = self.llm.prompt(_PROMPTS["summarize"].format(goal=goal, history=getattr(result, "transcript", "")[-12000:]))

            self._record_interaction({"goal": goal, "result": result.final_answer})
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

    def _record_interaction(self, entry: dict) -> None:
        if self.conversation_history_window <= 0:
            return

        self.memory["conversation_history"].append(entry)
        self.memory["conversation_history"][:] = self.memory["conversation_history"][-self.conversation_history_window:]

    @staticmethod
    def _normalize_history_window(window: int | None) -> int:
        """Validate ``conversation_history_window`` and return a safe non-negative int.

        Why this exists
            The previous logic ``max(0, conversation_history_window) if window is not None
            else 5`` silently accepted nonsense types (``float``, ``bool``, ``str``…) and
            *deferred* the failure to ``_record_interaction``, where a bad value would
            either crash with a confusing slice-time ``TypeError`` (``2.7``) or silently
            misbehave (``True`` → treated as 1 because ``bool`` is an ``int`` subclass).
            That's exactly the "memory leak on bad window size" failure mode tracked in
            issue #86 — invalid types meant trimming never ran, and history could grow
            without bound on the next solve().

        Rules
            * ``None``          → default of 5 (preserves backwards compatibility)
            * ``int >= 0``      → returned as-is
            * ``int < 0``       → clamped to 0 (also preserves prior behaviour;
              ``window=0`` is the documented "disable history" sentinel)
            * ``bool``          → rejected with ``TypeError`` (avoids the
              ``True`` → 1 / ``False`` → 0 silent coercion)
            * any other type    → ``TypeError`` with a clear message

        Args:
            window: The raw value passed by the caller.

        Returns:
            A validated non-negative ``int``.

        Raises:
            TypeError: ``window`` is not ``int`` or ``None`` (or is a ``bool``).
        """
        if window is None:
            return 5
        # NOTE: ``bool`` is a subclass of ``int`` in Python, so a plain
        # ``isinstance(window, int)`` check would silently accept ``True`` / ``False``.
        if isinstance(window, bool) or not isinstance(window, int):
            raise TypeError(
                "conversation_history_window must be an int or None, "
                f"got {type(window).__name__}={window!r}"
            )
        return max(0, window)

    @staticmethod
    def _resolve_timezone(tz_input: str | None) -> str | None:
        """Return a validated IANA timezone string from constructor/env, or None if unavailable."""

        for source, tz_name in (("constructor", tz_input), ("env", os.getenv("AGENT_TZ"))):
            if not tz_name:
                continue
            try:
                ZoneInfo(tz_name)  # validate IANA
                logger.info("timezone_resolved", source=source, kind="iana", tz=tz_name)
                return tz_name
            except Exception:
                logger.warning("invalid_timezone_string", source=source, tz_input=tz_name)
                continue

        logger.info("timezone_unset: env AGENT_TZ not set and no TZ passed in constructor ")
        return None
