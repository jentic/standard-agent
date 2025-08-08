from __future__ import annotations

from collections.abc import MutableMapping
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, List, Optional, TYPE_CHECKING

from agents.reasoner.base import BaseReasoner
from agents.reasoner.base import ReasoningResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase
from agents.tools.exceptions import ToolError, ToolCredentialsMissingError
from agents.reasoner.sequential.exceptions import ReasoningError

if TYPE_CHECKING:
    from agents.reasoner.sequential.planners.base import Plan
    from agents.reasoner.sequential.executors.base import ExecuteStep
    from agents.reasoner.sequential.reflectors.base import Reflect
    from agents.reasoner.sequential.summarizer.base import SummarizeResult


class StepStatus(str, Enum):
    """The lifecycle status of a single reasoning step."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Step:
    """A single bullet-plan step produced by an LLM planner."""

    text: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None

    # Explicit data-flow metadata (optional)
    output_key: Optional[str] = None  # snake_case key produced by this step
    input_keys: List[str] = field(default_factory=list)  # keys this step consumes

    # Error handling / reflection bookkeeping
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class ReasonerState:
    """Mutable state shared across reasoning iterations."""

    goal: str
    plan: Deque[Step] = field(default_factory=deque)
    history: List[str] = field(default_factory=list)
    is_complete: bool = False


class SequentialReasoner(BaseReasoner):
    DEFAULT_MAX_ITER = 20

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        plan: Plan | None = None,
        execute_step: ExecuteStep,
        reflect: Reflect | None = None,
        summarize_result: SummarizeResult,
        max_iterations: int = DEFAULT_MAX_ITER
    ):
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.plan = plan
        self.execute_step = execute_step
        self.reflect = reflect
        self.summarize_result = summarize_result
        self.max_iterations = max_iterations

    # ---------- main loop ---------------------------------------
    def run(self, goal: str) -> ReasoningResult:

        state = ReasonerState(goal=goal)

        # Plan
        if self.plan:
            state.plan = self.plan(goal)
            if not state.plan:
                raise RuntimeError("Planner produced an empty plan")
        else:
            state.plan = deque([Step(text=goal)])

        iterations = 0

        # Iterate through plan, executing and reflecting
        while state.plan and iterations < self.max_iterations and not state.is_complete:
            step = state.plan.popleft()
            try:
                self.execute_step(step, state)
                iterations += 1
            except (ReasoningError, ToolError) as exc:
                if isinstance(exc, ToolCredentialsMissingError):
                    state.history.append(f"Tool Unauthorized: {str(exc)}")
                if self.reflect:
                    self.reflect(exc, step, state)
                else:
                    raise

        # Summarize the result
        final_answer = self.summarize_result(state)

        success = state.is_complete and not state.plan

        return ReasoningResult(
            final_answer=final_answer,
            iterations=iterations,
            tool_calls=[],
            success=success,
        )
