"""Abstract **plan-first** reasoner contract.

This base class formalises a plan-first, iterative execution architecture with optional self-reflection on failure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from jentic_agents.reasoners.models import ReasoningResult

from jentic_agents.reasoners.models import ReasonerState, Step
from jentic_agents.tools.interface import ToolInterface
from jentic_agents.memory.base_memory import BaseMemory
from jentic_agents.utils.llm import BaseLLM
from jentic_agents.utils.logger import get_logger

__all__ = ["BaseReWOOReasoner"]


class BaseReWOOReasoner(ABC):
    """Abstract *plan-first* reasoner.

    Concrete subclasses must implement:
      • :pymeth:`_generate_plan` – populate ``state.plan`` from the goal
      • :pymeth:`_execute_step` – perform one step; mark success/failure &
        optionally append new sub-steps to ``state.plan``
      • :pymeth:`_reflect_on_failure` – attempt self-healing after an exception
      • :pymeth:`_synthesize_final_answer` – build the human-readable answer
    """

    def __init__(
        self,
        *,
        tool: ToolInterface,
        memory: BaseMemory,
        llm: BaseLLM,
    ) -> None:
        self.tool = tool
        self._memory = memory
        self._llm = llm
        self._logger = get_logger(self.__class__.__name__)

    # Public API

    def run(self, goal: str, max_iterations: int = 20) -> ReasoningResult:  # noqa: D401
        """Execute the reasoning loop until completion or iteration cap.

        Returns a lightweight dict with summary metadata. Sub-classes may
        extend/replace this with a richer model if desired.
        """
        log = self._logger
        log.info("=== REASONER START === goal=%s", goal)
        state = ReasonerState(goal=goal)

        # PLAN
        self._generate_plan(state)
        if not state.plan:
            raise RuntimeError("Planner produced an empty plan")
        log.info("Plan generated with %d steps", len(state.plan))

        tool_calls: List[Dict[str, Any]] = []
        iteration = 0

        # EXECUTE LOOP
        while state.plan and iteration < max_iterations and not state.is_complete:
            iteration += 1
            step = state.plan.popleft()
            log.info("→ Iteration %d | step='%s' (indent=%d)" % (iteration, step.text, step.indent))

            # Classify step type before execution
            step.step_type = self._classify_step(step, state)
            log.info("Step classified as: %s", step.step_type)

            try:
                meta = self._execute_step(step, state)
                if isinstance(meta, dict):
                    tool_calls.append(meta)
            except Exception as exc:  # noqa: BLE001
                log.warning("Step failed: %s", exc)
                try:
                    self._reflect_on_failure(exc, step, state)
                except Exception as reflect_exc:  # noqa: BLE001
                    log.error("Reflection also failed: %s", reflect_exc)
            log.info("Step status after execution: %s", step.status)

        # FINISH
        final_answer = self._synthesize_final_answer(state)
        success = state.is_complete
        log.info("=== REASONER END === success=%s iterations=%d", success, iteration)

        return ReasoningResult(
            final_answer=final_answer,
            iterations=iteration,
            tool_calls=tool_calls,
            success=success,
        )

    # Sub-class hooks

    @abstractmethod
    def _generate_plan(self, state: ReasonerState) -> None:  # pragma: no cover
        """Populate ``state.plan`` with an initial deque[Step]."""

    @abstractmethod
    def _execute_step(self, step: Step, state: ReasonerState) -> Dict[str, Any] | None:  # pragma: no cover
        """Execute **one** plan step.

        Should set ``step.status`` and may append further steps to
        ``state.plan``.  Return optional metadata describing any tool call
        that occurred (id, params, duration, etc.) so the parent loop can
        aggregate statistics.
        """

    @abstractmethod
    def _reflect_on_failure(self, error: Exception, step: Step, state: ReasonerState) -> None:  # noqa: D401 pragma: no cover
        """Attempt to repair or rephrase a failed step.

        Typical strategies:
          • retry with modified parameters
          • ask the LLM to propose a different tool
          • skip / mark failed and continue
        """

    @abstractmethod
    def _synthesize_final_answer(self, state: ReasonerState) -> str:  # pragma: no cover
        """Combine state/history into the final user-facing answer."""

    @abstractmethod
    def _classify_step(self, step: Step, state: ReasonerState) -> Step.StepType:  # pragma: no cover
        """Determine if a step is a TOOL step or a REASONING step."""
