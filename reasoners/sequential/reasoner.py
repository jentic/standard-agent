from __future__ import annotations

from typing import Any, Dict, List

from reasoners.base_reasoner import BaseReasoner
from reasoners.models import ReasoningResult, ReasonerState, Step, RuntimeContext
from reasoners.sequential.interface import Planner, Reflector, StepExecutor, AnswerBuilder
from collections import deque


class SequentialReasoner(BaseReasoner):
    DEFAULT_MAX_ITER = 20

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        step_executor: StepExecutor,
        reflector: Reflector | None = None,
        answer_builder: AnswerBuilder
    ):
        self.planner = planner
        self.step_executor = step_executor
        self.reflector = reflector
        self.answer_builder = answer_builder

    # ---------- Broadcasting context to components --------------
    def _pass_context_to_components(self, ctx: RuntimeContext) -> None:
        for comp in (self.planner, self.step_executor, self.reflector, self.answer_builder):
            if comp is not None:
                comp.set_context(ctx)

    # ---------- main loop ---------------------------------------
    def run(self, goal: str, *, meta: Dict[str, Any] | None = None) -> ReasoningResult:
        if not hasattr(self, "llm"):
            raise RuntimeError("attach_services() was never called")

        meta = meta or {}
        max_iter = meta.get("max_iterations", self.DEFAULT_MAX_ITER)

        state = ReasonerState(goal=goal)

        # Plan (or seed single step)
        if self.planner:
            state.plan = self.planner.plan(goal)
        else:

            state.plan = deque([Step(text=goal)])

        if not state.plan:
            raise RuntimeError("Planner produced an empty plan")

        tool_calls: List[Dict[str, Any]] = []
        iterations = 0

        while state.plan and iterations < max_iter and not state.is_complete:
            step = state.plan.popleft()
            try:
                meta = self.step_executor.execute(step, state)
                if isinstance(meta, dict):
                    tool_calls.append(meta)
                iterations += 1
            except Exception as exc:
                if self.reflector:
                    self.reflector.handle(exc, step, state)
                else:
                    raise

        final_answer = self.answer_builder.build(state)
        success = state.is_complete and not state.plan

        return ReasoningResult(
            final_answer=final_answer,
            iterations=iterations,
            tool_calls=tool_calls,
            success=success,
        )
