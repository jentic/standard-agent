from __future__ import annotations

from typing import Any, Dict, List
from collections import deque

from reasoners.base_reasoner import BaseReasoner
from reasoners.models import ReasoningResult, ReasonerState, Step
from reasoners.sequential.interface import Planner, Reflector, StepExecutor, AnswerBuilder
from llm.base_llm import BaseLLM
from tools.interface import ToolInterface
from memory.base_memory import BaseMemory


class SequentialReasoner(BaseReasoner):
    DEFAULT_MAX_ITER = 20

    def __init__(
        self,
        *,
        llm: BaseLLM | None = None,
        tools: ToolInterface | None = None,
        memory: BaseMemory | None = None,
        planner: Planner | None = None,
        step_executor: StepExecutor,
        reflector: Reflector | None = None,
        answer_builder: AnswerBuilder
    ):
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.planner = planner
        self.step_executor = step_executor
        self.reflector = reflector
        self.answer_builder = answer_builder

        # Pass services to individual components
        self._pass_context_to_components()

    # ---------- Broadcasting context to components --------------
    def _pass_context_to_components(self) -> None:

        if self._llm is None and self._tools is None and self._memory is None:
            # not wired yet – nothing to broadcast
            return

        for comp in (self.planner, self.step_executor, self.reflector, self.answer_builder):
            if comp is not None:
                comp.set_services(llm=self.llm, tools=self.tools, memory=self.memory)

    # ---------- main loop ---------------------------------------
    def run(self, goal: str, *, meta: Dict[str, Any] | None = None) -> ReasoningResult:

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
