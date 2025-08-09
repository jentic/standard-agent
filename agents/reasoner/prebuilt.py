from __future__ import annotations
from collections.abc import MutableMapping

from agents.reasoner.sequential.planners.bullet_list import BulletListPlan
from agents.reasoner.sequential.reasoner import SequentialReasoner
from agents.reasoner.sequential.reflectors.rewoo import ReWOOReflect
from .sequential.executors.rewoo import ReWOOExecuteStep
from .sequential.summarizer.default import DefaultSummarizeResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase

from agents.reasoner.implicit.reasoner import ImplicitReasoner
from agents.reasoner.implicit.policy.react import ReACTPolicy
from agents.reasoner.implicit.think.react import ReACTThink
from agents.reasoner.implicit.act.jit_act import JustInTimeAct
from agents.reasoner.implicit.summarizer.summarize import DefaultImplicitSummarizer


class ReWOOReasoner(SequentialReasoner):
    """
    Pre-wired SequentialReasoner that follows the ReWOO methodology:
      • Bullet-list planner
      • ReWOO step-executor
      • ReWOO reflector
      • LLM-based final-answer summarizer
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        max_retries: int = 2
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            plan=BulletListPlan(llm=llm),
            execute_step=ReWOOExecuteStep(llm=llm, tools=tools, memory=memory),
            reflect=ReWOOReflect(llm=llm, tools=tools, memory=memory, max_retries=max_retries),
            summarize_result=DefaultSummarizeResult(llm=llm),
        )


class ReACTReasoner(ImplicitReasoner):
    """Pre-wired ImplicitReasoner configured for ReACT-style operation."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        max_turns: int = 20,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            decide=ReACTPolicy(llm=llm),
            think=ReACTThink(llm=llm),
            act=JustInTimeAct(llm=llm, tools=tools),
            summarize=DefaultImplicitSummarizer(llm=llm),
            max_turns=max_turns,
        )
