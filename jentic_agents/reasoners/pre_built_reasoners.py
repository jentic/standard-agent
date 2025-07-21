from __future__ import annotations
from jentic_agents.reasoners.sequential.planners.bullet_list_planner import BulletListPlanner
from jentic_agents.reasoners.sequential.reasoner import SequentialReasoner
from jentic_agents.reasoners.sequential.reflectors.rewoo_reflector import ReWOOReflector
from .sequential.step_executors.rewoo_step_executor import ReWOOStepExecutor
from .sequential.synthesizers.final_answer_synthesizer import FinalAnswerSynthesizer


class ReWOOReasoner(SequentialReasoner):
    """
    Pre-wired SequentialReasoner that follows the ReWOO methodology:
      • Bullet-list planner
      • ReWOO step-executor
      • ReWOO reflector
      • LLM-based final-answer synthesizer
    """

    def __init__(
        self,
        *,
        max_retries: int = 2
    ):
        super().__init__(
            planner=BulletListPlanner(),
            step_executor=ReWOOStepExecutor(),
            reflector=ReWOOReflector(max_retries=max_retries),
            synthesizer=FinalAnswerSynthesizer(),
        )
