from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from agents.reasoner.base import BaseReasoner, ReasoningResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase
from collections.abc import MutableMapping

from agents.reasoner.implicit.policy.base import DecidePolicy
from agents.reasoner.implicit.think.base import Think
from agents.reasoner.implicit.act.base import Act
from agents.reasoner.implicit.summarizer.base import Summarizer
from agents.reasoner.implicit.models import ReasonNode, ReasonKind
from agents.reasoner.implicit.policy.decision import Decision

from utils.logger import get_logger
logger = get_logger(__name__)

 

@dataclass
class Turn:
    thought: Optional[ReasonNode] = None
    action: Optional[Dict[str, Any]] = None
    observation: Optional[Any] = None


@dataclass
class ImplicitState:
    goal: str
    turns: List[Turn] = field(default_factory=list)
    is_complete: bool = False
    final_answer: Optional[str] = None
    # Cached for O(1) access by components (e.g., Act)
    last_thought: Optional[ReasonNode] = None
    last_observation: Optional[Any] = None


class ImplicitReasoner(BaseReasoner):
    """Generic implicit reasoner: decides per turn whether to reason or act.

    Subclasses override the small decision/reason/act hooks to implement
    variants (e.g., ReACT) while reusing the core loop and summarization path.
    """

    DEFAULT_MAX_TURNS = 20

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        think: Think | None = None,
        act: Act | None = None,
        decide: DecidePolicy | None = None,
        summarize: Summarizer | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.max_turns = max_turns
        self.decide = decide
        self.think = think
        self.act = act
        self.summarize = summarize

    def run(self, goal: str) -> ReasoningResult:
        state = ImplicitState(goal=goal)
        logger.info("implicit_run_start", goal=goal, max_turns=self.max_turns)

        for _ in range(self.max_turns):
            if state.is_complete:
                break

            decision = self.decide(state, self.memory)

            logger.info("policy_decision", decision=decision.value, turns=len(state.turns))
            if decision == Decision.HALT:
                state.is_complete = True
                logger.info("reasoning_complete", reason="policy_halt", turns=len(state.turns))
                break

            turn = Turn()
            if decision == Decision.REASON:
                node = self.think(state, self.memory)
                turn.thought = node
                state.last_thought = node
                if node.kind == ReasonKind.FINAL:
                    state.final_answer = node.text
                    state.is_complete = True
                    state.turns.append(turn)
                    logger.info("reasoning_complete", reason="final_thought", turns=len(state.turns))
                    break
                preview = node.text
                logger.info("thought_generated", thought=str(preview)[:200] + ("..." if preview and len(str(preview)) > 200 else ""))
            else:  # TOOL
                tool_id, params, observation = self.act(state, self.memory)
                turn.action = {"tool_id": tool_id, "params": params}
                turn.observation = observation
                state.last_observation = observation
                obs_preview = str(observation)
                if len(obs_preview) > 200:
                    obs_preview = obs_preview[:200] + "..."
                logger.info("tool_executed", tool_id=tool_id, param_count=len(params) if isinstance(params, dict) else None, observation_preview=obs_preview,)

            state.turns.append(turn)

        if not state.is_complete and not state.final_answer:
            state.final_answer = "ERROR: reasoning stopped after reaching the maximum number of steps."
            logger.warning("max_turns_reached", max_turns=self.max_turns, turns=len(state.turns))

        # Synthesize final answer if not already provided
        final_answer = state.final_answer or self.summarize(state)
        success = state.is_complete or bool(final_answer)
        return ReasoningResult(final_answer=final_answer, iterations=len(state.turns), success=success)



