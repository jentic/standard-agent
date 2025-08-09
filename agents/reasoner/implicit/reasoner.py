from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
from agents.reasoner.sequential.exceptions import ToolSelectionError
from agents.tools.exceptions import ToolExecutionError

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

    def get_reasoning_transcript(self) -> List[str]:
        lines: List[str] = [f"Goal: {self.goal}"]
        for t in self.turns:
            if t.thought:
                lines.append(f"{t.thought.kind.name}: {t.thought.text}")
            if t.action:
                tool = t.action.get("tool_id") if isinstance(t.action, dict) else str(t.action)
                lines.append(f"ACTION_EXECUTED: tool_id={tool}")
            if t.observation is not None:
                lines.append(f"OBSERVATION: {str(t.observation)}")
        return "\n".join(lines) if lines else lines


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

            decision = self.decide(state)

            logger.info("policy_decision", decision=decision.value, turns=len(state.turns))
            if decision == Decision.HALT:
                state.is_complete = True
                logger.info("reasoning_complete", reason="policy_halt", turns=len(state.turns))
                break

            turn = Turn()
            if decision == Decision.REASON:
                node = self.think(state)
                turn.thought = node
                if node.kind == ReasonKind.FINAL:
                    state.final_answer = node.text
                    state.is_complete = True
                    state.turns.append(turn)
                    logger.info("reasoning_complete", reason="final_thought", turns=len(state.turns))
                    break
                preview = node.text
                logger.info("thought_generated", thought=str(preview)[:200] + ("..." if preview and len(str(preview)) > 200 else ""))
            else:
                try:
                    tool_id, params, observation = self.act(state)
                    turn.action = {"tool_id": tool_id, "params": params}
                    turn.observation = observation
                    obs_preview = str(observation)
                    if len(obs_preview) > 200:
                        obs_preview = obs_preview[:200] + "..."
                    logger.info("tool_executed", tool_id=tool_id, param_count=len(params) if isinstance(params, dict) else None, observation_preview=obs_preview,)
                except ToolSelectionError as exc:
                    turn.observation = f"ERROR: ToolSelectionError: {str(exc)}"
                    logger.warning("tool_selection_failed", error=str(exc))
                except ToolExecutionError as exc:
                    turn.observation = f"ERROR: ToolExecutionError: {str(exc)}"
                    logger.error("tool_execution_failed", error=str(exc))
                except Exception as exc:
                    turn.observation = f"ERROR: UnexpectedError: {str(exc)}"
                    logger.error("tool_unexpected_error", error=str(exc), exc_info=True)

            state.turns.append(turn)

        if not state.is_complete and not state.final_answer:
            state.final_answer = "ERROR: reasoning stopped after reaching the maximum number of steps."
            logger.warning("max_turns_reached", max_turns=self.max_turns, turns=len(state.turns))

        # Synthesize final answer if not already provided
        final_answer = state.final_answer or self.summarize(state)
        success = state.is_complete or bool(final_answer)
        return ReasoningResult(final_answer=final_answer, iterations=len(state.turns), success=success)



