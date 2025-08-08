from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.reasoner.base import BaseReasoner, ReasoningResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase
from collections.abc import MutableMapping

from .policy import DecidePolicy, ReACTPolicy
from .think import Think, LLMThink
from .act import Act, JustInTimeAct
from .summarizer import Summarizer, DefaultImplicitSummarizer


@dataclass
class Turn:
    thought: Optional[str] = None
    action: Optional[Dict[str, Any]] = None  # {"tool_id": str, "params": dict}
    observation: Optional[Any] = None


@dataclass
class ImplicitState:
    goal: str
    turns: List[Turn] = field(default_factory=list)
    is_complete: bool = False
    final_answer: Optional[str] = None


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
        self.decide = decide or ReACTPolicy(llm=llm)
        self.think = think or LLMThink(llm=llm)
        self.act = act or JustInTimeAct(llm=llm, tools=tools)
        self.summarize = summarize or DefaultImplicitSummarizer(llm=llm)

    # ---------- core loop -----------------------------------------
    def run(self, goal: str) -> ReasoningResult:
        state = ImplicitState(goal=goal)

        for _ in range(self.max_turns):
            if state.is_complete:
                break

            decision = self.decide(state, self.memory)
            if decision == "HALT":
                state.is_complete = True
                break

            turn = Turn()
            if decision == "REASON":
                turn.thought = self.think(state, self.memory)
                # Loop-level halt: Thought can signal final answer via 'FINAL:'
                if turn.thought and turn.thought.strip().upper().startswith("FINAL:"):
                    state.final_answer = turn.thought.split(":", 1)[1].strip()
                    state.is_complete = True
                    state.turns.append(turn)
                    break
            else:  # TOOL
                tool_id, params, observation = self.act(state, self.memory)
                turn.action = {"tool_id": tool_id, "params": params}
                turn.observation = observation

            state.turns.append(turn)

        if not state.is_complete and not state.final_answer:
            state.final_answer = "ERROR: reasoning stopped after reaching the maximum number of steps."

        # Synthesize final answer if not already provided
        final_answer = state.final_answer or self.summarize(state)
        success = state.is_complete or bool(final_answer)
        return ReasoningResult(final_answer=final_answer, iterations=len(state.turns), success=success)



