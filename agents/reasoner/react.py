from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import MutableMapping

from agents.reasoner.base import BaseReasoner, ReasoningResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase, ToolBase
from agents.tools.exceptions import ToolExecutionError

from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------- Prompts ---------------------------------

_THINK_PROMPT = dedent(
    """
    <role>
    You are the Reasoning Engine within an agent. Decide the immediate next step to progress the goal.
    Return exactly ONE JSON object with fields: kind and text.
    </role>

    <goal>
    Achieve the user's goal using only the transcript below.
    </goal>

    <transcript>
    {transcript}
    </transcript>

    <instructions>
    1. kind MUST be one of: "THOUGHT", "ACTION", "FINAL".
       - If kind == "FINAL":
         • text = the final user-facing answer. Concise, factual, no internal details.
       - If kind == "ACTION":
         • text = a single, clear, executable instruction in plain language (e.g., "send hi to discord channel 1234", "search nytimes for articles about Artificial Intelligence").
         • Only include ONE action; no multi-step plans.
       - If kind == "THOUGHT":
         • text = a brief reasoning step describing what to figure out next; no tool names or API parameters.
    2. Be specific and build on the latest Observation if present. Do not repeat earlier Thoughts verbatim.
    3. Output ONLY the JSON object. No markdown, no commentary.
    </instructions>

    <output_format>
    {{"kind": "THOUGHT|ACTION|FINAL", "text": "..."}}
    </output_format>
    """
).strip()

_TOOL_SELECTION_PROMPT = dedent(
    """
    <role>
    You are an expert orchestrator working within the Agent API ecosystem.
    Your job is to select the best tool to execute a specific step, using a list of available tools.
    Each tool may vary in API domain, supported actions, and required parameters.
    Return exactly one id from the candidates (or `none`).
    </role>

    <instructions>
    Analyze the provided step and evaluate all candidate tools. Use the scoring criteria to assess each tool's fitness for executing the step.
    Return the tool id with the highest total score. If no tool scores ≥60, return the word none.
    </instructions>

    <input>
    Step: {step}

    Tools (JSON):
    {tools_json}
    </input>

    <output_format>
    Respond with a single line containing exactly the selected tool id or `none`.
    </output_format>
    """
).strip()

_PARAMETER_GENERATION_PROMPT = dedent(
    """
    <role>
    You are a Parameter Builder within the Agent ecosystem.
    Your mission is to enable seamless API execution by generating precise parameters from step context and transcript data.
    </role>

    <goal>
    Generate precise JSON parameters for the specified API call by extracting relevant data from step context and transcript.
    </goal>

    <input>
    STEP: {step}
    DATA: {data}
    SCHEMA: {schema}
    ALLOWED_KEYS: {allowed_keys}
    </input>

    <data_extraction_rules>
    • Articles/News: Extract title/headline and URL fields, format as "Title: URL\n"
    • Arrays: Process each item, combine into formatted string
    • Nested Objects: Access properties using dot notation
    • Quantities: "a/an/one" = 1, "few" = 3, "several" = 5, numbers = exact
    • Array Slicing: Look for quantity constraints in the STEP text and slice accordingly
    • Never use placeholder text - always extract real data from DATA
    </data_extraction_rules>

    <instructions>
    1. Extract actual values using the rules
    2. CRITICAL: Check STEP text for quantity constraints
    3. Format content appropriately for the target API
    4. Generate valid parameters using only ALLOWED_KEYS
    5. CRITICAL: Only use parameters documented in the SCHEMA
    </instructions>

    <constraints>
    - Output ONLY valid JSON - no markdown or commentary
    - Use only keys from ALLOWED_KEYS
    - Extract actual data values from DATA
    </constraints>

    <output_format>
    Valid JSON object starting with {{ and ending with }}
    </output_format>
    """
).strip()

_SUMMARY_PROMPT = dedent(
    """
    <role>
    You are the Final Answer Synthesizer for an agent. Produce a clear, correct, and helpful final answer using only the transcript.
    </role>

    <goal>
    Provide the final answer to the user's goal based solely on the transcript of reasoning and tool use.
    </goal>

    <transcript>
    {transcript}
    </transcript>

    <instructions>
    1. The transcript consists of lines like:
       - THOUGHT: <text>
       - ACTION: <text>
       - ACTION_EXECUTED: tool_id=…
       - OBSERVATION: <text/json>
       - FINAL: <text>
    2. If a FINAL line exists, use its text (polish grammar only).
    3. Otherwise, synthesize a concise answer from OBSERVATION lines; use THOUGHT/ACTION for context.
    4. If evidence is insufficient, return exactly: "ERROR: insufficient data for a reliable answer."
    </instructions>

    <output_format>
    A short, user-facing answer in plain text.
    </output_format>
    """
).strip()


# ----------------------------- Data structures -------------------------


class Decision(Enum):
    REASON = "REASON"
    TOOL = "TOOL"
    HALT = "HALT"


class Kind(Enum):
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    FINAL = "FINAL"


@dataclass
class Node:
    kind: Kind
    text: str


@dataclass
class Turn:
    thought: Optional[Node] = None
    action: Optional[Dict[str, Any]] = None
    observation: Optional[Any] = None


@dataclass
class ImplicitState:
    goal: str
    turns: List[Turn]
    is_complete: bool = False
    final_answer: Optional[str] = None

    def get_reasoning_transcript(self) -> str:
        lines: List[str] = [f"Goal: {self.goal}"]
        for t in self.turns:
            if t.thought:
                lines.append(f"{t.thought.kind.name}: {t.thought.text}")
            if t.action:
                tool = t.action.get("tool_id") if isinstance(t.action, dict) else str(t.action)
                lines.append(f"ACTION_EXECUTED: tool_id={tool}")
            if t.observation is not None:
                lines.append(f"OBSERVATION: {str(t.observation)}")
        return "\n".join(lines)


# ----------------------------- Local exceptions ------------------------


class ImplicitReasoningError(Exception):
    """Raised when an error occurs during Implicit Reasoning."""


class ActionNodeMissingError(ImplicitReasoningError):
    """Raised when the Act component is invoked without a preceding Action node."""


class ThinkFormatError(ImplicitReasoningError):
    """Raised when the Think component returns an invalid or unparsable output."""


class ToolSelectionError(ImplicitReasoningError):
    """A suitable tool could not be selected from candidates."""


# ----------------------------- Reasoner --------------------------------


class ReACTReasoner(BaseReasoner):
    DEFAULT_MAX_TURNS = 20

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        max_turns: int = DEFAULT_MAX_TURNS,
        top_k: int = 15,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.max_turns = max_turns
        self.top_k = top_k

    def run(self, goal: str) -> ReasoningResult:
        state = ImplicitState(goal=goal, turns=[])
        logger.info("implicit_run_start", goal=goal, max_turns=self.max_turns)

        for _ in range(self.max_turns):
            if state.is_complete:
                break

            decision = self._decide(state)
            logger.info("policy_decision", decision=decision.value, turns=len(state.turns))

            if decision == Decision.HALT:
                state.is_complete = True
                logger.info("reasoning_complete", reason="policy_halt", turns=len(state.turns))
                break

            turn = Turn()
            if decision == Decision.REASON:
                node = self._think(state)
                turn.thought = node
                if node.kind == Kind.FINAL:
                    state.final_answer = node.text
                    state.is_complete = True
                    state.turns.append(turn)
                    logger.info("reasoning_complete", reason="final_thought", turns=len(state.turns))
                    break
                preview = node.text
                logger.info(
                    "thought_generated",
                    thought=str(preview)[:200] + ("..." if preview and len(str(preview)) > 200 else ""),
                )
            else:
                try:
                    tool_id, params, observation = self._act(state)
                    turn.action = {"tool_id": tool_id, "params": params}
                    turn.observation = observation
                    obs_preview = str(observation)
                    if len(obs_preview) > 200:
                        obs_preview = obs_preview[:200] + "..."
                    logger.info(
                        "tool_executed",
                        tool_id=tool_id,
                        param_count=len(params) if isinstance(params, dict) else None,
                        observation_preview=obs_preview,
                    )
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

        final_answer = state.final_answer or self._summarize(state)
        success = state.is_complete or bool(final_answer)
        return ReasoningResult(final_answer=final_answer, iterations=len(state.turns), success=success)

    # ----------------------------- Internals ---------------------------

    def _decide(self, state: ImplicitState) -> Decision:
        """ReACT policy: FINAL -> HALT; ACTION -> TOOL; else REASON."""
        last = state.turns[-1] if state.turns else None
        if last and isinstance(last.thought, Node):
            if last.thought.kind == Kind.FINAL:
                return Decision.HALT
            if last.thought.kind == Kind.ACTION:
                return Decision.TOOL
        return Decision.REASON

    def _think(self, state: ImplicitState) -> Node:
        prompt = _THINK_PROMPT.format(transcript=state.get_reasoning_transcript())
        try:
            obj = self.llm.prompt_to_json(prompt, max_retries=0) or {}
            kind_raw = (obj.get("kind") or "").strip().upper()
            text = (obj.get("text") or "").strip()
            if kind_raw in {"FINAL", "ACTION", "THOUGHT"} and text:
                return Node(kind=Kind[kind_raw], text=text)
            logger.error("Invalid think output", kind=kind_raw, text_present=bool(text))
            raise ThinkFormatError(f"Invalid think output: kind='{kind_raw}', text_present={bool(text)}")
        except Exception:
            return Node(kind=Kind.THOUGHT, text="Continuing reasoning to determine next step.")

    def _act(self, state: ImplicitState) -> Tuple[str, Dict[str, Any], Any]:
        last = state.turns[-1] if state.turns else None
        if not last or not isinstance(last.thought, Node) or last.thought.kind != Kind.ACTION:
            raise ActionNodeMissingError("Act called without an Action node")
        action_node: Node = last.thought
        query = action_node.text

        tool = self._select_and_load_tool(query)
        params = self._generate_params(tool, state, query)
        observation = self.tools.execute(tool, params)
        return tool.id, params, observation

    def _select_and_load_tool(self, query: str) -> ToolBase:
        candidates: List[ToolBase] = self.tools.search(query, top_k=self.top_k)
        logger.info("tool_search", query=query, top_k=self.top_k, candidate_count=len(candidates))
        tools_json = "\n".join([t.get_summary() for t in candidates])

        tool_resp = self.llm.prompt(_TOOL_SELECTION_PROMPT.format(step=query, tools_json=tools_json)) or ""
        resp = tool_resp.strip()

        if resp.lower() == "none" or not resp:
            raise ToolSelectionError(f"No suitable tool selected for step: {query}")

        tool = next((t for t in candidates if t.id == resp), None)
        if tool is None:
            # Fallback: try containment if LLM returned extra tokens
            tool = next((t for t in candidates if t.id in resp), None)
        if tool is None:
            raise ToolSelectionError(f"Selected tool id '{resp}' not in candidate list")

        return self.tools.load(tool)

    def _generate_params(self, tool: ToolBase, state: ImplicitState, step_text: str) -> Dict[str, Any]:
        schema = tool.get_parameters() or {}
        allowed_keys = ",".join(schema.keys()) if isinstance(schema, dict) else ""
        data: Dict[str, Any] = {"trace": state.get_reasoning_transcript()}

        params_raw = self.llm.prompt_to_json(
            _PARAMETER_GENERATION_PROMPT.format(
                step=step_text,
                data=json.dumps(data, ensure_ascii=False),
                schema=json.dumps(schema, ensure_ascii=False),
                allowed_keys=allowed_keys,
            ),
            max_retries=2,
        ) or {}
        params: Dict[str, Any] = {k: v for k, v in params_raw.items() if k in schema}
        logger.info("params_generated", tool_id=tool.id, params=params)
        return params

    def _summarize(self, state: ImplicitState) -> str:
        if state.final_answer:
            return state.final_answer
        prompt = _SUMMARY_PROMPT.format(transcript=state.get_reasoning_transcript())
        reply = self.llm.prompt(prompt)
        return reply or "ERROR: insufficient data for a reliable answer."


