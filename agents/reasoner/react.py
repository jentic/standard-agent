from __future__ import annotations

import json
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


# ----------------------------- State model -----------------------------


class ReactState:
    def __init__(self, goal: str):
        self.goal: str = goal
        self.final_answer: Optional[str] = None
        self.complete: bool = False
        self.lines: List[str] = [f"Goal: {goal}"]
        self.last_kind: Optional[str] = None  # "THOUGHT" | "ACTION" | "FINAL"
        self.last_text: Optional[str] = None

    def append_thought(self, kind: str, text: str) -> None:
        self.last_kind = kind
        self.last_text = text
        self.lines.append(f"{kind}: {text}")

    def append_action(self, tool_id: str, observation: Any) -> None:
        self.lines.append(f"ACTION_EXECUTED: tool_id={tool_id}")
        self.lines.append(f"OBSERVATION: {str(observation)}")
        # After an action executes, clear last_kind to allow policy to REASON next
        self.last_kind = None
        self.last_text = None

    def transcript(self) -> str:
        return "\n".join(self.lines)


# ----------------------------- Local exception -------------------------


class ToolSelectionError(Exception):
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
        state = ReactState(goal=goal)
        logger.info("implicit_run_start", goal=goal, max_turns=self.max_turns)

        for _ in range(self.max_turns):
            if state.complete:
                break

            decision = self._decide(state)
            logger.info("policy_decision", decision=decision, turns=len(state.lines))

            if decision == "HALT":
                state.complete = True
                logger.info("reasoning_complete", reason="policy_halt", turns=len(state.lines))
                break

            if decision == "REASON":
                node = self._think(state)
                kind, text = node
                state.append_thought(kind, text)
                if kind == "FINAL":
                    state.final_answer = text
                    state.complete = True
                    logger.info("reasoning_complete", reason="final_thought", turns=len(state.lines))
                    break
                preview = text
                logger.info(
                    "thought_generated",
                    thought=str(preview)[:200] + ("..." if preview and len(str(preview)) > 200 else ""),
                )
            else:
                try:
                    tool_id, params, observation = self._act(state)
                    state.append_action(tool_id, observation)
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
                    state.lines.append(f"OBSERVATION: ERROR: ToolSelectionError: {str(exc)}")
                    logger.warning("tool_selection_failed", error=str(exc))
                except ToolExecutionError as exc:
                    state.lines.append(f"OBSERVATION: ERROR: ToolExecutionError: {str(exc)}")
                    logger.error("tool_execution_failed", error=str(exc))
                except Exception as exc:
                    state.lines.append(f"OBSERVATION: ERROR: UnexpectedError: {str(exc)}")
                    logger.error("tool_unexpected_error", error=str(exc), exc_info=True)

        if not state.complete and not state.final_answer:
            state.final_answer = "ERROR: reasoning stopped after reaching the maximum number of steps."
            logger.warning("max_turns_reached", max_turns=self.max_turns, turns=len(state.lines))

        final_answer = state.final_answer or self._summarize(state)
        success = state.complete or bool(final_answer)
        return ReasoningResult(final_answer=final_answer, iterations=len(state.lines), success=success)

    # ----------------------------- Internals ---------------------------

    def _decide(self, state: ReactState) -> str:
        """ReACT policy: FINAL -> HALT; ACTION -> TOOL; else REASON."""
        if state.last_kind == "FINAL":
            return "HALT"
        if state.last_kind == "ACTION":
            return "TOOL"
        return "REASON"

    def _think(self, state: ReactState) -> Tuple[str, str]:
        prompt = _THINK_PROMPT.format(transcript=state.transcript())
        try:
            obj = self.llm.prompt_to_json(prompt, max_retries=0) or {}
            kind_raw = (obj.get("kind") or "").strip().upper()
            text = (obj.get("text") or "").strip()
            if kind_raw in {"FINAL", "ACTION", "THOUGHT"} and text:
                return kind_raw, text
            logger.error("Invalid think output", kind=kind_raw, text_present=bool(text))
        except Exception:
            pass
        return "THOUGHT", "Continuing reasoning to determine next step."

    def _act(self, state: ReactState) -> Tuple[str, Dict[str, Any], Any]:
        if state.last_kind != "ACTION" or not state.last_text:
            raise ToolSelectionError("Act called without an Action node")
        query = state.last_text

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

    def _generate_params(self, tool: ToolBase, state: ReactState, step_text: str) -> Dict[str, Any]:
        schema = tool.get_parameters() or {}
        allowed_keys = ",".join(schema.keys()) if isinstance(schema, dict) else ""
        data: Dict[str, Any] = {"trace": state.transcript()}

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

    def _summarize(self, state: ReactState) -> str:
        if state.final_answer:
            return state.final_answer
        prompt = _SUMMARY_PROMPT.format(transcript=state.transcript())
        reply = self.llm.prompt(prompt)
        return reply or "ERROR: insufficient data for a reliable answer."


