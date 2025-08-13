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
    1. kind MUST be one of: "THINK", "ACT", "STOP".
       - If kind == "STOP":
         • text = the final user-facing answer. Concise, factual, no internal details.
       - If kind == "ACT":
         • text = a single, clear, executable instruction in plain language (e.g., "send hi to discord channel 1234", "search nytimes for articles about Artificial Intelligence").
         • Only include ONE action; no multi-step plans.
       - If kind == "THINK":
         • text = a brief reasoning step describing what to figure out next; no tool names or API parameters.
    2. Be specific and build on the latest Observation if present. Do not repeat earlier steps verbatim.
    3. Output ONLY the JSON object. No markdown, no commentary.
    </instructions>

    <output_format>
    {{"kind": "THINK|ACT|STOP", "text": "..."}}
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
        # No cached last step fields; loop branches directly on returned step_type

    def record_step(self, step_type: str, text: str) -> str:
        label = (step_type or "").upper()
        self.lines.append(f"{label}: {text}")
        return label

    def append_action(self, tool_id: str, observation: Any) -> None:
        self.lines.append(f"ACT_EXECUTED: tool_id={tool_id}")
        self.lines.append(f"OBSERVATION: {str(observation)}")
        # No policy cache to reset; branching is driven by the next _think() output

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
        logger.info("ReACT reasoner started", goal=goal, max_turns=self.max_turns)

        for _ in range(self.max_turns):
            if state.complete:
                break

            # Always THINK first, then ACT if requested, STOP if final
            step_type_raw, step_text = self._think(state)
            step_type = state.record_step(step_type_raw, step_text)

            if step_type == "STOP":
                state.final_answer = step_text
                state.complete = True
                logger.info("reasoning_complete", reason="final_thought", turns=len(state.lines))
                break

            if step_type == "ACT":
                try:
                    tool_id, params, observation = self._act(state, step_text)
                    state.append_action(tool_id, observation)
                    obs_preview = str(observation)
                    if len(obs_preview) > 200:
                        obs_preview = obs_preview[:200] + "..."
                    logger.info("tool_executed",tool_id=tool_id, params=params if isinstance(params, dict) else None, observation_preview=obs_preview)
                except ToolSelectionError as exc:
                    state.lines.append(f"OBSERVATION: ERROR: ToolSelectionError: {str(exc)}")
                    logger.warning("tool_selection_failed", error=str(exc))
                except ToolExecutionError as exc:
                    state.lines.append(f"OBSERVATION: ERROR: ToolExecutionError: {str(exc)}")
                    logger.error("tool_execution_failed", error=str(exc))
                except Exception as exc:
                    state.lines.append(f"OBSERVATION: ERROR: UnexpectedError: {str(exc)}")
                    logger.error("tool_unexpected_error", error=str(exc), exc_info=True)
            else:
                # step_type == THINK, just proceed to next iteration
                logger.info("thought_generated", thought=str(step_text)[:200] + ("..." if step_text and len(str(step_text)) > 200 else ""))

        if not state.complete and not state.final_answer:
            logger.warning("max_turns_reached", max_turns=self.max_turns, turns=len(state.lines))

        # Do not synthesize here; agent-level summarizer will create final answer
        transcript = "\n".join(state.lines)
        success = state.complete
        return ReasoningResult(final_answer=state.final_answer or "", iterations=len(state.lines), success=success, transcript=transcript)

    def _think(self, state: ReactState) -> Tuple[str, str]:
        prompt = _THINK_PROMPT.format(transcript=state.transcript())
        try:
            obj = self.llm.prompt_to_json(prompt, max_retries=0) or {}
            step_type_raw = (obj.get("kind") or "").strip().upper()
            text = (obj.get("text") or "").strip()
            legacy_to_new = {"THOUGHT": "THINK", "ACTION": "ACT", "FINAL": "STOP"}
            if step_type_raw in {"THINK", "ACT", "STOP"} and text:
                return step_type_raw, text
            mapped = legacy_to_new.get(step_type_raw)
            if mapped and text:
                return mapped, text
            logger.error("Invalid think output", kind=step_type_raw, text_present=bool(text))
        except Exception:
            pass
        return "THINK", "Continuing reasoning to determine next step."

    def _act(self, state: ReactState, action_text: str) -> Tuple[str, Dict[str, Any], Any]:
        query = action_text

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


