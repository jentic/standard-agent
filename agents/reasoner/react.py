from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import MutableMapping

from agents.reasoner.base import BaseReasoner, ReasoningResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase, ToolBase
from agents.tools.exceptions import ToolExecutionError, ToolCredentialsMissingError

from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------- Prompts ---------------------------------

_THINK_PROMPT = dedent(
    """
    <role>
    You are the Reasoning Engine within an agent. Decide the immediate next step to progress the goal.
    Return exactly ONE JSON object with fields: step_type and text.
    </role>

    <goal>
    Achieve the user's goal using only the transcript below.
    </goal>

    <transcript>
    {transcript}
    </transcript>

    <instructions>
    1. step_type MUST be one of: "THINK", "ACT", "STOP".
       - If step_type == "STOP":
         • text = the final user-facing answer. Concise, factual, no internal details.
       - If step_type == "ACT":
         • text = a single, clear, executable instruction in plain language (e.g., "send hi to discord channel 1234", "search nytimes for articles about Artificial Intelligence").
         • Only include ONE action; no multi-step plans.
       - If step_type == "THINK":
         • text = a brief reasoning step describing what to figure out next; no tool names or API parameters.
    2. Be specific and build on the latest Observation if present. Do not repeat earlier steps verbatim.
    3. Error recovery policy:
       • If the latest lines include "OBSERVATION: ERROR:" (e.g., ToolExecutionError, Unauthorized, 5xx), do NOT output STOP on the first failure.
       • Prefer step_type == "ACT" with a different approach/tool, or step_type == "THINK" with a brief recovery plan.
       • Avoid selecting the same tool id as the most recent ACT_EXECUTED if it failed.
       • Only STOP after multiple distinct failed ACT attempts or when the goal is clearly impossible from available context.
    4. Output ONLY the JSON object. No markdown, no commentary.
    </instructions>

    <output_format>
    {{"step_type": "THINK|ACT|STOP", "text": "..."}}
    </output_format>
    """
).strip()

_TOOL_SELECTION_PROMPT = dedent(
    """
    <role>
    You are an expert orchestrator working within the Agent API ecosystem.
    Your job is to select the best tool to execute a specific plan step, using a list of available tools.
    Each tool may vary in API domain, supported actions, and required parameters.
    You must evaluate each tool's suitability and return the single best matching tool — or the word none if none qualify.

    Your selection will be executed by an agent, so precision and compatibility are critical.
    </role>

    <instructions>
    Analyze the provided step and evaluate all candidate tools. Use the scoring criteria to assess each tool's fitness for executing the step.
    Return the tool id with the highest total score. If no tool scores ≥60, return the word none.
    You are selecting the most execution-ready tool, not simply the closest match.
    </instructions>

    <input>
    Step: {step}

    Tools (JSON):
    {tools_json}
    </input>

    <scoring_criteria>
    - Action Compatibility (35 pts): Evaluate how well the tool's primary action matches the step's intent.
    - API Domain Match (30 pts): If the step explicitly mentions a platform, require a direct api_name match; otherwise pick a relevant domain.
    - Parameter Compatibility (20 pts): Required parameters should be present or inferable.
    - Workflow Fit (10 pts): Logical integration into surrounding workflow.
    - Simplicity & Efficiency (5 pts): Prefer direct solutions over unnecessarily complex ones.
    </scoring_criteria>

    <rules>
    1. Score each tool using the weighted criteria above. Max score: 100 points.
    2. Select the tool with the highest total score.
    3. If multiple tools tie for the highest score, choose the first.
    4. If no tool scores at least 60 points, return none.
    5. Do not select the same tool id as the most recent failed attempt if an error was observed.
    6. Output only the selected tool id or none.
    </rules>

    <output_format>
    Respond with a single line that contains exactly the selected tool's id — no quotes or extra text and no extra reasoning.
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
        logger.info("ReACT reasoner started", goal=goal, max_turns=self.max_turns)

        reasoning_trace: List[str] = [f"Goal: {goal}"]
        final_answer: Optional[str] = None
        complete: bool = False

        for _ in range(self.max_turns):
            if complete:
                break

            reasoning_transcript = "\n".join(reasoning_trace)
            step_type, step_text = self._think(reasoning_transcript)
            reasoning_trace.append(f"{step_type}: {step_text}")

            if step_type == "STOP":
                reasoning_trace.append(f"FINAL ANSWER: {step_text}")
                complete = True
                logger.info("reasoning_complete", reason="final_thought", turns=len(reasoning_trace))
                break

            if step_type == "ACT":
                try:
                    tool_id, params, observation = self._act(step_text, "\n".join(reasoning_trace))
                    reasoning_trace.append(f"ACT_EXECUTED: tool_id={tool_id}")
                    reasoning_trace.append(f"OBSERVATION: {str(observation)}")
                    obs_preview = str(observation)
                    if len(obs_preview) > 200:
                        obs_preview = obs_preview[:200] + "..."
                    logger.info("tool_executed", tool_id=tool_id, params=params if isinstance(params, dict) else None, observation_preview=obs_preview)
                except ToolCredentialsMissingError as exc:
                    err_tool_id = getattr(getattr(exc, "tool", None), "id", None)
                    suffix = f" tool_id={err_tool_id}" if err_tool_id else ""
                    reasoning_trace.append(f"Tool Unauthorized:{suffix} {str(exc)}")
                    logger.warning("tool_unauthorized", error=str(exc))
                except ToolSelectionError as exc:
                    reasoning_trace.append(f"OBSERVATION: ERROR: ToolSelectionError: {str(exc)}")
                    logger.warning("tool_selection_failed", error=str(exc))
                except ToolExecutionError as exc:
                    err_tool_id = getattr(getattr(exc, "tool", None), "id", None)
                    suffix = f" tool_id={err_tool_id}" if err_tool_id else ""
                    reasoning_trace.append(f"OBSERVATION: ERROR: ToolExecutionError:{suffix} {str(exc)}")
                    logger.error("tool_execution_failed", error=str(exc))
                except Exception as exc:
                    reasoning_trace.append(f"OBSERVATION: ERROR: UnexpectedError: {str(exc)}")
                    logger.error("tool_unexpected_error", error=str(exc), exc_info=True)
            else:
                logger.info("thought_generated", thought=str(step_text)[:200] + ("..." if step_text and len(str(step_text)) > 200 else ""))

        if not complete and not final_answer:
            logger.warning("max_turns_reached", max_turns=self.max_turns, turns=len(reasoning_trace))

        reasoning_transcript = "\n".join(reasoning_trace)
        success = complete
        return ReasoningResult(final_answer=final_answer or "", iterations=len(reasoning_trace), success=success, transcript=reasoning_transcript)

    def _think(self, transcript: str) -> Tuple[str, str]:
        prompt = _THINK_PROMPT.format(transcript=transcript)
        try:
            obj = self.llm.prompt_to_json(prompt, max_retries=0) or {}
            step_type = (obj.get("step_type") or "").strip().upper()
            text = (obj.get("text") or "").strip()
            if step_type in {"THINK", "ACT", "STOP"} and text:
                return step_type.upper(), text
            logger.error("Invalid think output", step_type=step_type, text_present=bool(text))
        except Exception:
            logger.error("think_parse_failed", exc_info=True)
        return "THINK", "Continuing reasoning to determine next step."

    def _act(self, action_text: str, transcript: str) -> Tuple[str, Dict[str, Any], Any]:
        query = action_text
        # Single preferred selection; on failure, let the loop continue and THINK again
        candidates: List[ToolBase] = self.tools.search(query, top_k=self.top_k)
        logger.info("tool_search", query=query, top_k=self.top_k, candidate_count=len(candidates))
        tools_json = "\n".join([t.get_summary() for t in candidates])
        chosen_id = (self.llm.prompt(_TOOL_SELECTION_PROMPT.format(step=query, tools_json=tools_json)) or "").strip()
        if not chosen_id or chosen_id.lower() == "none":
            raise ToolSelectionError(f"No suitable tool selected for step: {query}")
        chosen = next((t for t in candidates if t.id == chosen_id), None)
        if chosen is None:
            raise ToolSelectionError(f"Selected tool id '{chosen_id}' not in candidate list")
        tool = self.tools.load(chosen)
        params = self._generate_params(tool, transcript, query)
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

    def _generate_params(self, tool: ToolBase, transcript: str, step_text: str) -> Dict[str, Any]:
        schema = tool.get_parameters() or {}
        allowed_keys = ",".join(schema.keys()) if isinstance(schema, dict) else ""
        data: Dict[str, Any] = {"reasoning trace": transcript}

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


