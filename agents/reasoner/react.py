from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
from collections.abc import MutableMapping

from agents.reasoner.base import BaseReasoner, ReasoningResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase, ToolBase
from agents.tools.exceptions import ToolExecutionError, ToolCredentialsMissingError
from agents.reasoner.exceptions import ToolSelectionError

from utils.logger import get_logger
logger = get_logger(__name__)

from agents.prompts import load_prompts
_PROMPTS = load_prompts("reasoners/react", required_prompts=["think", "tool_select", "param_gen"])

class ReACTReasoner(BaseReasoner):
    DEFAULT_MAX_TURNS = 20

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        max_turns: int = DEFAULT_MAX_TURNS,
        top_k: int = 25,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.max_turns = max_turns
        self.top_k = top_k

    def run(self, goal: str) -> ReasoningResult:
        logger.info("ReACT reasoner started", goal=goal, max_turns=self.max_turns)

        reasoning_trace: List[str] = [f"Goal: {goal}"]
        complete: bool = False
        failed_tool_ids: List[str] = []
        tool_calls: List[dict] = []
        turns: int = 0

        for _ in range(self.max_turns):
            if complete:
                break

            step_type, step_text = self._think("\n".join(reasoning_trace))
            reasoning_trace.append(f"{step_type}: {step_text}")
            turns += 1

            if step_type == "STOP":
                reasoning_trace.append(f"FINAL ANSWER: {step_text}")
                complete = True
                logger.info("reasoning_complete", reason="final_thought", turns=turns)
                break

            if step_type == "ACT":
                try:
                    tool, params, observation = self._act(step_text, "\n".join(reasoning_trace), failed_tool_ids)
                    reasoning_trace.append(f"ACT_EXECUTED: tool={tool.get_summary()}")
                    reasoning_trace.append(f"OBSERVATION: {str(observation)}")
                    tool_calls.append({"tool_id": tool.id, "summary": tool.get_summary()})
                    logger.info("tool_executed", tool_id=tool.id, params=params if isinstance(params, dict) else None, observation_preview=str(observation)[:200] + "..." if len(str(observation)) > 200 else observation)
                except ToolCredentialsMissingError as exc:
                    tid = getattr(getattr(exc, "tool", None), "id", None)
                    if tid: failed_tool_ids.append(tid)
                    reasoning_trace.append(f"Tool Unauthorized:{f' tool_id={tid}' if tid else ''} {exc}")
                    logger.warning("tool_unauthorized", error=str(exc))
                except ToolSelectionError as exc:
                    reasoning_trace.append(f"OBSERVATION: ERROR: ToolSelectionError: {str(exc)}")
                    logger.warning("tool_selection_failed", error=str(exc))
                except ToolExecutionError as exc:
                    tid = getattr(getattr(exc, "tool", None), "id", None)
                    if tid: failed_tool_ids.append(tid)
                    reasoning_trace.append(f"OBSERVATION: ERROR: ToolExecutionError:{f' tool_id={tid}' if tid else ''} {exc}")
                    logger.error("tool_execution_failed", error=str(exc))
                except Exception as exc:
                    reasoning_trace.append(f"OBSERVATION: ERROR: UnexpectedError: {str(exc)}")
                    logger.error("tool_unexpected_error", error=str(exc), exc_info=True)
            else:
                logger.info("thought_generated", thought=step_text)

        if not complete:
            logger.warning("max_turns_reached", max_turns=self.max_turns, turns=turns)

        reasoning_transcript = "\n".join(reasoning_trace)
        success = complete
        return ReasoningResult(iterations=turns, success=success, transcript=reasoning_transcript, tool_calls=tool_calls)

    def _think(self, transcript: str) -> Tuple[str, str]:
        VALID_STEP_TYPES = {"THINK", "ACT", "STOP"}
        try:
            think_response = self.llm.prompt_to_json(_PROMPTS["think"].format(transcript=transcript), max_retries=0)
            step_type = think_response.get("step_type").strip().upper()
            text = think_response.get("text").strip()
            if step_type in VALID_STEP_TYPES and text:
                return step_type, text
            logger.error("think_invalid_output", step_type=step_type, text_present=bool(text))
        except Exception as e:
            logger.error("think_parse_failed", error=str(e), exc_info=True)
        return "THINK", "Continuing reasoning to determine next step."

    def _act(self, action_text: str, transcript: str, failed_tool_ids: List[str]) -> Tuple[ToolBase, Dict[str, Any], Any]:
        tool = self._select_tool(action_text, failed_tool_ids)
        params = self._generate_params(tool, transcript, action_text)
        observation = self.tools.execute(tool, params)
        return tool, params, observation

    def _select_tool(self, action_text: str, failed_tool_ids: List[str]) -> ToolBase:
        tool_candidates = [t for t in self.tools.search(action_text, top_k=self.top_k) if t.id not in set(failed_tool_ids)]
        logger.info("tool_search", query=action_text, top_k=self.top_k, candidate_count=len(tool_candidates))

        tools_json = "\n".join(t.get_summary() for t in tool_candidates)
        prompt = _PROMPTS["tool_select"].format(step=action_text, tools_json=tools_json)
        if failed_tool_ids:
            failed_block = "\n".join(f"- {tid}" for tid in failed_tool_ids[-3:])
            prompt += f"\n\n<failed_tools>\n{failed_block}\n</failed_tools>\n"
        selected_tool_id = self.llm.prompt(prompt).strip()

        if not selected_tool_id or selected_tool_id.lower() == "none":
            raise ToolSelectionError(f"No suitable tool selected for step: {action_text}")

        selected_tool = next((t for t in tool_candidates if t.id == selected_tool_id), None)
        if selected_tool is None:
            raise ToolSelectionError(f"Selected tool id '{selected_tool_id}' not in candidate list")

        return self.tools.load(selected_tool)

    def _generate_params(self, tool: ToolBase, transcript: str, step_text: str) -> Dict[str, Any]:
        schema = tool.get_parameters() or {}
        allowed_keys = ",".join(schema.keys()) if isinstance(schema, dict) else ""
        data: Dict[str, Any] = {"reasoning trace": transcript}

        params_raw = self.llm.prompt_to_json(
            _PROMPTS["param_gen"].format(
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


