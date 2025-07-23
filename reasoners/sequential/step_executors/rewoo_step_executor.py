from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from reasoners.models import ReasonerState, Step, StepStatus
from reasoners.sequential.interface import StepExecutor
from reasoners.prompts import (
    STEP_CLASSIFICATION_PROMPT,
    REASONING_STEP_PROMPT,
    TOOL_SELECTION_PROMPT,
    PARAMETER_GENERATION_PROMPT
)
from tools.interface import Tool
from reasoners.sequential.exceptions import (
    ParameterGenerationError,
    ToolSelectionError,
    ReasoningStepError,
    MissingInputError
)

from utils.logger import get_logger
logger = get_logger(__name__)

_JSON_FENCE_RE   = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")


class ReWOOStepExecutor(StepExecutor):
    """Executes ReWOO steps (plan-first + reflection)."""

    def __init__(self) -> None:
        self._tool_cache: Dict[str, Tool] = {}
        logger.info("phase=REWOO_STEP_EXECUTOR_INITIALIZED")

    # ---------- public --------------------------------------------------
    def execute(self, step: Step, state: ReasonerState) -> Dict[str, Any] | None:
        if not (self.llm and self.tools and self.memory):
            raise RuntimeError("Services not attached; call attach_services()")
        logger.info(f"phase=EXECUTE_STEP step_text='{step.text}'")
        step.status = StepStatus.RUNNING
        inputs = self._fetch_inputs(step)

        step_type = self._classify_step(step)
        logger.info(f"phase=STEP_CLASSIFIED step_type={step_type}")

        if step_type == Step.StepType.REASONING:
            self._do_reasoning(step, inputs, state)
        else:
            self._do_tool(step, inputs, state)
        logger.info(f"phase=EXECUTE_STEP_COMPLETE result='{step.result}'")

    # ---------- step kinds ---------------------------------------------
    def _do_reasoning(
        self, step: Step, inputs: Dict[str, Any], state: ReasonerState
    ):
        logger.debug("phase=REASONING_STEP_START")
        try:
            mem_snippet = json.dumps(inputs, ensure_ascii=False)
            prompt = REASONING_STEP_PROMPT.format(
                step_text=step.text, mem_snippet=mem_snippet
            )
            reply = self._call_llm(prompt)

            step.result = reply
            step.status = StepStatus.DONE
            self._store_output(step, state)
            logger.info("phase=REASONING_STEP_SUCCESS")
        except Exception as exc:
            logger.error(f"phase=REASONING_STEP_FAILED error='{exc}'")
            raise ReasoningStepError(str(exc)) from exc

    def _do_tool(
        self, step: Step, inputs: Dict[str, Any], state: ReasonerState
    ) -> Dict[str, Any]:
        logger.debug("phase=TOOL_STEP_START")
        tool_id = self._select_tool(step)
        if tool_id == "none":
            logger.error("phase=TOOL_SELECTION_FAILED reason='No suitable tool found'")
            raise ToolSelectionError("No suitable tool found")
        logger.info(f"phase=TOOL_SELECTED tool_id='{tool_id}'")

        params = self._generate_params(step, tool_id, inputs)
        logger.debug(f"phase=PARAMS_GENERATED params='{params}'")

        result = self.tools.execute(tool_id, params)
        payload = result["result"].output if isinstance(result, dict) else result
        logger.info(f"phase=TOOL_EXECUTED result='{payload}'")

        step.result = payload
        step.status = StepStatus.DONE
        self._store_output(step, state)

        return {"tool_id": tool_id, "params": params, "result": payload}

    # ---------- helpers -------------------------------------------------
    def _classify_step(self, step: Step) -> Step.StepType:
        keys_list = ", ".join(getattr(self.memory, "keys", lambda: [])())
        prompt    = STEP_CLASSIFICATION_PROMPT.format(step_text=step.text, keys_list=keys_list)
        reply     = self._call_llm(prompt).strip().lower()
        return Step.StepType.TOOL if reply.startswith("tool") else Step.StepType.REASONING

    def _fetch_inputs(self, step: Step) -> Dict[str, Any]:
        logger.debug(f"phase=FETCH_INPUTS keys='{step.input_keys}'")
        inputs = {}
        for key in step.input_keys:
            try:
                inputs[key] = self.memory.retrieve(key)
            except Exception as _:
                logger.error(f"phase=FETCH_INPUTS_FAILED reason='Missing key' key='{key}'")
                raise MissingInputError(key)
        logger.debug(f"phase=FETCH_INPUTS_SUCCESS inputs='{inputs}'")
        return inputs

    def _store_output(self, step: Step, state: ReasonerState) -> None:
        if step.output_key and step.result is not None:
            logger.debug(
                f"phase=STORE_OUTPUT key='{step.output_key}' value='{step.result}'"
            )
            self.memory.store(step.output_key, step.result)
            state.history.append(f"stored {step.output_key} : {step.result}")

    # ---------- LLM wrapper --------------------------------------------
    def _call_llm(self, prompt: str, **kw) -> str:
        logger.debug(f"phase=LLM_CALL prompt='{prompt[:100]}...'")
        response = self.llm.chat([{"role": "user", "content": prompt}], **kw)
        logger.debug(f"phase=LLM_RESPONSE response='{response[:100]}...'")
        return response

    # ---------- tool selection -----------------------------------------
    def _select_tool(self, step: Step) -> str:
        tools = self.tools.search(step.text, top_k=20)
        prompt = TOOL_SELECTION_PROMPT.format(
            step=step.text,
            tools_json=json.dumps([t.dict() for t in tools], ensure_ascii=False),
        )
        reply = self._call_llm(prompt).strip()
        if reply == "none" or _is_valid_tool(reply, tools):
            return reply
        logger.error(
            f"phase=TOOL_SELECTION_FAILED reason='Invalid tool id' tool_id='{reply}'"
        )
        raise ToolSelectionError(f"Invalid tool id '{reply}'", tool_id=reply)

    # ---------- param generation ---------------------------------------
    def _generate_params(
        self, step: Step, tool_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        raw = ""
        try:
            schema = self._get_tool(tool_id).parameters or {}
            allowed = ",".join(schema.keys())
            prompt  = PARAMETER_GENERATION_PROMPT.format(
                step=step.text,
                tool_schema=json.dumps(schema, ensure_ascii=False),
                step_inputs=json.dumps(inputs, ensure_ascii=False),
                allowed_keys=allowed,
            )
            raw = self._call_llm(prompt).strip()

            params = _json_or_retry(raw, prompt)
            return {k: v for k, v in params.items() if k in schema}
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(
                f"phase=PARAM_GENERATION_FAILED error='{e}' raw_response='{raw}'"
            )
            raise ParameterGenerationError(
                tool_id=tool_id, message=f"Failed to generate valid JSON parameters: {e}"
            ) from e

    # ---------- caching -------------------------------------------------
    def _get_tool(self, tool_id: str) -> Tool:
        if tool_id not in self._tool_cache:
            self._tool_cache[tool_id] = self.tools.load(tool_id)
        return self._tool_cache[tool_id]


# ---------- helper funcs -----------------------------------------------
def _is_valid_tool(reply: str, tools: List[Tool]) -> bool:
    return any(t.id == reply for t in tools)

def _json_or_retry(raw: str, prompt: str) -> Dict[str, Any]:
    """Parse JSON with fallback for markdown-wrapped responses."""
    # First try raw parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([^`]+)\s*```")
    match = _JSON_FENCE_RE.search(raw)
    if match:
        extracted = match.group(1).strip()
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    
    # Log failure and raise error
    logger.warning(f"phase=JSON_PARSE_FAIL raw='{raw}'")
    raise ParameterGenerationError(f"LLM returned invalid JSON: {raw}")

