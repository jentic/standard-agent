from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from reasoners.models import ReasonerState, Step
from reasoners.sequential.interface import StepExecutor
from tools.interface import Tool
from reasoners.sequential.exceptions import (
    ParameterGenerationError,
    ToolSelectionError,
    ReasoningStepError,
    MissingInputError
)

logger = logging.getLogger(__name__)


### Prompts
STEP_CLASSIFICATION_PROMPT: str = (
    """
    Your task is to classify a step as either 'tool' or 'reasoning'.
    - 'tool': The step requires calling an external API or tool to fetch new information or perform an action in the outside world (e.g., search, send email, post a message).
    - 'reasoning': The step involves processing, filtering, summarizing, or transforming data that is ALREADY AVAILABLE in memory.

    Carefully examine the task and the available data in 'Existing memory keys'.

    STRICT RULES:
    1. If the task can be accomplished using ONLY the data from 'Existing memory keys', you MUST classify it as 'reasoning'.
    2. If the task requires fetching NEW data or interacting with an external system, classify it as 'tool'.
    3. Your reply must be a single word: either "tool" or "reasoning". No other text.

    Task: {step_text}
    Existing memory keys: {keys_list}
    """
)

REASONING_STEP_PROMPT: str = (
    """
    You are an expert data processor. Your task is to perform an internal reasoning step based on the provided data.

    **Current Sub-Task:** {step_text}
    **Available Data (JSON):**
    ```json
    {mem_snippet}
    ```

    **Instructions:**
    1.  Carefully analyze the `Current Sub-Task` and the `Available Data`.
    2.  Execute the task based *only* on the provided data.
    3.  Produce a single, final output.

    **Output Format Rules:**
    -   If the result is structured (e.g., a list or object), you MUST return a single, valid JSON object. Do NOT use markdown code fences or add explanations.
    -   If the result is a simple text answer (e.g., a summary or a single value), return only the raw text.
    -   Do NOT add any commentary, introductory phrases, or conversational text.

    **Final Answer:**
    """
)

TOOL_SELECTION_PROMPT: str = (
    """You are an expert orchestrator. Given the *step* and the *tools* list below,\n"
    "return **only** the `id` of the single best tool to execute the step, or\n"
    "the word `none` if **none of the tools in the provided list are suitable** for the step.\n\n"
    "Step:\n{step}\n\n"
    "Tools (JSON):\n{tools_json}\n\n"
    "Respond with just the id (e.g. `tool_123`) or `none`. Do not include any other text."""
)

PARAMETER_GENERATION_PROMPT = ("""
    "You are Parameter‑Builder AI.\n\n"

    "🛑 OUTPUT FORMAT REQUIREMENT 🛑\n"
    "You must respond with a **single, valid JSON object** only.\n"
    "→ No markdown, no prose, no backticks, no ```json blocks.\n"
    "→ Do not escape newlines (no '\\n' inside strings unless part of real content).\n"
    "→ All values must be properly quoted and valid JSON types.\n\n"

    "ALLOWED_KEYS in the response parameters:\n{allowed_keys}\n\n"

    "STEP:\n{step}\n\n"
    "MEMORY CONTEXT:\n{step_inputs}\n\n"
    "TOOL SCHEMA (JSON):\n{tool_schema}\n\n"

    "RULES:\n"
    "1. Only include keys from ALLOWED_KEYS — do NOT invent new ones.\n"
    "2. Extract values from Step and MEMORY CONTEXT; do not include MEMORY CONTEXT keys themselves.\n"
    "3. If a key's value would be null or undefined, omit it entirely.\n"
    "4. If IDs must be parsed from URLs, extract only the required portion.\n\n"

    "BEFORE YOU RESPOND:\n"
    "✅ Confirm that all keys are in ALLOWED_KEYS\n"
    "✅ Confirm the output starts with '{{' and ends with '}}'\n"
    "✅ Confirm the output is parsable with `JSON.parse()`\n\n"

    "🚨 FINAL RULE: Your reply MUST contain only a single raw JSON object. No explanation. No markdown. No escaping. No backticks."
    "Note: Authentication credentials will be automatically injected by the platform."
    """
)

JSON_CORRECTION_PROMPT: str = (
    """Your previous response was not valid JSON. Please correct it.

    STRICT RULES:
    1.  Your reply MUST be a single, raw, valid JSON object.
    2.  Do NOT include any explanation, markdown, or code fences.
    3.  Do NOT change the data, only fix the syntax.

    Original Prompt:
    ---
    {original_prompt}
    ---

    Faulty JSON Response:
    ---
    {bad_json}
    ---

    Corrected JSON Response:
    """
)

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
        step.status = "running"
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
            step.status = "done"
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
        step.status = "done"
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
        raise ToolSelectionError(f"Invalid tool id '{reply}'")

    # ---------- param generation ---------------------------------------
    def _generate_params(
        self, step: Step, tool_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        schema = self._get_tool(tool_id).parameters or {}
        allowed = ",".join(schema.keys())
        prompt  = PARAMETER_GENERATION_PROMPT.format(
            step=step.text,
            tool_schema=json.dumps(schema, ensure_ascii=False),
            step_inputs=json.dumps(inputs, ensure_ascii=False),
            allowed_keys=allowed,
        )
        raw = self._call_llm(prompt).strip()
        try:
            params = _json_or_retry(raw, prompt, self._call_llm)
            return {k: v for k, v in params.items() if k in schema}
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(
                f"phase=PARAM_GENERATION_FAILED error='{e}' raw_response='{raw}'"
            )
            raise ParameterGenerationError(
                f"Failed to generate valid JSON parameters: {e}"
            ) from e

    # ---------- caching -------------------------------------------------
    def _get_tool(self, tool_id: str) -> Tool:
        if tool_id not in self._tool_cache:
            self._tool_cache[tool_id] = self.tools.load(tool_id)
        return self._tool_cache[tool_id]


# ---------- helper funcs -----------------------------------------------
def _is_valid_tool(reply: str, tools: List[Tool]) -> bool:
    return any(t.id == reply for t in tools)

def _json_or_retry(raw: str, prompt: str, llm_call) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"phase=JSON_DECODE_FAILED, attempting retry raw='{raw}'")
        fixed = llm_call(
            JSON_CORRECTION_PROMPT.format(bad_json=raw, original_prompt=prompt)
        ).strip()
        logger.info(f"phase=JSON_DECODE_RETRY_SUCCESS fixed='{fixed}'")
        return json.loads(fixed)

