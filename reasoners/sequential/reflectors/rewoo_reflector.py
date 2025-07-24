from __future__ import annotations
import json, re
from copy import deepcopy
from typing import Dict, Any
from pydantic import BaseModel

from reasoners.models import ReasonerState, Step, StepStatus
from reasoners.sequential.interface import Reflector
from tools.exceptions import ToolExecutionError
from reasoners.sequential.exceptions import (
    ParameterGenerationError,
    ToolSelectionError,
)

from utils.logger import get_logger
logger = get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")

### Prompts ###
BASE_REFLECTION_PROMPT: str = (
    """You are a self-healing reasoning engine. A step in your plan failed. Your task is to analyze the error and propose a single, precise fix.

🛑 **OUTPUT FORMAT REQUIREMENT** 🛑
Your reply MUST be a single, raw, valid JSON object. No explanation, no markdown, no backticks.
Your reply MUST start with '{{' and end with '}}' - nothing else.

**JSON Schema (for reference only, do NOT include this block in your reply)**
{{
  "reasoning": "A brief explanation of why the step failed.",
  "action": "one of 'retry_params', 'change_tool', 'rephrase_step', or 'give_up'",
  "tool_id": "(Required if action is 'change_tool') The ID of the new tool to use.",
  "params": "(Required if action is 'retry_params' or 'change_tool') A valid JSON object of parameters for the tool.",
  "step": "(Required if action is 'rephrase_step') The new, improved text for the step."
}}


**Example of a valid response (for reference only):**
{{
  "reasoning": "The error indicates a required parameter 'channel_id' was missing, which can be extracted from the goal.",
  "action": "retry_params",
  "params": {{
    "channel_id": "#general",
    "content": "Welcome!"
  }}
}}


---

✅ BEFORE YOU RESPOND, SILENTLY SELF-CHECK:
1. Does your reply start with '{{' and end with '}}'?
2. Is your reply valid JSON parsable by `JSON.parse()`?
3. Are all required keys present and correctly typed?
4. Have you removed ALL markdown, code fences, and explanatory text?
   - If any check fails, REGENERATE your answer.

---

**Your Turn: Real Context**

**Goal:**
{goal}

**Failed Step:**
{step}

**Failed Tool:**
{failed_tool_id}

**Error:**
{error_type}: {error_message}

**Tool Schema (if available):**
{tool_schema}
"""
)

ALTERNATIVE_TOOLS_SECTION: str = (
    """
    **Alternative Tools:**
    The previous tool failed. Please select a more suitable tool from the following list to achieve the step's goal.
    {alternative_tools}
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

class ReWOOReflector(Reflector):
    """Retry-oriented reflection for ReWOO executors."""

    def __init__(self, *, max_retries: int = 2):
        self.max_retries = max_retries
        logger.info("phase=REWOO_REFLECTOR_INITIALIZED max_retries=%d", self.max_retries)

    # ------------------------------- public ------------------------------
    def handle(
        self,
        error: Exception,
        step: Step,
        state: ReasonerState,
        *,
        failed_tool_id: str | None = None,
    ) -> None:
        logger.info(
            "phase=HANDLE_ERROR error='%s' step_text='%s'",
            error.__class__.__name__,
            step.text,
        )
        if not (self.llm and self.tools and self.memory):
            raise RuntimeError("attach_services() not called on Reflector")

        step.status, step.error = StepStatus.FAILED, str(error)

        if step.retry_count >= self.max_retries:
            logger.warning(
                "phase=GIVE_UP reason='max_retries_exceeded' step_text='%s'", step.text
            )
            state.history.append(
                f"Give-up after {self.max_retries} retries: {step.text}"
            )
            return

        failed_tool_id = failed_tool_id or getattr(error, "tool_id", "unknown")
        tool_schema    = self._tool_schema(failed_tool_id)

        if isinstance(tool_schema, BaseModel):
            tool_schema_json = tool_schema.model_dump_json(indent=2)
        else:
            tool_schema_json = json.dumps(tool_schema, indent=2)

        prompt = BASE_REFLECTION_PROMPT.format(
            goal           = state.goal,
            step           = step.text,
            failed_tool_id = failed_tool_id,
            error_type     = error.__class__.__name__,
            error_message  = str(error),
            tool_schema    = tool_schema_json,
        )

        if isinstance(error, (ToolExecutionError, ToolSelectionError, ParameterGenerationError)):
            prompt = self._add_alternatives(prompt, step, failed_tool_id)

        raw = self._call_llm(prompt)
        raw = _FENCE_RE.sub(lambda m: m.group(1).strip(), raw)
        decision = self._json_or_retry(raw, prompt)
        logger.info("phase=REFLECTION_DECISION decision=%s", decision)

        self._apply_decision(decision, step, state)

    # ------------------------------ helpers ------------------------------
    def _tool_schema(self, tool_id: str) -> Dict[str, Any]:
        logger.debug("phase=LOAD_TOOL_SCHEMA tool_id='%s'", tool_id)
        if tool_id and tool_id not in ("unknown", "none"):
            try:
                schema = self.tools.load(tool_id) or {}
                logger.debug("phase=LOAD_TOOL_SCHEMA_SUCCESS")
                return schema
            except Exception as e:
                logger.warning("phase=LOAD_TOOL_SCHEMA_FAILED tool_id=%s error=%s", tool_id, e)
        return {}

    def _add_alternatives(self, prompt: str, step: Step, failed_tool_id: str) -> str:
        logger.debug("phase=SEARCH_ALTERNATIVE_TOOLS step_text='%s'", step.text)
        try:
            alt = [
                t
                for t in self.tools.search(step.text, top_k=25)
                if t.id != failed_tool_id
            ]
            if alt:
                logger.debug("phase=ALTERNATIVE_TOOLS_FOUND count=%d", len(alt))
                lite = [
                    {
                        "id": t.id,
                        "name": t.name,
                        "description": t.description,
                        "api_name": t.api_name,
                    }
                    for t in alt
                ]
                return prompt + ALTERNATIVE_TOOLS_SECTION.format(
                    alternative_tools=json.dumps(lite, indent=2)
                )
            else:
                logger.debug("phase=NO_ALTERNATIVE_TOOLS_FOUND")
        except Exception as exc:
            logger.warning("phase=ALTERNATIVE_TOOL_SEARCH_FAILED error='%s'", exc)
        return prompt

    def _apply_decision(self, d: Dict[str, Any], step: Step, state: ReasonerState):
        action = d.get("action")
        logger.info("phase=APPLY_DECISION action='%s'", action)
        state.history.append(f"Reflection decision: {d}")

        if action == "give_up":
            return

        new_step = deepcopy(step)
        new_step.retry_count += 1
        new_step.status = StepStatus.PENDING

        if action == "rephrase_step":
            new_step.text = str(d.get("step", new_step.text))

        elif action == "change_tool":
            self.memory.store(f"forced_tool:{new_step.text}", d.get("tool_id"))

        elif action == "retry_params":
            self.memory.store(f"forced_params:{new_step.text}", d.get("params"))

        state.plan.appendleft(new_step)
        logger.debug("phase=APPLY_DECISION_SUCCESS new_step='%s'", new_step.text)

    # ----------------------------- llm/json ------------------------------
    def _call_llm(self, prompt: str) -> str:
        logger.debug("phase=LLM_CALL prompt='%.100s...'", prompt)
        response = self.llm.chat([{"role": "user", "content": prompt}]).strip()
        logger.debug("phase=LLM_RESPONSE response='%.100s...'", response)
        return response

    def _json_or_retry(self, raw: str, prompt: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("phase=JSON_DECODE_FAILED, attempting retry raw='%.100s...'", raw)
            fixed = self._call_llm(
                JSON_CORRECTION_PROMPT.format(bad_json=raw, original_prompt=prompt)
            )
            logger.info("phase=JSON_DECODE_RETRY_SUCCESS fixed='%.100s...'", fixed)
            return json.loads(fixed)