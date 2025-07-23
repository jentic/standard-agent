from __future__ import annotations
import json, re
from copy import deepcopy
from typing import Dict, Any

from reasoners.models import ReasonerState, Step, StepStatus
from reasoners.sequential.interface import Reflector
from reasoners.prompts import BASE_REFLECTION_PROMPT, ALTERNATIVE_TOOLS_SECTION
from tools.exceptions import ToolExecutionError
from reasoners.sequential.exceptions import (
    ParameterGenerationError,
    ToolSelectionError,
)

from utils.logger import get_logger
logger = get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")

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

        prompt = BASE_REFLECTION_PROMPT.format(
            goal           = state.goal,
            step           = step.text,
            failed_tool_id = failed_tool_id,
            error_type     = error.__class__.__name__,
            error_message  = str(error),
            tool_schema    = json.dumps(tool_schema),
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
                logger.warning(f"phase=LOAD_TOOL_SCHEMA_FAILED tool_id={tool_id} error={e}", tool_id)
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
        logger.warning("phase=JSON_PARSE_FAIL raw='%s'", raw)
        raise ParameterGenerationError(f"LLM returned invalid JSON: {raw}")