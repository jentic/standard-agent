from __future__ import annotations

import json, re, logging
from typing import Any, Dict, List

from jentic_agents.reasoners.models import ReasonerState, Step
from jentic_agents.reasoners.sequential.interface import StepExecutor
from jentic_agents.tools.interface import Tool, ToolInterface

### Exceptions
class MissingInputError(Exception):
    """Raised when a required input key is not found in memory."""


class ToolSelectionError(Exception):
    """Raised when a suitable tool cannot be selected."""


class ParameterGenerationError(Exception):
    """Raised when parameters for a tool cannot be generated."""


class ReasoningStepError(Exception):
    """Raised when a reasoning-only step fails."""

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
log = logging.getLogger("ReWOOStepExecutor")


class ReWOOStepExecutor(StepExecutor):
    """Executes ReWOO steps (plan-first + reflection)."""

    def __init__(self) -> None:
        self._tool_cache: Dict[str, Tool] = {}

    # ---------- public --------------------------------------------------
    def execute(self, step: Step, state: ReasonerState) -> Dict[str, Any] | None:
        if not (self.llm and self.tools and self.memory):
            raise RuntimeError("Services not attached; call attach_services()")

        step.status = "running"
        inputs = self._fetch_inputs(step)   # may raise MissingInputError

        step.step_type = self._classify_step(step)
        if step.step_type is Step.StepType.REASONING:
            return self._do_reasoning(step, inputs, state)
        else:
            return self._do_tool(step, inputs, state)

    # ---------- step kinds ---------------------------------------------
    def _do_reasoning(
        self, step: Step, inputs: Dict[str, Any], state: ReasonerState
    ) -> None:
        try:
            payload  = json.dumps(inputs, ensure_ascii=False)
            prompt   = REASONING_STEP_PROMPT.format(step_text=step.text, mem_snippet=payload)
            reply    = self._call_llm(prompt).strip()
            if m := _JSON_FENCE_RE.search(reply):
                reply = m.group(1).strip()
            step.result = reply
            step.status = "done"
            self._store_output(step, state)
        except Exception as exc:
            raise ReasoningStepError(str(exc)) from exc

    def _do_tool(
        self, step: Step, inputs: Dict[str, Any], state: ReasonerState
    ) -> Dict[str, Any]:
        tool_id = self._select_tool(step)
        if tool_id == "none":
            raise ToolSelectionError("No suitable tool found")

        params   = self._generate_params(step, tool_id, inputs)
        result   = self.tools.execute(tool_id, params)
        payload  = result["result"].output if isinstance(result, dict) else result

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
        inputs = {}
        for key in step.input_keys:
            try:
                inputs[key] = self.memory.retrieve(key)
            except Exception as _:
                raise MissingInputError(key)
        return inputs

    def _store_output(self, step: Step, state: ReasonerState) -> None:
        if step.output_key and step.result is not None:
            self.memory.store(step.output_key, step.result)
            state.history.append(f"stored {step.output_key}")

    # ---------- LLM wrapper --------------------------------------------
    def _call_llm(self, prompt: str, **kw) -> str:
        log.debug("LLM prompt %.80s…", prompt.replace("\n", " ")[:80])
        return self.llm.chat([{"role": "user", "content": prompt}], **kw)

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
        params = _json_or_retry(raw, prompt, self._call_llm)
        return {k: v for k, v in params.items() if k in schema}

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
        fixed = llm_call(JSON_CORRECTION_PROMPT.format(bad_json=raw, original_prompt=prompt)).strip()
        return json.loads(fixed)

