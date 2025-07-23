from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from reasoners.models import ReasonerState, Step, StepStatus
from reasoners.sequential.interface import StepExecutor
from tools.interface import Tool
from reasoners.sequential.exceptions import (
    ParameterGenerationError,
    ToolSelectionError,
    ReasoningStepError,
    MissingInputError
)

from utils.logger import get_logger
logger = get_logger(__name__)


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
    """
    <role>
    You are an expert orchestrator working within the Jentic API ecosystem.
    Your job is to select the best tool to execute a specific plan step, using a list of available tools. Each tool may vary in API domain, supported actions, and required parameters. You must evaluate each tool's suitability and return the **single best matching tool** — or the wordnone if none qualify.

    Your selection will be executed by an agent, so precision and compatibility are critical.
    </role>

    <instructions>
    Analyze the provided step and evaluate all candidate tools. Use the scoring criteria to assess each tool’s fitness for executing the step. Return the tool `id` with the highest total score. If no tool scores ≥60, return the word none.
    You are selecting the **most execution-ready** tool, not simply the closest match.
    </instructions>

    <input>
    Step:
    {step}

    Tools (JSON):
    {tools_json}
    </input>

    <scoring_criteria>
    - **API Domain Match** (30 pts): Relevance of the tool’s API domain to the step's intent.
    - **Action Compatibility** (25 pts): How well the tool’s action matches the step’s intent, considering common verb synonyms (e.g., "send" maps well to "post", "create" to "add").
    - **Parameter Compatibility** (20 pts): Whether required parameters are available or can be inferred from the current context.
    - **Workflow Fit** (15 pts): Alignment with the current workflow’s sequence and memory state.
    - **Simplicity & Efficiency** (10 pts): Prefer tools that perform the intended action directly and efficiently; if both an operation and a workflow accomplish the same goal, favor the simpler operation unless the workflow provides a clear added benefit.
    </scoring_criteria>

    <rules>
    1. Score each tool using the weighted criteria above. Max score: 100 points.
    2. Select the tool with the highest total score.
    3. If no tool scores at least 60 points, return none.
    4. Do **not** include any explanation, formatting, or metadata — only the tool `id` or none.
    5. Use available step context and known inputs to inform scoring.
    6. Penalize tools misaligned with the intended action.
    </rules>

    <output_format>
    Respond with a **single line** which only includes the selected tool’s `id`
    **No additional text** should be included.
    </output_format>
    """
)

PARAMETER_GENERATION_PROMPT_1 = ("""
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
    "2. You **may use** any key from ALLOWED_KEYS, **but only when needed**  – omit keys that the current STEP does not require\n"
    "3. Extract values from Step and MEMORY CONTEXT; do not include MEMORY CONTEXT keys themselves.\n"
    "4. If a key's value would be null or undefined, omit it entirely.\n"
    "5. If IDs must be parsed from URLs, extract only the required portion.\n\n"

    "BEFORE YOU RESPOND:\n"
    "✅ Confirm that all keys are in ALLOWED_KEYS\n"
    "✅ Confirm the output starts with '{{' and ends with '}}'\n"
    "✅ Confirm the output is parsable with `JSON.parse()`\n\n"

    "🚨 FINAL RULE: Your reply MUST contain only a single raw JSON object. No explanation. No markdown. No escaping. No backticks."
    "Note: Authentication credentials will be automatically injected by the platform."
    """
)

PARAMETER_GENERATION_PROMPT = (
    """
    <role>
    You are a Parameter Builder within the Jentic agent ecosystem. Your mission is to enable seamless API execution by generating precise parameters from step context and memory data. You specialize in data extraction, content formatting, and parameter mapping to ensure successful tool execution.

    Your core responsibilities:
    - Extract meaningful data from complex memory structures
    - Format content appropriately for target APIs
    - Apply quantity constraints and filtering logic
    - Generate valid parameters that enable successful API calls
    </role>

    <goal>
    Generate precise JSON parameters for the specified API call by extracting relevant data from step context and memory.
    </goal>

    <input>
    STEP: {step}
    MEMORY: {step_inputs}
    SCHEMA: {tool_schema}
    ALLOWED_KEYS: {allowed_keys}
    </input>

    <data_extraction_rules>
    • **Articles/News**: Extract title/headline and URL fields, format as "Title: URL\n"
    • **Arrays**: Process each item, combine into formatted string
    • **Nested Objects**: Access properties using dot notation
    • **Quantities**: "a/an/one" = 1, "few" = 3, "several" = 5, numbers = exact
    • **Never use placeholder text** - always extract real data from memory
    </data_extraction_rules>

    <instructions>
    1. Analyze MEMORY for relevant data structures
    2. Extract actual values using the data extraction rules
    3. Format content appropriately for the target API
    4. Apply quantity constraints from step language
    5. Generate valid parameters using only ALLOWED_KEYS
    </instructions>

    <constraints>
    - Output ONLY valid JSON - no markdown, explanations, or backticks
    - Use only keys from ALLOWED_KEYS
    - Extract actual data values from MEMORY, never placeholder text
    - For messaging APIs: format as readable text with titles and links
    - Required parameters take priority over optional ones
    </constraints>

    <output_format>
    Valid JSON object starting with {{ and ending with }} and confirm the output is parsable with `JSON.parse()`
    </output_format>
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
        step.status = StepStatus.RUNNING
        inputs = self._fetch_inputs(step)

        step_type = self._classify_step(step)
        logger.info(f"phase=STEP_CLASSIFIED step_type={step_type}")

        if step_type == Step.StepType.REASONING:
            self._do_reasoning(step, inputs, state)
        else:
            self._do_tool(step, inputs, state)

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

            params = _json_or_retry(raw, prompt, self._call_llm)
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

