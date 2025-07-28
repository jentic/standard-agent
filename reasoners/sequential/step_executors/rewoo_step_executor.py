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

_JSON_FENCE_RE   = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")

# Prompts
STEP_CLASSIFICATION_PROMPT: str = (
    """
    <role>
    You are a Step Classifier within the Jentic agent ecosystem. Your sole purpose is to determine whether a given step requires external API/tool execution or can be completed through internal reasoning alone.
    </role>

    <goal>
    Classify the provided step as either TOOL or REASONING based on whether it requires external API calls.
    </goal>

    <input>
    Step: {step_text}
    Available Memory Keys: {keys_list}
    </input>

    <classification_rules>
    TOOL steps require:
    - External API calls (e.g., "search articles", "send email", "create task")
    - Third-party service interactions
    - Data retrieval from external sources

    REASONING steps include:
    - Data transformation or formatting
    - Summarization or analysis of existing data
    - Logic operations using available memory
    - Internal calculations or processing
    </classification_rules>

    <output_format>
    Respond with ONLY one word: either "TOOL" or "REASONING"
    </output_format>
    """
)

TOOL_SELECTION_PROMPT = (
   """
   <role>
   You are an expert orchestrator working within the Jentic API ecosystem.
   Your job is to select the best tool to execute a specific plan step, using a list of available tools. Each tool may vary in API domain, supported actions, and required parameters. You must evaluate each tool's suitability and return the **single best matching tool** — or the word none if none qualify.

   Your selection will be executed by an agent, so precision and compatibility are critical.
   </role>

   <instructions>
   Analyze the provided step and evaluate all candidate tools. Use the scoring criteria to assess each tool's fitness for executing the step. Return the tool `id` with the highest total score. If no tool scores ≥60, return the word none.
   You are selecting the **most execution-ready** tool, not simply the closest match.
   </instructions>

   <input>
   Step:
   {step}

   Tools (JSON):
   {tools_json}
   </input>

   <scoring_criteria>
   - **Action Compatibility** (35 pts): Evaluate how well the tool's primary action matches the step's intent. Consider synonyms (e.g., "send" ≈ "post", "create" ≈ "add"), but prioritize tools that closely reflect the intended verb-object structure and scope. Penalize mismatches in type, scope, or intent (e.g., "get all members" for "get new members").

   - **API Domain Match** (30 pts): This is a critical criterion.
       - **If the step EXPLICITLY mentions a specific platform or system (e.g., "Gmail", "Asana", "Microsoft Teams")**:
           - **Perfect Match (30 pts):** If the tool's `api_name` directly matches the explicitly mentioned platform.
           - **Severe Penalty (0 pts):** If the tool's `api_name` does *not* match the explicitly mentioned platform. Do NOT select tools from other domains in this scenario.
       - **If NO specific platform or system is EXPLICITLY mentioned (e.g., "book a flight", "send an email")**:
           - **Relevant Match (25-30 pts):** If the tool's `api_name` is generally relevant to the task (e.g., a flight booking tool for "book a flight"). Prefer tools with broader applicability if multiple options exist.
           - **Irrelevant Match (0-10 pts):** If the tool's `api_name` is clearly irrelevant.

   - **Parameter Compatibility** (20 pts): Determine if the tool's required parameters are explicitly present in the step or clearly inferable. Penalize tools with ambiguous, unsupported, or overly strict input requirements.

   - **Workflow Fit** (10 pts): Assess how logically the tool integrates into the surrounding workflow. Does it build upon prior steps or prepare outputs needed for future ones?

   - **Simplicity & Efficiency** (5 pts): Prefer tools that accomplish the task directly and without unnecessary complexity. Penalize overly complex workflows if a simpler operation would suffice. This includes preferring a single-purpose tool over a multi-purpose tool if the single-purpose tool directly addresses the step's need (e.g., "Get a user" over "Get multiple users" if only one user is needed).
   </scoring_criteria>

   <rules>
   1. Score each tool using the weighted criteria above. Max score: 100 points.
   2. Select the tool with the highest total score.
   3. If no tool scores at least 60 points, return none.
   4. Do **not** include any explanation, formatting, or metadata — only the tool `id` or none.
   5. Use available step context and known inputs to inform scoring.
   6. Penalize tools severely if they are misaligned with the intended action or platform (if mentioned in the step).
   7. Never select a tool from an incorrect domain if the step explicitly specifies a specific one.
   </rules>

   <output_format>
   Respond with a **single line** which only includes the selected tool's `id`
   **No additional text** should be included.
   </output_format>
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
    • **Array Slicing**: When processing arrays from memory, look for quantity constraints in the STEP text and slice accordingly
    • **Never use placeholder text** - always extract real data from memory
    </data_extraction_rules>

    <instructions>
    1. Analyze MEMORY for relevant data structures
    2. Extract actual values using the data extraction rules
    3. **CRITICAL**: Check STEP text for quantity constraints (e.g., "send 3 articles", "post 2 items")
    4. If processing arrays from memory and STEP has quantity constraint, slice array to that size
    5. Format content appropriately for the target API
    6. Generate valid parameters using only ALLOWED_KEYS
    7. **CRITICAL**: Only use parameters that are explicitly documented in the SCHEMA - do not infer or add undocumented parameters
    </instructions>

    <constraints>
    - Output ONLY valid JSON - no markdown, explanations, or backticks
    - Use only keys from ALLOWED_KEYS
    - Extract actual data values from MEMORY, never placeholder text
    - For messaging APIs: format as readable text with titles and links
    - Required parameters take priority over optional ones
    </constraints>

    <output_format>
    Valid JSON object starting with {{ and ending with }}
    </output_format>
    """
)


REASONING_STEP_PROMPT: str = (
    """
    <role>
    You are a Data Processor within the Jentic agent ecosystem. Your mission is to perform precise data transformations and reasoning operations on available information. You specialize in content analysis, data extraction, and logical processing to support agent workflows.

    Your core responsibilities:
    - Process data using only available information
    - Perform logical reasoning and analysis tasks
    - Transform data into required formats
    - Generate accurate, context-appropriate outputs
    </role>

    <goal>
    Execute the specified sub-task using only the provided data to produce a single, accurate output.
    </goal>

    <input>
    Sub-Task: {step_text}
    Available Data: {mem_snippet}
    </input>

    <instructions>
    1. Analyze the sub-task and available data carefully
    2. Execute the task using ONLY the provided data
    3. Produce a single, final output based on the task requirements
    4. Do not add commentary, explanations, or conversational text
    </instructions>

    <output_format>
    - For structured results (lists, objects): Valid JSON object without code fences
    - For simple text results (summaries, values): Raw text only
    - No introductory phrases or explanations
    </output_format>
    """
)


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

