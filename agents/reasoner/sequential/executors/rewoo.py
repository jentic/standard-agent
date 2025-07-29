from __future__ import annotations

import json
from textwrap import dedent

from agents.reasoner.sequential.reasoner import StepStatus
from agents.reasoner.sequential.executors.base import ExecuteStep
from agents.reasoner.sequential.exceptions import (
    ParameterGenerationError,
    ToolSelectionError,
    MissingInputError
)

from utils.logger import get_logger, trace_method
logger = get_logger(__name__)


### Prompts
STEP_CLASSIFICATION_PROMPT = dedent("""
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
""")

REASONING_STEP_PROMPT = dedent("""
    You are an expert data processor. Your task is to perform an internal reasoning step based on the provided data.

    **Current Sub-Task:** {step_text}
    **Available Data (JSON):**
    ```json
    {available_data}
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
""")

TOOL_SELECTION_PROMPT = dedent("""
    You are an expert orchestrator. Given the *step* and the *tools* list below,
    return **only** the `id` of the single best tool to execute the step, or
    the word `none` if **none of the tools in the provided list are suitable** for the step.

    Step:
    {step}

    Tools (JSON):
    {tools_json}

    Respond with just the id (e.g. `tool_123`) or `none`. Do not include any other text.
""")

PARAMETER_GENERATION_PROMPT = dedent("""
    You are Parameter‑Builder AI.

    🛑 OUTPUT FORMAT REQUIREMENT 🛑
    You must respond with a **single, valid JSON object** only.
    → No markdown, no prose, no backticks, no ```json blocks.
    → Do not escape newlines (no '\n' inside strings unless part of real content).
    → All values must be properly quoted and valid JSON types.

    ALLOWED_KEYS in the response parameters:
    {allowed_keys}

    STEP:
    {step}

    MEMORY CONTEXT:
    {step_inputs}

    TOOL SCHEMA (JSON):
    {tool_schema}

    RULES:
    1. Only include keys from ALLOWED_KEYS — do NOT invent new ones.
    2. You **may use** any key from ALLOWED_KEYS, **but only when needed**  – omit keys that the current STEP does not require
    3. Extract values from Step and MEMORY CONTEXT; do not include MEMORY CONTEXT keys themselves.
    4. If a key's value would be null or undefined, omit it entirely.
    5. If IDs must be parsed from URLs, extract only the required portion.

    BEFORE YOU RESPOND:
    ✅ Confirm that all keys are in ALLOWED_KEYS
    ✅ Confirm the output starts with '{{' and ends with '}}'
    ✅ Confirm the output is parsable with `JSON.parse()`

    🚨 FINAL RULE: Your reply MUST contain only a single raw JSON object. No explanation. No markdown. No escaping. No backticks.
    Note: Authentication credentials will be automatically injected by the platform.
""")



class ReWOOExecuteStep(ExecuteStep):
    """Executes ReWOO steps (plan-first + reflection)."""

    @trace_method
    def __call__(self, step, state):
        step.status = StepStatus.RUNNING

        try:
            # Gather inputs from memory
            inputs = {key: self.memory[key] for key in step.input_keys}

        except KeyError as e:
            missing_key = e.args[0]
            raise MissingInputError(f"Required memory key '{missing_key}' not found for step: {step.text}") from e

        # Classify plan step as reasoning or tool call
        step_type_response = self.llm.prompt(
            STEP_CLASSIFICATION_PROMPT.format(
                step_text=step.text,
                keys_list=", ".join(self.memory.keys())
            )
        )

        if "reasoning" in step_type_response:
            step.result = self.llm.prompt(REASONING_STEP_PROMPT.format(
                step_text=step.text, available_data=json.dumps(inputs, ensure_ascii=False)
            ))
        else:
            tool =  self._select_tool(step)
            params = self._generate_params(step, tool, inputs)
            step.result = self.tools.execute(tool, params)

        step.status = StepStatus.DONE
        self._remember(step, state)

        # Always track step execution in history
        state.history.append(f"Executed step: {step.text} -> {step.result}")
        logger.info("step_executed", step_text=step.text, step_type=step_type_response, result=str(step.result)[:100] if step.result is not None else None)


    def _remember(self, step, state):
        if step.output_key:
            # Remember the entire response if the plan expects an output
            self.memory[step.output_key] = step.result
            state.history.append(f"remembered {step.output_key} : {step.result}")

    def _select_tool(self, step):
        tools = self.tools.search(step.text, top_k=20)
        tool_id = self.llm.prompt(TOOL_SELECTION_PROMPT.format(
            step=step.text,
            tools_json="\n".join([t.get_summary() for t in tools]),
        ))
        if tool_id == "none":
            raise ToolSelectionError(f"No suitable tool was found for step: {step.text}")

        tool =  next((tool for tool in tools if tool.id == tool_id), None)
        if tool is None:
            raise ToolSelectionError(f"Selected tool ID '{tool_id}' is invalid for step: {step.text}")
        logger.info("tool_selected", step_text=step.text, tool=tool)

        # Load and return the full tool details
        return self.tools.load(tool)

    def _generate_params(self, step, tool, inputs):
        try:
            param_schema = tool.get_parameters() or {}
            prompt  = PARAMETER_GENERATION_PROMPT.format(
                step=step.text,
                tool_schema=json.dumps(param_schema, ensure_ascii=False),
                step_inputs=json.dumps(inputs, ensure_ascii=False),
                allowed_keys=",".join(param_schema.keys()),
            )

            params = self.llm.prompt_to_json(prompt, max_retries=2)
            # dict comprehension to limit returned params to those defined in the schema.
            return {k: v for k, v in params.items() if k in param_schema}

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ParameterGenerationError(
                f"Failed to generate valid JSON parameters for step '{step.text}': {e}", tool
            ) from e
