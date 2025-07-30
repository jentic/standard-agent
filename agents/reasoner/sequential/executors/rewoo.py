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
from agents.tools.jentic import JenticTool
from utils.logger import get_logger, trace_method
logger = get_logger(__name__)

### Prompts
STEP_CLASSIFICATION_PROMPT = dedent("""
    <role>
    You are a Step Classifier within the Agent ecosystem. 
    Your sole purpose is to determine whether a given step requires external API/tool execution or can be completed through internal reasoning alone.
    </role>

    <goal>
    Classify the provided step as either TOOL or REASONING based on whether it requires external API calls. 
    Use classification_rules for guidance
    </goal>

    <input>
    Step: {step_text}
    Available Memory Keys: {keys_list}
    </input>

    <classification_rules>
    TOOL steps require:
    - External API calls (e.g., "search articles", "send email", etc.)
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
""")

REASONING_STEP_PROMPT = dedent("""
    <role>
    You are a Data Processor within the Agent ecosystem. 
    Your mission is to perform precise data transformations and reasoning operations on available information.
    You specialize in content analysis, data extraction, and logical processing to support agent workflows.

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
    Available Data: {available_data}
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
""")

TOOL_SELECTION_PROMPT = dedent("""
    <role>
   You are an expert orchestrator working within the Agent API ecosystem.
   Your job is to select the best tool to execute a specific plan step, using a list of available tools. 
   Each tool may vary in API domain, supported actions, and required parameters. 
   You must evaluate each tool's suitability and return the **single best matching tool** — or the word none if none qualify.

   Your selection will be executed by an agent, so precision and compatibility are critical.
   </role>

   <instructions>
   Analyze the provided step and evaluate all candidate tools. Use the scoring criteria to assess each tool's fitness for executing the step. 
   Return the tool `id` with the highest total score. If no tool scores ≥60, return the word none.
   You are selecting the **most execution-ready** tool, not simply the closest match.
   </instructions>

   <input>
   Step: {step}

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
   3. If multiple tools tie for the highest score, choose the one that appears first in the Tools list.
   4. If no tool scores at least 60 points, return none.
   5. Do **not** include any explanation, formatting, or metadata — only the tool `id` or none.
   6. Use available step context and known inputs to inform scoring.
   7. Penalize tools severely if they are misaligned with the intended action or platform (if mentioned in the step).
   8. Never select a tool from an incorrect domain if the step explicitly specifies a specific one.
   </rules>

   <output_format>
   Respond with a **single line** that contains exactly the selected tool's `id` — no quotes, backticks, or leading/trailing whitespace.
   **No additional text or formatting** should be included.
   </output_format>
""")

PARAMETER_GENERATION_PROMPT = dedent("""
    <role>
    You are a Parameter Builder within the Agent ecosystem. 
    Your mission is to enable seamless API execution by generating precise parameters from step context and memory data. 
    You specialize in data extraction, content formatting, and parameter mapping to ensure successful tool execution.

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

        if "reasoning" in step_type_response.lower():
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
        if rewoo_reflector_suggested_tool_id := self.memory.get(f"rewoo_reflector_suggested_tool:{step.text}"):
            logger.info("using_rewoo_reflector_suggested_tool", step_text=step.text, rewoo_reflector_suggested_tool_id=rewoo_reflector_suggested_tool_id)
            del self.memory[f"rewoo_reflector_suggested_tool:{step.text}"]
            return self.tools.load(JenticTool({"id": rewoo_reflector_suggested_tool_id}))

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
        if rewoo_reflector_suggested_params := self.memory.get(f"rewoo_reflector_suggested_params:{step.text}"):
            logger.info("using_rewoo_reflector_suggested_params", step_text=step.text, rewoo_reflector_suggested_params=rewoo_reflector_suggested_params)
            del self.memory[f"rewoo_reflector_suggested_params:{step.text}"]
            param_schema = tool.get_parameters() or {}
            return {k: v for k, v in rewoo_reflector_suggested_params.items() if k in param_schema}

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
