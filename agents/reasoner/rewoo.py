from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from textwrap import dedent
from typing import Any, Deque, Dict, List, Optional
from copy import deepcopy

from agents.reasoner.base import BaseReasoner, ReasoningResult
from agents.llm.base_llm import BaseLLM
from agents.tools.base import JustInTimeToolingBase, ToolBase
from agents.tools.jentic import JenticTool
from agents.tools.exceptions import ToolError, ToolCredentialsMissingError
from agents.reasoner.exceptions import (ReasoningError, ToolSelectionError, ParameterGenerationError)

from utils.logger import get_logger

logger = get_logger(__name__)


# ReWOO-specific exception for missing plan inputs
class MissingInputError(ReasoningError, KeyError):
    """A required memory key by a step is absent (ReWOO plan dataflow)."""

    def __init__(self, message: str, missing_key: str | None = None):
        super().__init__(message)
        self.missing_key = missing_key


# ----------------------------- Prompts ---------------------------------

_PLAN_PROMPT = dedent(
    """
    <role>
    You are a world-class planning assistant operating within the Agent ecosystem.
    You specialize in transforming high-level user goals into structured, step-by-step plans that can be executed by API-integrated agents.
    </role>

    <goal>
    Decompose the user goal below into a markdown bullet-list plan.
    </goal>

    <output_format>
    1. Return only the fenced list (triple back-ticks) — no prose before or after.
    2. Each bullet should be on its own line, starting with "- ".
    3. Each bullet = <verb> <object> … followed, in this order, by (input: key_a, key_b) (output: key_c)
       where the parentheses are literal.
    4. output: key is mandatory when the step's result is needed later; exactly one snake_case identifier.
    5. input: is optional; if present, list comma-separated snake_case keys produced by earlier steps.
    6. Do not mention specific external tool names.
    </output_format>

    <self_check>
    After drafting, silently verify — regenerate the list if any check fails:
    • All output keys unique & snake_case.
    • All input keys reference existing outputs.
    • No tool names or extra prose outside the fenced block.
    </self_check>

    <real_goal>
    Goal: {goal}
    </real_goal>
    """
).strip()

_STEP_CLASSIFICATION_PROMPT = dedent(
    """
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
    """
).strip()

_REASONING_STEP_PROMPT = dedent(
    """
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
    """
).strip()

_TOOL_SELECTION_PROMPT = dedent(
    """
    <role>
    You are an expert orchestrator working within the Agent API ecosystem.
    Your job is to select the best tool to execute a specific plan step, using a list of available tools.
    Each tool may vary in API domain, supported actions, and required parameters.
    You must evaluate each tool's suitability and return the single best matching tool — or the word none if none qualify.

    Your selection will be executed by an agent, so precision and compatibility are critical.
    </role>

    <instructions>
    Analyze the provided step and evaluate all candidate tools. Use the scoring criteria to assess each tool's fitness for executing the step.
    Return the tool id with the highest total score. If no tool scores ≥60, return the word none.
    You are selecting the most execution-ready tool, not simply the closest match.
    </instructions>

    <input>
    Step: {step}

    Tools (JSON):
    {tools_json}
    </input>

    <scoring_criteria>
    - Action Compatibility (35 pts): Evaluate how well the tool's primary action matches the step's intent.
    - API Domain Match (30 pts): If the step explicitly mentions a platform, require a direct api_name match; otherwise pick a relevant domain.
    - Parameter Compatibility (20 pts): Required parameters should be present or inferable.
    - Workflow Fit (10 pts): Logical integration into surrounding workflow.
    - Simplicity & Efficiency (5 pts): Prefer direct solutions over unnecessarily complex ones.
    </scoring_criteria>

    <rules>
    1. Score each tool using the weighted criteria above. Max score: 100 points.
    2. Select the tool with the highest total score.
    3. If multiple tools tie for the highest score, choose the first.
    4. If no tool scores at least 60 points, return none.
    5. Output only the selected tool id or none.
    </rules>

    <output_format>
    Respond with a single line that contains exactly the selected tool's id — no quotes or extra text and no extra reasoning.
    </output_format>
    """
).strip()

_PARAMETER_GENERATION_PROMPT = dedent(
    """
    <role>
    You are a Parameter Builder within the Agent ecosystem.
    Your mission is to enable seamless API execution by generating precise parameters from step context and memory data.
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
    • Articles/News: Extract title/headline and URL fields, format as "Title: URL\n"
    • Arrays: Process each item, combine into formatted string
    • Nested Objects: Access properties using dot notation
    • Quantities: "a/an/one" = 1, "few" = 3, "several" = 5, numbers = exact
    • Array Slicing: When processing arrays from memory, look for quantity constraints in the STEP text and slice accordingly
    • Never use placeholder text - always extract real data from memory
    </data_extraction_rules>

    <instructions>
    1. Analyze MEMORY for relevant data structures
    2. Extract actual values using the data extraction rules
    3. CRITICAL: Check STEP text for quantity constraints
    4. Format content appropriately for the target API
    5. Generate valid parameters using only ALLOWED_KEYS
    6. CRITICAL: Only use parameters documented in the SCHEMA
    </instructions>

    <constraints>
    - Output ONLY valid JSON - no markdown or commentary
    - Use only keys from ALLOWED_KEYS
    - Extract actual data values from MEMORY
    </constraints>

    <output_format>
    Valid JSON object starting with {{ and ending with }}
    </output_format>
    """
).strip()

_REFLECTION_PROMPT = dedent(
    """
    <role>
    You are a Self-Healing Engine operating within the Agent ecosystem. Diagnose step failures and propose a precise corrective action.
    </role>

    <goal>
    Analyze the failed step and propose a single, precise fix that will allow the workflow to continue successfully.
    </goal>

    <input>
    Goal: {goal}
    Failed Step: {step}
    Failed Tool: {failed_tool_id}
    Error: {error_type}: {error_message}
    Tool Details: {tool_details}
    </input>

    <decision_guide>
    • retry_params – tool appropriate, inputs invalid; derive correct values
    • change_tool   – current tool cannot accomplish step; pick a better one
    • rephrase_step – step text ambiguous/misleading; rewrite it
    • give_up       – required parameter or critical info missing and cannot be inferred
    </decision_guide>

    <constraints>
    - Output ONLY valid JSON parsable by JSON.parse()
    - Choose one action: retry_params | change_tool | rephrase_step | give_up
    - Provide required fields for chosen action
    </constraints>

    <output_format>
    {{
      "reasoning": "...",
      "action": "retry_params|change_tool|rephrase_step|give_up",
      "tool_id": "(if action is change_tool)",
      "params": "(if action is retry_params or change_tool) object",
      "step": "(if action is rephrase_step) new text"
    }}
    </output_format>
    """
).strip()

_REFLECTION_ALTERNATIVES_SECTION = dedent(
    """
    Alternative Tools:
    {alternative_tools}
    """
).strip()

# ----------------------------- Data structures -------------------------


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Step:
    text: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    output_key: Optional[str] = None
    input_keys: List[str] = field(default_factory=list)
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class ReasonerState:
    goal: str
    plan: Deque[Step] = field(default_factory=deque)
    history: List[str] = field(default_factory=list)
    is_complete: bool = False


# ----------------------------- Reasoner --------------------------------

class ReWOOReasoner(BaseReasoner):
    DEFAULT_MAX_ITER = 20

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        max_iterations: int = DEFAULT_MAX_ITER,
        max_retries: int = 2,
        top_k: int = 15,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.top_k = top_k

    def run(self, goal: str) -> ReasoningResult:
        state = ReasonerState(goal=goal)

        # Plan
        state.plan = self._plan(goal)
        if not state.plan:
            raise RuntimeError("Planner produced an empty plan")

        iterations = 0

        # Execute with reflection
        while state.plan and iterations < self.max_iterations and not state.is_complete:
            step = state.plan.popleft()
            try:
                self._execute(step, state)
                iterations += 1
            except (ReasoningError, ToolError) as exc:
                if isinstance(exc, ToolCredentialsMissingError):
                    state.history.append(f"Tool Unauthorized: {str(exc)}")

                if isinstance(exc, MissingInputError):
                    state.history.append(
                        f"Stopping: missing dependency '{getattr(exc, 'missing_key', None)}' for step '{step.text}'. Proceeding to final answer."
                    )
                    break

                self._reflect(exc, step, state)

        transcript = "\n".join(state.history)
        success = not state.plan
        return ReasoningResult(iterations=iterations, success=success, transcript=transcript)

    def _plan(self, goal: str) -> Deque[Step]:
        generated_plan = (self.llm.prompt(_PLAN_PROMPT.format(goal=goal)) or "").strip("`").lstrip("markdown").strip()
        logger.info("plan_generated", goal=goal, plan=generated_plan)

        steps: Deque[Step] = deque()
        produced_keys: set[str] = set()

        BULLET_RE = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)(.*)$")
        IO_RE = re.compile(r"\((input|output):\s*([^)]*)\)")

        for raw_line in filter(str.strip, generated_plan.splitlines()):
            match = BULLET_RE.match(raw_line)
            if not match:
                continue
            bullet = match.group(1).rstrip()

            input_keys: List[str] = []
            output_key: Optional[str] = None

            for io_match in IO_RE.finditer(bullet):
                directive_type, keys_info = io_match.groups()
                if directive_type == "input":
                    input_keys.extend(k.strip() for k in keys_info.split(',') if k.strip())
                else:
                    output_key = keys_info.strip() or None

            for key in input_keys:
                if key not in produced_keys:
                    logger.warning("invalid_input_key", key=key, step_text=bullet)
                    raise ValueError(f"Input key '{key}' used before being defined.")

            if output_key:
                if output_key in produced_keys:
                    logger.warning("duplicate_output_key", key=output_key, step_text=bullet)
                    raise ValueError(f"Duplicate output key found: '{output_key}'")
                produced_keys.add(output_key)

            cleaned_text = IO_RE.sub("", bullet).strip()
            steps.append(Step(text=cleaned_text, output_key=output_key, input_keys=input_keys))

        if not steps:
            logger.warning("empty_plan_generated", goal=goal)
            return deque([Step(text=goal)])

        logger.info("plan_validation_success", step_count=len(steps))
        for s in steps:
            logger.info("plan_step", step_text=s.text, output_key=s.output_key, input_keys=s.input_keys)
        return steps

    def _execute(self, step: Step, state: ReasonerState) -> None:
        step.status = StepStatus.RUNNING

        try:
            inputs = {key: self.memory[key] for key in step.input_keys}
        except KeyError as e:
            missing_key = e.args[0]
            raise MissingInputError(f"Required memory key '{missing_key}' not found for step: {step.text}", missing_key=missing_key) from e

        step_type_response = self.llm.prompt(_STEP_CLASSIFICATION_PROMPT.format(step_text=step.text,keys_list=", ".join(self.memory.keys())))

        if "reasoning" in step_type_response.lower():
            step.result = self.llm.prompt(_REASONING_STEP_PROMPT.format(step_text=step.text, available_data=json.dumps(inputs, ensure_ascii=False)))
        else:
            tool = self._select_tool(step)
            params = self._generate_params(step, tool, inputs)
            step.result = self.tools.execute(tool, params)

        step.status = StepStatus.DONE

        if step.output_key:
            self.memory[step.output_key] = step.result
            state.history.append(f"remembered {step.output_key} : {step.result}")

        state.history.append(f"Executed step: {step.text} -> {step.result}")
        logger.info("step_executed", step_text=step.text, step_type=step_type_response, result=str(step.result)[:100] if step.result is not None else None)

    def _select_tool(self, step: Step) -> ToolBase:
        suggestion = self.memory.get(f"rewoo_reflector_suggestion:{step.text}")
        if suggestion and suggestion.get("action") in ("change_tool", "retry_params"):
            logger.info("using_reflector_suggested_tool", step_text=step.text, tool_id=suggestion.get("tool_id"))
            if suggestion.get("action") == "change_tool":
                del self.memory[f"rewoo_reflector_suggestion:{step.text}"]
            return self.tools.load(JenticTool({"id": suggestion.get("tool_id")}))

        tool_candidates = self.tools.search(step.text, top_k=20)
        tool_id = self.llm.prompt(_TOOL_SELECTION_PROMPT.format(step=step.text, tools_json="\n".join([t.get_summary() for t in tool_candidates])))

        if tool_id == "none":
            raise ToolSelectionError(f"No suitable tool was found for step: {step.text}")

        selected_tool = next((t for t in tool_candidates if t.id == tool_id), None)
        if selected_tool is None:
            raise ToolSelectionError(f"Selected tool ID '{tool_id}' is invalid for step: {step.text}")
        logger.info("tool_selected", step_text=step.text, tool=selected_tool)

        return self.tools.load(selected_tool)

    def _generate_params(self, step: Step, tool: ToolBase, inputs: Dict[str, Any]) -> Dict[str, Any]:
        suggestion = self.memory.pop(f"rewoo_reflector_suggestion:{step.text}", None)
        if suggestion and suggestion["action"] == "retry_params" and "params" in suggestion:
            logger.info("using_reflector_suggested_params", step_text=step.text, params=suggestion["params"])
            param_schema = tool.get_parameters() or {}
            return {k: v for k, v in suggestion["params"].items() if k in param_schema}

        try:
            param_schema = tool.get_parameters() or {}
            prompt = _PARAMETER_GENERATION_PROMPT.format(
                step=step.text,
                tool_schema=json.dumps(param_schema, ensure_ascii=False),
                step_inputs=json.dumps(inputs, ensure_ascii=False),
                allowed_keys=",".join(param_schema.keys()),
            )
            params_raw = self.llm.prompt_to_json(prompt, max_retries=self.max_retries)
            return {k: v for k, v in (params_raw or {}).items() if k in param_schema}
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ParameterGenerationError(
                f"Failed to generate valid JSON parameters for step '{step.text}': {e}", tool
            ) from e

    def _reflect(self, error: Exception, step: Step, state: ReasonerState) -> None:
        logger.info("step_error_recovery", error_type=error.__class__.__name__, step_text=step.text, retry_count=step.retry_count)
        step.status = StepStatus.FAILED
        step.error = str(error)

        if step.retry_count >= self.max_retries:
            logger.warning("max_retries_exceeded", step_text=step.text, max_retries=self.max_retries)
            state.history.append(f"Giving-up after {self.max_retries} retries: {step.text}")
            return

        failed_tool_id = error.tool.id if isinstance(error, ToolError) else None
        tool_details = error.tool.get_details() if isinstance(error, ToolError) else None

        prompt = _REFLECTION_PROMPT.format(
            goal=state.goal,
            step=step.text,
            failed_tool_id=failed_tool_id,
            error_type=error.__class__.__name__,
            error_message=str(error),
            tool_details=tool_details,
        )

        alternatives = [t for t in self.tools.search(step.text, top_k=self.top_k) if t.id != failed_tool_id]
        prompt += "\n" + _REFLECTION_ALTERNATIVES_SECTION.format(
            alternative_tools="\n".join([t.get_summary() for t in alternatives])
        )

        decision = self.llm.prompt_to_json(prompt, max_retries=2)
        action = (decision or {}).get("action")
        state.history.append(f"Reflection decision: {decision}")

        if action == "give_up":
            logger.warning(
                "reflection_giving_up",
                step_text=step.text,
                error_type=error.__class__.__name__,
                retry_count=step.retry_count,
                reasoning=(decision or {}).get("reasoning"),
            )
            return

        # Prepare a new step object to add to the plan.
        new_step = deepcopy(step)
        new_step.retry_count += 1
        new_step.status = StepStatus.PENDING

        if action == "rephrase_step":
            new_step.text = str((decision or {}).get("step", new_step.text))
            logger.info("reflection_rephrase", original_step=step.text, new_step=new_step.text)

        elif action == "change_tool":
            new_tool_id = (decision or {}).get("tool_id")
            self._save_reflector_suggestion(new_step, "change_tool", new_tool_id)
            logger.info("reflection_change_tool", step_text=new_step.text, new_tool_id=new_tool_id)

        elif action == "retry_params":
            params = (decision or {}).get("params", {})
            self._save_reflector_suggestion(new_step, "retry_params", failed_tool_id, params)
            logger.info("reflection_retry_params", step_text=new_step.text, params=params)

        state.plan.appendleft(new_step)

    def _save_reflector_suggestion(self, new_step: Step, action: str, tool_id: Optional[str], params: Dict[str, Any] | None = None) -> None:
        suggestion: Dict[str, Any] = {"action": action, "tool_id": tool_id}
        if params is not None:
            suggestion["params"] = params
        self.memory[f"rewoo_reflector_suggestion:{new_step.text}"] = suggestion


