from __future__ import annotations
from copy import deepcopy
from textwrap import dedent
from collections.abc import MutableMapping

from agents.reasoner.sequential.reasoner import ReasonerState, Step, StepStatus
from agents.reasoner.sequential.reflectors.base import Reflect
from agents.tools.base import JustInTimeToolingBase
from agents.tools.exceptions import ToolError
from agents.llm.base_llm import BaseLLM

from utils.logger import get_logger, trace_method
logger = get_logger(__name__)


### Prompts ###
BASE_REFLECTION_PROMPT = dedent("""
    <role>
    You are a Self-Healing Engine operating within the Jentic agent ecosystem. Your mission is to enable resilient agentic applications by diagnosing step failures and proposing precise corrective actions. You specialize in error analysis, parameter adjustment, and workflow recovery to maintain system reliability.

    Your core responsibilities:
    - Analyze step failures and identify root causes
    - Propose targeted fixes for parameter or tool issues
    - Maintain workflow continuity through intelligent recovery
    - Enable autonomous error resolution within the agent pipeline
    </role>

    <goal>
    Analyze the failed step and propose a single, precise fix that will allow the workflow to continue successfully.
    </goal>

    <input>
    Goal: {goal}
    Failed Step: {step}
    Failed Tool: {failed_tool_id}
    Error: {error_type}: {error_message}
    Tool Schema: {tool_schema}
    </input>

    <constraints>
    - Output ONLY valid JSON - no explanation, markdown, or backticks and should be parsable using JSON.parse()
    - Must start with '{{' and end with '}}'
    - Choose one action: 'retry_params', 'change_tool', 'rephrase_step', or 'give_up'
    - Provide all required fields for the chosen action
    </constraints>
    
    <self_check>
    After drafting, silently verify — all the constraints are met, if not, regenerate your answer
    </self_check>

    <output_format>
    {{
      "reasoning": "Brief explanation of why the step failed",
      "action": "one of 'retry_params', 'change_tool', 'rephrase_step', or 'give_up'",
      "tool_id": "(Required if action is 'change_tool') The ID of the new tool to use",
      "params": "(Required if action is 'retry_params' or 'change_tool') Valid JSON object of parameters",
      "step": "(Required if action is 'rephrase_step') The new, improved text for the step"
    }}
    </output_format>
""").strip()

ALTERNATIVE_TOOLS_SECTION = dedent("""
    **Alternative Tools:**
    The previous tool failed. Please select a more suitable tool from the following list to achieve the step's goal.
    {alternative_tools}
""").strip()


class ReWOOReflect(Reflect):
    """Retry-oriented reflection for ReWOO executors."""

    def __init__(self, *, llm: BaseLLM, tools: JustInTimeToolingBase, memory: MutableMapping, max_retries: int = 2):
        super().__init__(llm=llm, tools=tools, memory=memory)
        self.max_retries = max_retries

    # ------------------------------- public ------------------------------
    @trace_method
    def __call__(
        self,
        error: Exception,
        step: Step,
        state: ReasonerState
    ) -> None:
        logger.info("step_error_recovery", error_type=error.__class__.__name__, step_text=step.text, retry_count=step.retry_count)
        step.status = StepStatus.FAILED
        step.error = str(error)

        if step.retry_count >= self.max_retries:
            logger.warning("max_retries_exceeded", step_text=step.text, max_retries=self.max_retries)
            state.history.append(
                f"Giving-up after {self.max_retries} retries: {step.text}"
            )
            return

        failed_tool_id = error.tool.id if isinstance(error, ToolError) else None
        tool_details = error.tool.get_details() if isinstance(error, ToolError) else None

        prompt = BASE_REFLECTION_PROMPT.format(
            goal           = state.goal,
            step           = step.text,
            failed_tool_id = failed_tool_id,
            error_type     = error.__class__.__name__,
            error_message  = str(error),
            tool_details    = tool_details,
        )

        # Try to find alternative tools to call.
        alternatives = "\n".join([
            t.get_summary()
            for t in self.tools.search(step.text, top_k=25)
            if t.id != failed_tool_id
        ])
        prompt += ALTERNATIVE_TOOLS_SECTION.format(alternative_tools=alternatives)

        # Get the LLM to reflect
        decision = self.llm.prompt_to_json(prompt, max_retries=2)

        # Do as the LLM says
        action = decision.get("action")
        state.history.append(f"Reflection decision: {decision}")

        if action == "give_up":
            logger.warning("reflection_giving_up", step_text=step.text, error_type=error.__class__.__name__, retry_count=step.retry_count, reasoning=decision.get("reasoning"))
            return

        # Prepare a new step object to add to the plan.
        new_step = deepcopy(step)
        new_step.retry_count += 1
        new_step.status = StepStatus.PENDING

        if action == "rephrase_step":
            new_step.text = str(decision.get("step", new_step.text))
            logger.info("reflection_rephrase", original_step=step.text, new_step=new_step.text)

        elif action == "change_tool":
            tool_id = decision.get("tool_id")
            self.memory[f"forced_tool:{new_step.text}"] = tool_id
            logger.info("reflection_change_tool", step_text=new_step.text, forced_tool_id=tool_id)

        elif action == "retry_params":
            params = decision.get("params")
            self.memory[f"forced_params:{new_step.text}"] = params
            logger.info("reflection_retry_params", step_text=new_step.text, forced_params=params)

        state.plan.appendleft(new_step)
