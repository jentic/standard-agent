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
    You are a self-healing reasoning engine. A step in your plan failed. Your task is to analyze the error and propose a single, precise fix.

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

    **Tool Details (if available):**
    {tool_details}
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
