from __future__ import annotations

from collections import deque
import re
from textwrap import dedent
from typing import Deque

from agents.reasoner.sequential.reasoner import Step
from agents.reasoner.sequential.planners.base import Plan
from agents.llm.base_llm import BaseLLM
from utils.logger import get_logger, trace_method
logger = get_logger(__name__)


PLAN_GENERATION_PROMPT = dedent("""
    You are an expert planning assistant.

    TASK
    • Decompose the *user goal* below into a **markdown bullet-list** plan.

    OUTPUT FORMAT
    1. Return **only** the fenced list (triple back-ticks) — no prose before or after.
    2. Each bullet should be on its own line, starting with "- ".
    3. Each bullet = <verb> <object> … followed, in this order, by (input: key_a, key_b) (output: key_c)
       where the parentheses are literal.
    4. `output:` key is mandatory when the step's result is needed later; exactly one **snake_case** identifier.
    5. `input:` is optional; if present, list comma-separated **snake_case** keys produced by earlier steps.
    6. Do **not** mention specific external tool names.

    SELF-CHECK
    After drafting, silently verify — regenerate the list if any check fails:
    • All output keys unique & snake_case.
    • All input keys reference existing outputs.
    • No tool names or extra prose outside the fenced block.

    REAL GOAL
    Goal: {goal}
""").strip()

class BulletListPlan(Plan):
    """An LLM-based planner that generates a markdown bullet list."""
    
    _BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)(.*)$")
    _IO_DIRECTIVE_PATTERN = re.compile(r"\((input|output):\s*([^)]*)\)")

    def __init__(self, *, llm: BaseLLM, max_retries: int = 1):
        super().__init__(llm=llm)
        self.max_retries = max_retries

    def _parse_and_validate(self, markdown: str) -> Deque[Step]:
        """Parse and validate a flat markdown bullet list into a queue of ``Step`` objects.

        Raises:
            ValueError: If the plan is empty, has duplicate output keys,
                        or uses an input key before it is defined.
        """
        steps: Deque[Step] = deque()
        available_outputs: set[str] = set()

        for raw_line in markdown.splitlines():
            if not raw_line.strip() or not (bullet_match := self._BULLET_PATTERN.match(raw_line)):
                continue
            bullet = bullet_match.group(1).rstrip()

            # Parse input/output directives
            # Example: "Search for data (input: user_query, api_key) (output: search_results)"
            input_keys: list[str] = []
            output_key = None
            for io_match in self._IO_DIRECTIVE_PATTERN.finditer(bullet):
                directive_type, keys_info = io_match.groups()
                if directive_type == "input":
                    input_keys.extend(k.strip() for k in keys_info.split(',') if k.strip())
                else:  # output
                    output_key = keys_info.strip()

            # Validate input keys against previously defined outputs
            for key in input_keys:
                if key not in available_outputs:
                    logger.error("invalid_input_key", key=key, step_text=bullet)
                    raise ValueError(f"Input key '{key}' used before being defined.")

            # Validate output key uniqueness
            if output_key:
                if output_key in available_outputs:
                    logger.error("duplicate_output_key", key=output_key, step_text=bullet)
                    raise ValueError(f"Duplicate output key found: '{output_key}'")
                available_outputs.add(output_key)

            # Create step only after validation passes
            cleaned_text = self._IO_DIRECTIVE_PATTERN.sub("", bullet).strip()
            steps.append(
                Step(
                    text=cleaned_text,
                    output_key=output_key,
                    input_keys=input_keys,
                )
            )

        # Check for empty plan
        if not steps:
            logger.error("empty_plan_generated", markdown=markdown)
            raise ValueError("Planner produced an empty plan")

        return steps

    @trace_method
    def __call__(self, goal: str) -> Deque[Step]:
        """Generate and validate a plan, with retries on failure."""
        prompt = PLAN_GENERATION_PROMPT.format(goal=goal)

        for attempt in range(self.max_retries + 1):
            response = self.llm.prompt(prompt)
            logger.info("plan_generated", goal=goal, plan=response[:200] + "..." if len(response) > 200 else response)

            # Strip optional markdown code fence
            response = response.strip("`").lstrip("markdown").strip()

            try:
                result = self._parse_and_validate(response)
                logger.info("plan_validation_success", step_count=len(result))
                logger.info("Goal", goal=goal)
                for i, step in enumerate(result, 1):
                    logger.info("Step", step_text=step.text, output_key=step.output_key, input_keys=step.input_keys)
                return result
            except ValueError as e:
                logger.warning("plan_validation_failed", goal=goal, attempt=attempt, error=str(e))
                continue

        # Fallback to a single, do-everything step if all retries fail
        logger.warning("plan_fallback_used", goal=goal, max_retries=self.max_retries)
        return deque([Step(text=goal)])
