from __future__ import annotations

from collections import deque
import re
from typing import Deque, List

from reasoners.models import Step
from reasoners.sequential.interface import Planner
from reasoners.prompts import PLAN_GENERATION_PROMPT
from utils.logger import get_logger
logger = get_logger(__name__)

_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)(.*)$")
_IO_DIRECTIVE_PATTERN = re.compile(r"\((input|output):\s*([^)]*)\)")

def _strip_bullet(text: str) -> str:
    """Remove leading bullet/number and extra whitespace."""
    match = _BULLET_PATTERN.match(text)
    return match.group(1).rstrip() if match else text.strip()

def _parse_bullet_plan(markdown: str) -> Deque[Step]:
    """Parse a flat markdown bullet list into a queue of ``Step`` objects."""

    steps: Deque[Step] = deque()
    lines = markdown.splitlines()
    i = 0
    
    while i < len(lines):
        raw_line = lines[i]
        if not raw_line.strip() or not _BULLET_PATTERN.match(raw_line):
            i += 1
            continue
            
        stripped = _strip_bullet(raw_line)
        input_keys: List[str] = []
        output_key = None
        keyword_search_query = None
        
        # Parse input/output directives
        for io_match in _IO_DIRECTIVE_PATTERN.finditer(stripped):
            kind, payload = io_match.groups()
            if kind == "output":
                output_key = payload.strip()
            else:
                input_keys = [k.strip() for k in payload.split(',') if k.strip()]
        
        cleaned_text = _IO_DIRECTIVE_PATTERN.sub("", stripped).strip()
        
        # Check next line for keyword search query
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith("→ keyword search query:"):
                keyword_search_query = next_line.split("→ keyword search query:", 1)[1].strip().strip('"')
                i += 1  # Skip the keyword line in next iteration
        
        # Create step with all parsed information
        steps.append(
            Step(
                text=cleaned_text,
                output_key=output_key,
                input_keys=input_keys,
                keyword_search_query=keyword_search_query,
            )
        )
        logger.debug(f"phase=PLAN PARSED SUCCESSFULLY")
        i += 1
        
    return steps

def _validate_plan(steps: Deque[Step]) -> None:
    """Checks for logical consistency in a plan.

    Raises:
        ValueError: If the plan is empty, has duplicate output keys,
                    or uses an input key before it is defined.
    """
    if not steps:
        logger.error("Planner produced an empty plan")
        raise ValueError("Planner produced an empty plan")

    seen_outputs: set[str] = set()
    for step in steps:
        # Check for undefined input keys against outputs from *previous* steps
        for key in step.input_keys:
            if key not in seen_outputs:
                logger.error(f"Input key '{key}' used before being defined.")
                raise ValueError(f"Input key '{key}' used before being defined.")

        # Check for duplicate output keys and then add the current one
        if step.output_key:
            if step.output_key in seen_outputs:
                logger.error(f"phase=PLAN VALIDATION FAILED: Duplicate output key found: '{step.output_key}'")
                raise ValueError(f"Duplicate output key found: '{step.output_key}'")
            seen_outputs.add(step.output_key)

class BulletListPlanner(Planner):
    """An LLM-based planner that generates a markdown bullet list."""

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries

    def plan(self, goal: str) -> Deque[Step]:
        """Generate and validate a plan, with retries on failure."""
        if not self.llm:
            logger.error(f"{__name__}: LLM not attached. Call attach_services first.")
            raise RuntimeError(f"{__name__}: LLM not attached. Call attach_services first.")
        prompt = PLAN_GENERATION_PROMPT.format(goal=goal)
        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.max_retries + 1):
            response = self.llm.chat(messages).strip()

            logger.info(f"phase=PLAN_GENERATED plan={response}")

            # Strip optional markdown code fence
            if response.startswith("```"):
                response = response.strip("`").lstrip("markdown").strip()

            try:
                steps = _parse_bullet_plan(response)
                _validate_plan(steps)  # Raises ValueError on failure
                return steps  # Success
            except ValueError:
                logger.error(f"phase=PLAN_GENERATION FAILED : Invalid plan")
                # Plan was invalid, loop will retry if possible
                continue

        # Fallback to a single, do-everything step if all retries fail
        return deque([Step(text=goal)])
