from __future__ import annotations

from collections import deque
import re
from typing import Deque, List

from jentic_agents.utils.llm import BaseLLM

from ...models import Step
from ..interface import Planner

_INDENT_SIZE = 2
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)(.*)$")
_IO_DIRECTIVE_PATTERN = re.compile(r"\((input|output):\s*([^)]*)\)")

PLAN_GENERATION_PROMPT: str = (
    """
    You are an expert planning assistant.

    TASK
    • Decompose the *user goal* below into a **markdown bullet-list** plan.

    OUTPUT FORMAT
    1. Return **only** the fenced list (triple back-ticks) — no prose before or after.
    2. Each top-level bullet starts at indent 0 with "- "; sub-steps indent by exactly two spaces.
    3. Each bullet = <verb> <object> … followed, in this order, by (input: key_a, key_b) (output: key_c)
       where the parentheses are literal.
    4. `output:` key is mandatory when the step’s result is needed later; exactly one **snake_case** identifier.
    5. `input:` is optional; if present, list comma-separated **snake_case** keys produced by earlier steps.
    6. Do **not** mention specific external tool names.

    SELF-CHECK  
    After drafting, silently verify — regenerate the list if any check fails:
    • All output keys unique & snake_case.  
    • All input keys reference existing outputs.  
    • Indentation correct (2 spaces per level).  
    • No tool names or extra prose outside the fenced block.

    REAL GOAL
    Goal: {goal}
    ```
    """
)

def _line_indent(text: str) -> int:
    """Return indent level (0-based) from leading spaces."""
    spaces = len(text) - len(text.lstrip(" "))
    return spaces // _INDENT_SIZE

def _strip_bullet(text: str) -> str:
    """Remove leading bullet/number and extra whitespace."""
    match = _BULLET_PATTERN.match(text)
    return match.group(1).rstrip() if match else text.strip()

def _parse_bullet_plan(markdown: str) -> Deque[Step]:
    """Parse an indented markdown bullet list into a queue of ``Step`` objects."""
    steps: Deque[Step] = deque()
    for raw_line in markdown.splitlines():
        if not raw_line.strip() or not _BULLET_PATTERN.match(raw_line):
            continue
        indent = _line_indent(raw_line)
        stripped = _strip_bullet(raw_line)
        input_keys: List[str] = []
        output_key = None
        for io_match in _IO_DIRECTIVE_PATTERN.finditer(stripped):
            kind, payload = io_match.groups()
            if kind == "output":
                output_key = payload.strip()
            else:
                input_keys = [k.strip() for k in payload.split(',') if k.strip()]
        cleaned_text = _IO_DIRECTIVE_PATTERN.sub("", stripped).strip()
        steps.append(
            Step(
                text=cleaned_text,
                indent=indent,
                output_key=output_key,
                input_keys=input_keys,
            )
        )
    return steps


def _validate_plan(steps: Deque[Step]) -> None:
    """Checks for logical consistency in a plan.

    Raises:
        ValueError: If the plan is empty, has duplicate output keys,
                    or uses an input key before it is defined.
    """
    if not steps:
        raise ValueError("Planner produced an empty plan")

    seen_outputs: set[str] = set()
    for step in steps:
        # Check for undefined input keys against outputs from *previous* steps
        for key in step.input_keys:
            if key not in seen_outputs:
                raise ValueError(f"Input key '{key}' used before being defined.")

        # Check for duplicate output keys and then add the current one
        if step.output_key:
            if step.output_key in seen_outputs:
                raise ValueError(f"Duplicate output key found: '{step.output_key}'")
            seen_outputs.add(step.output_key)

class BulletListPlanner(Planner):
    """A planner that generates a bullet-point plan using an LLM."""

    def __init__(self, llm: BaseLLM, *, max_retries: int = 1):
        self.llm = llm
        self.max_retries = max_retries

    def plan(self, goal: str) -> Deque[Step]:
        """Generate and validate a plan, with retries on failure."""
        prompt = PLAN_GENERATION_PROMPT.format(goal=goal)
        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.max_retries + 1):
            response = self.llm.chat(messages).strip()

            # Strip optional markdown code fence
            if response.startswith("```"):
                response = response.strip("`").lstrip("markdown").strip()

            try:
                steps = _parse_bullet_plan(response)
                _validate_plan(steps)  # Raises ValueError on failure
                return steps  # Success
            except ValueError:
                # Plan was invalid, loop will retry if possible
                continue

        # Fallback to a single, do-everything step if all retries fail
        return deque([Step(text=goal, indent=0)])
