from __future__ import annotations

from collections import deque
import re
from typing import Deque, List

from reasoners.models import Step
from reasoners.sequential.interface import Planner
from utils.logger import get_logger
logger = get_logger(__name__)

_BULLET_PATTERN = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s)(.*)$")
_IO_DIRECTIVE_PATTERN = re.compile(r"\((input|output):\s*([^)]*)\)")

# Bullet List Planner specific prompt
PLAN_GENERATION_PROMPT: str = (
    """
    <role>
    You are a world-class planning assistant operating within the Jentic platform.
    You specialize in transforming high-level user goals into structured, step-by-step plans that can be executed by API-integrated agents.
    </role>

    <goal>
    Decompose the user goal below into a markdown bullet-list plan.
    </goal>

    <output_format>
    1. Return **only** the bullet list — no prose before or after.
    2. Each bullet should be on its own line, starting with "- ".
    3. Each bullet = <verb> <object> ... followed, in this order, by (input: key_a, key_b) (output: key_c)
       where the parentheses are literal.
    4. `output:` key is mandatory when the step's result is needed later; exactly one **snake_case** identifier.
    5. `input:` is optional; if present, list comma-separated **snake_case** keys produced by earlier steps.
    6. Do **not** mention specific external tool names.
    7. **CRITICAL**: Preserve quantity constraints from the goal throughout the plan.
    </output_format>

    <self_check>
    After drafting, silently verify — regenerate the list if any check fails:
    • All output keys unique & snake_case.
    • All input keys reference existing outputs.
    • No tool names or extra prose outside the list.
    </self_check>

    <examples>
    Example 1 — Goal: "Search 2 NYT articles about artificial intelligence and send them to Discord channel 12345"
    - Get 2 recent New York Times articles mentioning "artificial intelligence" (output: nyt_articles)
    - Send 2 articles as a Discord message to Discord channel 12345 (input: nyt_articles) (output: post_confirmation)

    Example 2 — Goal: "Gather the latest 10 Hacker News posts about 'AI', summarise them, and email the summary to alice@example.com"
    - Fetch latest 10 Hacker News posts containing "AI" (output: hn_posts)
    - Summarise hn_posts into a concise bullet list (input: hn_posts) (output: summary_text)
    - Email summary_text to alice@example.com (input: summary_text) (output: email_confirmation)
    </examples>

    <real_goal>
    Goal: {goal}
    </real_goal>
    """
)

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
        
        # Parse input/output directives
        for io_match in _IO_DIRECTIVE_PATTERN.finditer(stripped):
            kind, payload = io_match.groups()
            if kind == "output":
                output_key = payload.strip()
            else:
                input_keys = [k.strip() for k in payload.split(',') if k.strip()]
        
        cleaned_text = _IO_DIRECTIVE_PATTERN.sub("", stripped).strip()
        
        # Create step with parsed information
        steps.append(
            Step(
                text=cleaned_text,
                output_key=output_key,
                input_keys=input_keys,
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
