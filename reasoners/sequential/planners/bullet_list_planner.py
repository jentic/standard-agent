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
    Jentic enables agentic systems to discover, evaluate, and execute API operations through a unified agent interface, powered by the Open Agentic Knowledge (OAK) project and MCP (Multi-Agent Coordination Protocol) protocol. You specialize in transforming high-level user goals into structured, step-by-step plans that can be executed by Jentic-aligned agents.

    Your responsibilities include:
    - Decomposing goals into modular, API-compatible actions
    - Sequencing steps logically with clear data flow
    - Labeling outputs for downstream use in later steps
    - Anticipating failure points and providing graceful fallback logic

    Each step you output may correspond to an API lookup, data transformation, or action — and is designed to be executed by another system, not a human. Your plans must be structurally strict and executable without revision.
    </role>

    <main_instructions>
    Transform the user's goal into a structured plan, optimized for execution by API-integrated agents.

    <rules>
    1. Each step must be a top-level bullet beginning with `- ` and no indentation.
    2. For any step that requires an API or tool call, the `→ keyword search query:` line must be indented by exactly two spaces and placed immediately after the relevant step.
    3. Add `→ keyword search query:` only if the API call is completely necessary. Pure reasoning or internal transformation steps do **not** need it.
    4. Each step must follow this format: `<verb> <object> (input: input_a, input_b) (output: result_key)`
      - Use `(input: ...)` only when the step requires prior outputs.
      - `(output: result_key)` is a **required** unique snake_case identifier for the step's result.
    5. Do **not** include tool names or APIs within the step description. Avoid explanatory prose outside the described structure.
    6. Try to use CRUD and API specific verbs in your steps.
    </rules>
    </main_instructions>

    <keyword_instructions>
      <role>
      You are a highly specialized Keyword Generator. Your sole purpose is to create concise and effective keyword search queries for API or tool calls.
      Generate a focused keyword search query for an API/tool call based on the given step.
      </role>

      <rules>
      - **Focus on Capability:** Describe *what* the tool does, not specific user data.
      - **Concise:** 4-6 keywords; prioritize precision over brevity.
      - **Structure:** `ACTION + RESOURCE TYPE + [SERVICE] + [CONTEXT]`.
      </rules>

      <component_definitions>
      1. **Primary Action Verb:** Tool's fundamental operation. Choose from: `send`, `post`, `notify`, `message`, `get`, `fetch`, `list`, `search`, `find`, `create`, `add`, `make`, `generate`, `upload`, `download`, `save`, `attach`, `update`, `delete`, `assign`, `manage`, `invite`, `remove`.
      2. **Resource Type:** Object tool acts on (`email`, `message`, `file`, `event`, `issue`, `video`, `member`, `article`, `task`, `card`, `link`, `channel`, `user`, `playlist`).
      3. **Service Context:** Platform if explicitly mentioned (`gmail`, `slack`, `discord`, `github`, `spotify`, `stripe`, `asana`, `trello`, `youtube`, `twilio`). **Always include when explicit; use standard abbreviations.**
      4. **Distinguishing Context:** Critical qualifiers for differentiation:
          - **Location:** `channel`, `folder`, `repository`, `board`, `list`, `server`, `inbox`
          - **Temporal:** `latest`, `new`, `recent`
          - **Scope:** `user`, `member`, `group`
          - **Operation:** `assign`, `notification`, `attachment`
      </component_definitions>

      <exclusions>
      **NEVER INCLUDE:** User-specific content (search queries, names, dates, IDs, file names, message content), or generic terms (`content`, `data`, `information`, `about`).
      **NEVER INCLUDE:** Markdown code block delimiters (``` or ```markdown) in your output.
      </exclusions>

      <critical_patterns>
      **Essential Qualifiers:**
      - **User/Member Operations:** Always include `user` or `member` (e.g., "get user asana", "add member mailchimp")
      - **Channel/Group Communications:** Include platform + `channel` (e.g., "send message discord channel")
      - **Assignment Operations:** Use `assign` verb specifically (e.g., "create task asana assign")
      - **Latest/New Content:** Add temporal qualifier (e.g., "get email gmail new")
      - **File with Destination:** Include location context (e.g., "upload file drive folder")
      - **Playlist/List Operations:** Include container type (e.g., "get playlist spotify", "get list trello")
      </critical_patterns>

      <quality_validation>
      **Before finalizing, verify:**
      1. Does this distinguish from similar operations on the same platform?
      2. Would this rank the correct tool above alternatives?
      3. Is the service context included when it's critical for tool selection?
      4. Are the keywords specific enough to avoid generic matches?
      </quality_validation>

      <keyword_output_format>
      `→ keyword search query: "<action_verb> <resource_type> <service> [context]"`
      </keyword_output_format>
    </keyword_instructions>

      <skip_queries_for>
      Pure reasoning tasks, data transformation without external tools, or logic operations.
      </skip_queries_for>

      <examples>
      Example 1 — Goal: "Search NYT articles about artificial intelligence and send them to Discord channel 12345"

      <example_output>
      - Get recent New York Times (NYT) articles mentioning "artificial intelligence" (output: nyt_articles)
        → keyword search query: "get article nytimes new york times search query filter"
      - send articles as a Discord message to Discord channel 12345 (input: nyt_articles) (output: post_confirmation)
        → keyword search query: "send message discord channel post content"
      </example_output>

      Example 2 — Goal: "Gather the latest 10 Hacker News posts about 'AI', summarise them, and email the summary to alice@example.com"

      <example_output>
      - fetch latest 10 Hacker News posts containing "AI" (output: hn_posts)
        → keyword search query: "get fetch posts hackernews searchquery filter"
      - summarise hn_posts into a concise bullet list (input: hn_posts) (output: summary_text)
      - email summary_text to alice@example.com (input: summary_text) (output: email_confirmation)
        → keyword search query: "post send email gmail to user"
      </example_output>
      </examples>

    <output_format>
    Output a bullet list of steps, each following the required step and keyword query formatting.
    Do not include the goal statement or any explanatory prose or formatting.
    Do **not** include markdown code block delimiters (``` or ```markdown) in your output.
    </output_format>

    <goal>
    Goal: {goal}
    </goal>
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
