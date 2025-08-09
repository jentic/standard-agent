from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, Tuple

from agents.reasoner.sequential.exceptions import ToolSelectionError
from agents.reasoner.implicit.exceptions import ActionNodeMissingError
from agents.tools.base import ToolBase
from typing import TYPE_CHECKING
from agents.reasoner.implicit.act.base import Act
from agents.reasoner.implicit.models import ReasonNode, ReasonKind
if TYPE_CHECKING:
    from agents.reasoner.implicit.reasoner import ImplicitState

from utils.logger import get_logger
logger = get_logger(__name__)

TOOL_SELECTION_PROMPT = dedent(
    """
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
"""
).strip()


PARAMETER_GENERATION_PROMPT = dedent(
    """
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
    DATA: {data}
    SCHEMA: {schema}
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
    1. Extract actual values using the data extraction rules
    2. **CRITICAL**: Check STEP text for quantity constraints (e.g., "send 3 articles", "post 2 items")
    3. Format content appropriately for the target API
    4. Generate valid parameters using only ALLOWED_KEYS
    5. **CRITICAL**: Only use parameters that are explicitly documented in the SCHEMA - do not infer or add undocumented parameters
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
    """
).strip()


class ReACTAct(Act):
    """Selects a tool via LLM and executes it via JustInTimeToolingBase.

    Simplicity first: single-pass selection driven by structured ACTION JSON.
    No blind retries. If selection fails, surface an error so the next turn can REASON.
    """

    def __call__(self, state: "ImplicitState") -> Tuple[str, Dict[str, Any], Any]:
        # Require last thought to be an Action node
        last = state.turns[-1] if state.turns else None
        if not last or not isinstance(last.thought, ReasonNode) or last.thought.kind != ReasonKind.ACTION:
            raise ActionNodeMissingError("Act called without an Action node")
        action_node: ReasonNode = last.thought
        query = action_node.text

        tool = self._select_and_load_tool(query)
        params = self._generate_params(tool, state, query)
        observation = self.tools.execute(tool, params)
        return tool.id, params, observation


    def _select_and_load_tool(self, query: str):
        candidates: list[ToolBase] = self.tools.search(query, top_k=self.top_k)
        logger.info("tool_search", query=query, top_k=self.top_k, candidate_count=len(candidates))
        tools_json = "\n".join([t.get_summary() for t in candidates])

        tool_id = self.llm.prompt(TOOL_SELECTION_PROMPT.format(step=query, tools_json=tools_json)).strip()
        logger.info("tool selected", tool_id=tool_id)
        if tool_id == "none" or not tool_id:
            raise ToolSelectionError(f"No suitable tool selected for step: {query}")

        tool = next((t for t in candidates if t.id == tool_id), None)
        if tool is None:
            raise ToolSelectionError(f"Selected tool id '{tool_id}' not in candidate list")

        return self.tools.load(tool)

    def _generate_params(
        self,
        tool,
        state: "ImplicitState",
        step_text: str,
    ) -> Dict[str, Any]:
        schema = tool.get_parameters() or {}
        allowed_keys = ",".join(schema.keys()) if isinstance(schema, dict) else ""

        # Build full reasoning trace (goal + all turns) as plain strings
        trace_lines: list[str] = [f"Goal: {state.goal}"]
        for t in state.turns:
            if t.thought is not None:
                trace_lines.append(f"Thought: {t.thought}")
            if t.action is not None:
                trace_lines.append(f"Action: {t.action}")
            if t.observation is not None:
                trace_lines.append(f"Observation: {t.observation}")

        data: Dict[str, Any] = {"trace": "\n".join(trace_lines),}

        params_raw = self.llm.prompt_to_json(
            PARAMETER_GENERATION_PROMPT.format(
                step=step_text,
                data=json.dumps(data, ensure_ascii=False),
                schema=json.dumps(schema, ensure_ascii=False),
                allowed_keys=allowed_keys,
            ),
            max_retries=2,
        )
        params: Dict[str, Any] = {k: v for k, v in params_raw.items() if k in schema}
        logger.info("params_generated", tool_id=tool.id, params=params)
        return params


