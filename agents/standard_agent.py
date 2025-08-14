"""
StandardAgent

Lightweight façade that wires together the core runtime services (LLM, memory,
external tools) with a pluggable *reasoner* implementation.  The
agent owns the services; the reasoner simply uses what the agent provides.
"""
from __future__ import annotations

from  collections.abc import MutableMapping
from  collections import deque
from  agents.reasoner.base import BaseReasoner, ReasoningResult
from  agents.llm.base_llm import BaseLLM
from  agents.tools.base import JustInTimeToolingBase
from  agents.goal_preprocessor.base import BaseGoalPreprocessor

from  uuid import uuid4
from  enum import Enum
from textwrap import dedent

_SUMMARIZE_PROMPT = dedent(
    """
    <role>
    You are the Final Answer Synthesizer for autonomous agents within the Agent ecosystem. Your mission is to transform raw execution logs into clear, user-friendly responses that demonstrate successful goal achievement. You specialize in data interpretation, content formatting, and user communication.

    Your core responsibilities:
    - Analyze execution logs to extract meaningful results
    - Assess data sufficiency for reliable answers
    - Format responses using clear markdown presentation
    - Maintain professional, helpful tone in all communications
    </role>

    <goal>
    Generate a comprehensive final answer based on the execution log that directly addresses the user's original goal.
    </goal>

    <input>
    User's Goal: {goal}
    Execution Log: {history}
    </input>

    <instructions>
    1. Review the execution log to understand what actions were taken
    2. Assess if the collected data is sufficient to achieve the user's goal
    3. If insufficient data, respond with: "ERROR: insufficient data for a reliable answer."
    4. If sufficient, synthesize a comprehensive answer that:
       - Directly addresses the user's goal
       - Uses only information from the execution log
       - Presents content clearly with markdown formatting
       - Maintains helpful, professional tone
       - Avoids revealing internal technical details
    </instructions>

    <constraints>
    - Use only information from the execution log
    - Do not add external knowledge or assumptions
    - Do not reveal internal monologue or technical failures
    - Present results as if from a helpful expert assistant
    </constraints>

    <missing_api_keys>
    If the execution log shows a tool call failed for lack of credentials (look for Tool Unauthorized: in the Execution Log):

    Only include the following section when you cannot produce a sufficient, reliable answer (e.g., you would otherwise return "ERROR: insufficient data for a reliable answer.").
    If you can synthesize a complete answer that satisfies the goal, omit this section entirely.

    Return an additional short block that starts with  
    `Agent attempted tools that require configuration:`  ← only once, even if several tools failed

    **FOR EACH TOOL** the agent detected and attempted but could not complete due to missing configuration, include a separate block for each tool:
    • **Tool attempted** – the tool that was attempted, including api_name and api_vendor
    • **How to enable** – brief steps with official link (if known) to obtain credentials or connect the account
    • **Action step** – suggest configuring the required API credentials for this tool and then retrying the goal

    Wording guidance:
    - Keep tone helpful and proactive, focusing on enabling the tool.
    - No extra commentary—just clear, actionable instructions.
    </missing_api_keys>

    <output_format>
    Clear, user-friendly response using markdown formatting (headings, lists, bold text as appropriate)
    </output_format>
    """
).strip()

class AgentState(str, Enum):
    READY               = "READY"
    BUSY                = "BUSY"
    NEEDS_ATTENTION     = "NEEDS_ATTENTION"

class StandardAgent:
    """Top-level class that orchestrates the main components of the agent framework."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: JustInTimeToolingBase,
        memory: MutableMapping,
        reasoner: BaseReasoner,

        # Optionals
        goal_preprocessor: BaseGoalPreprocessor = None,
        conversation_history_window: int = 5
    ):
        """Initializes the agent.

        Args:
            llm: The language model instance.
            tools: The interface for accessing external tools.
            memory: The memory backend.
            reasoner: The reasoning engine that will use the services.

            goal_preprocessor: An OPTIONAL component to preprocess the user's goal.
            conversation_history_window: The number of past interactions to keep in memory.
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.reasoner = reasoner

        self.goal_preprocessor = goal_preprocessor
        self.memory.setdefault("conversation_history", deque(maxlen=conversation_history_window))

        self._state: AgentState = AgentState.READY

    @property
    def state(self) -> AgentState:
        return self._state

    def solve(self, goal: str) -> ReasoningResult:
        """Solves a goal synchronously (library-style API)."""
        run_id = uuid4().hex

        if self.goal_preprocessor:
            revised_goal, intervention_message = self.goal_preprocessor.process(goal, self.memory.get("conversation_history"))
            if intervention_message:
                self.memory["conversation_history"].append({ "goal": goal, "result": f"user intervention message: {intervention_message}"})
                return ReasoningResult(success=False, final_answer=intervention_message)
            goal = revised_goal

        self.memory[f"goal:{run_id}"] = goal
        self._state = AgentState.BUSY

        try:
            result = self.reasoner.run(goal)
            result.final_answer = self.llm.prompt(_SUMMARIZE_PROMPT.format(goal=goal, history=getattr(result, "transcript", "")))

            self.memory[f"result:{run_id}"] = result
            self.memory["conversation_history"].append({"goal": goal, "result": result.final_answer})
            self._state = AgentState.READY
            return result

        except Exception:
            self._state = AgentState.NEEDS_ATTENTION
            raise
