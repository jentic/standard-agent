"""
StandardAgent

Lightweight façade that wires together the core runtime services (LLM, memory,
external tools) with a pluggable *reasoner* implementation.  The
agent owns the services; the reasoner simply uses what the agent provides.
"""
from __future__ import annotations

from  agents.models import Goal
from  memory.base_memory import BaseMemory
from  reasoners.base_reasoner import BaseReasoner
from  llm.base_llm import BaseLLM
from  reasoners.models import ReasoningResult
from  tools.interface import ToolInterface
from  tools.exceptions import MissingAPIKeyError

from  uuid import uuid4
from  enum import Enum

class AgentState(str, Enum):
    READY               = "READY"
    BUSY                = "BUSY"
    NEEDS_ATTENTION     = "NEEDS_ATTENTION"

class StandardAgent:
    """Wires together a reasoner with shared services (LLM, memory, tools)."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        tools: ToolInterface,
        memory: BaseMemory,
        reasoner: BaseReasoner,
    ):
        """Initializes the agent and injects services into the reasoner.

        Args:
            llm: The language model instance.
            tools: The interface for accessing external tools.
            memory: The memory backend.
            reasoner: The reasoning engine that will use the services.
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.reasoner = reasoner

        # Explicit handshake to wire services into the reasoner
        self.reasoner.attach_services(llm=llm, tools=tools, memory=memory)

        self._state: AgentState = AgentState.READY

    @property
    def state(self) -> AgentState:
        return self._state

    def solve(self, goal: Goal) -> ReasoningResult:
        """Solves a goal synchronously (library-style API)."""
        run_id = uuid4().hex

        if hasattr(self.memory, "store"):
            self.memory.store(f"goal:{run_id}", goal)

        self._state = AgentState.BUSY

        try:
            result = self.reasoner.run(goal.text, meta=goal.metadata)
            self.memory.store(f"result:{run_id}", result.model_dump())
            self._state = AgentState.READY
            return result

        except MissingAPIKeyError as exc:
            self._state = AgentState.NEEDS_ATTENTION
            if self.llm:
                prompt = (
                    "You are an assistant helping a user provide a **missing API key**.\n\n"
                    f"Missing environment variable: `{exc.env_var}`\n"
                    f"API / Service: {getattr(exc, 'api_name', 'unknown')}\n\n"
                    "Please craft a short, actionable message for the user that includes:\n"
                    "1. A one-line explanation of why the key is required.\n"
                    "2. If you know how to obtain or generate this key, provide a brief hint or official link.\n"
                    "3. Show exactly how to send the key back in the format: `{exc.env_var}=<value>`.\n"
                    "4. Optionally mention that they can add it to an `.env` file for future runs.\n\n"
                    "Return only the helpful instructions—no extra commentary.\n"
                )
                try:
                    friendly_msg = self.llm.chat([{"role": "user", "content": prompt}]).strip()
                    raise MissingAPIKeyError(
                        env_var=exc.env_var,
                        tool_id=exc.tool_id,
                        api_name=getattr(exc, "api_name", None),
                        message=friendly_msg,
                    ) from exc
                except Exception:
                    # If LLM fails, fall back to original exception
                    raise
            raise
        except Exception:
            self._state = AgentState.NEEDS_ATTENTION
            raise
