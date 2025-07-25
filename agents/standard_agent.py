"""
StandardAgent

Lightweight façade that wires together the core runtime services (LLM, memory,
external tools) with a pluggable *reasoner* implementation.  The
agent owns the services; the reasoner simply uses what the agent provides.
"""
from __future__ import annotations

from  agents.models import Goal, PendingAPIKeyInfo
from  memory.base_memory import BaseMemory
from  reasoners.base_reasoner import BaseReasoner
from  llm.base_llm import BaseLLM
from  reasoners.models import ReasoningResult
from  tools.interface import ToolInterface
from  tools.exceptions import MissingAPIKeyError

from  uuid import uuid4
from  enum import Enum
import os

MISSING_API_KEY_PROMPT = """
You are an assistant helping a user provide a **missing API key**.

Missing environment variable: `{env_var}`
API / Service: {api_name}

Please craft a short, actionable message for the user that includes:
1. A one-line explanation of why the key is required.
2. If you know how to obtain or generate this key, provide a brief hint or official link.
3. Show exactly how to send the key back in the format: `{env_var}=<value>`.
4. Optionally mention that they can add it to an `.env` file for future runs.

Return only the helpful instructions—no extra commentary.
"""

class AgentState(str, Enum):
    READY                   = "READY"
    BUSY                    = "BUSY"
    NEEDS_ATTENTION         = "NEEDS_ATTENTION"
    WAITING_FOR_API_KEY     = "WAITING_FOR_API_KEY"

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
        self._pending_api_key_info: PendingAPIKeyInfo | None = None

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
            friendly_msg = f"Missing API Key for {exc.api_name or 'a required tool'}. Please provide a value for the environment variable: `{exc.env_var}`"
            if self.llm:
                prompt = MISSING_API_KEY_PROMPT.format(
                    env_var=exc.env_var,
                    api_name=getattr(exc, "api_name", "unknown"),
                )
                try:
                    friendly_msg = self.llm.chat([{"role": "user", "content": prompt}]).strip()
                except Exception:
                    pass

            self._state = AgentState.WAITING_FOR_API_KEY
            self._pending_api_key_info = PendingAPIKeyInfo(
                env_var=exc.env_var,
                tool_id=exc.tool_id,
                api_name=getattr(exc, "api_name", None),
                user_help_message=friendly_msg
            )

            raise exc

        except Exception:
            self._state = AgentState.NEEDS_ATTENTION
            raise

    def accept_api_key(self, key: str, value: str) -> None:
        """
        Injects the missing API key into the environment and resets state.
        """
        if not self._pending_api_key_info:
            raise RuntimeError("No API key is currently pending")

        os.environ[key] = value
        self._pending_api_key_info = None
        self._state = AgentState.READY

    def get_pending_api_key_info(self) -> PendingAPIKeyInfo | None:
        """
        Returns info about the missing key, or None if not waiting.
        """
        return self._pending_api_key_info
