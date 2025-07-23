from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from llm.base_llm import BaseLLM
from memory.base_memory import BaseMemory
from reasoners.models import ReasoningResult
from tools.interface import ToolInterface

class BaseReasoner(ABC):
    """
    Abstract contract for a reasoning-loop implementation.

    Each instance owns its own `llm`, `tools`, and `memory`.
    You can supply them in the constructor *or* later via `set_services()`.
    """

    def __init__(
        self,
        *,
        llm: BaseLLM | None = None,
        tools: ToolInterface | None = None,
        memory: BaseMemory | None = None,
    ) -> None:
        self._llm: BaseLLM | None = None
        self._tools: ToolInterface | None = None
        self._memory: BaseMemory | None = None

        # Constructor-time injection (optional)
        if llm or tools or memory:
            self.set_services(llm=llm, tools=tools, memory=memory)

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            raise RuntimeError("Reasoner has no LLM wired in")
        return self._llm

    @property
    def tools(self) -> ToolInterface:
        if self._tools is None:
            raise RuntimeError("Reasoner has no Tools interface wired in")
        return self._tools

    @property
    def memory(self) -> BaseMemory:
        if self._memory is None:
            raise RuntimeError("Reasoner has no Memory wired in")
        return self._memory

    def set_services(
        self,
        *,
        llm: BaseLLM | None = None,
        tools: ToolInterface | None = None,
        memory: BaseMemory | None = None,
    ) -> None:
        """Inject (or overwrite) any combination of services."""
        if llm is not None:
            self._llm = llm
        if tools is not None:
            self._tools = tools
        if memory is not None:
            self._memory = memory

        # Push the context to sub-components if any
        self._pass_context_to_components()

    def _pass_context_to_components(self) -> None:
        """
        Hook for subclasses to broadcast the services to their components.

        Base implementation does nothing.
        """
        return

    @abstractmethod
    def run(self, goal: str, *, meta: Dict[str, Any] | None = None) -> ReasoningResult:
        raise NotImplementedError
