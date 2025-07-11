"""Abstract interface for a tool provider."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from jentic_agents.tools.models import Tool


class ToolInterface(ABC):
    """Abstract contract for a tool-providing backend."""

    @abstractmethod
    def search(self, query: str, *, top_k: int = 10) -> List[Tool]:
        """Search for tools matching a natural language query."""
        raise NotImplementedError

    @abstractmethod
    def load(self, tool_id: str) -> Tool:
        """Load the full specification for a single tool by its ID."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with the given parameters."""
        raise NotImplementedError
