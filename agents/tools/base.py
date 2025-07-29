"""Abstract interface for a tool provider."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ToolBase(ABC):
    """Abstract base class for tool metadata."""

    def __init__(self, id: str):
        self.id = id

    def __str__(self) -> str:
        """Short string description for logging purposes."""
        return f"Tool({self.id})"

    @abstractmethod
    def get_summary(self) -> str:
        """Return summary information for LLM tool selection."""
        raise NotImplementedError

    @abstractmethod
    def get_details(self) -> str:
        """Return detailed information for LLM reflection."""
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return detailed parameter schema for LLM parameter generation."""
        raise NotImplementedError


class JustInTimeToolingBase(ABC):
    """Abstract contract for a tool-providing backend."""

    @abstractmethod
    def search(self, query: str, *, top_k: int = 10) -> List[ToolBase]:
        """Search for tools matching a natural language query."""
        raise NotImplementedError

    @abstractmethod
    def load(self, tool: ToolBase) -> ToolBase:
        """Load the full specification for a single tool."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, tool: ToolBase, parameters: Dict[str, Any]) -> Any:
        """Execute a tool with the given parameters."""
        raise NotImplementedError
