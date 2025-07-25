"""Data models for the agent layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

__all__ = ["Goal", "PendingAPIKeyInfo"]


@dataclass
class Goal:
    """Represents a single objective for an agent to solve."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PendingAPIKeyInfo:
    """Represents a missing API info object."""

    env_var: str
    tool_id: str
    api_name: str | None
    user_help_message: str