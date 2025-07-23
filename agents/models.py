"""Data models for the agent layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

__all__ = ["Goal"]

@dataclass
class Goal:
    """Represents a single objective for an agent to solve."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

