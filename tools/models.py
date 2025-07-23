"""Data models for the generic tool interface."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel

__all__ = [
    "Tool",
]


class Tool(BaseModel):
    """Metadata for a single API tool (workflow or operation)."""

    id: str
    name: str
    description: str
    api_name: str = "unknown"
    parameters: dict[str, Any] = {}
