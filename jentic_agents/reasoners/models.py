"""Canonical data models shared by all reasoners.

This module consolidates the core data structures (`Step`, `Tool`, `ReasonerState`)
previously duplicated in individual reasoner packages.  All future reasoners
MUST import these models instead of defining local copies.
"""
from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Any, Deque, List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "Step",
    "ReasonerState",
    "ReasoningResult",
]


class Step(BaseModel):
    """A single bullet-plan step produced by an LLM planner.

    Attributes
    ----------
    text
        Raw step text with no leading bullet symbols.
    indent
        Indentation level (0-based). Determines nested sub-steps.
    status
        Execution status lifecycle: ``pending`` → ``running`` → ``done`` / ``failed``.
    result
        Data returned from executing the step (tool output or reasoning result).
    output_key / input_keys
        Explicit data-flow annotations to pass values between steps via memory.
    """

    text: str
    indent: int = 0
    status: str = "pending"  # pending | running | done | failed
    result: Optional[Any] = None

    # Explicit data-flow metadata (optional)
    output_key: Optional[str] = None  # snake_case key produced by this step
    input_keys: List[str] = Field(default_factory=list)  # keys this step consumes

    # ------------------------------------------------------------------
    # Execution modality (tool vs. pure reasoning)
    # ------------------------------------------------------------------
    class StepType(Enum):
        TOOL = auto()
        REASONING = auto()

    step_type: "Step.StepType" = StepType.TOOL

    # Error handling / reflection bookkeeping
    error: Optional[str] = None
    reflection_attempts: int = 0
    retry_count: int = 0

class ReasonerState(BaseModel):
    """Mutable state shared across reasoning iterations."""

    goal: str
    plan: Deque[Step] = Field(default_factory=deque)
    history: List[str] = Field(default_factory=list)
    is_complete: bool = False


class ReasoningResult(BaseModel):
    """Lightweight summary returned by a Reasoner run."""

    final_answer: str
    iterations: int
    tool_calls: List[dict[str, Any]]
    success: bool
    error_message: str | None = None

