from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Any, Deque, List, Optional
from pydantic import BaseModel, Field


__all__ = [
    "Step",
    "StepStatus",
    "ReasonerState",
    "ReasoningResult",
]

class StepStatus(str, Enum):
    """The lifecycle status of a single reasoning step."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class Step(BaseModel):
    """A single bullet-plan step produced by an LLM planner."""

    text: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None

    # Explicit data-flow metadata (optional)
    output_key: Optional[str] = None  # snake_case key produced by this step
    input_keys: List[str] = Field(default_factory=list)  # keys this step consumes

    class StepType(Enum):
        TOOL = auto()
        REASONING = auto()

    step_type: "Step.StepType" = StepType.TOOL

    # Error handling / reflection bookkeeping
    error: Optional[str] = None
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
