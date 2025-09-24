from __future__ import annotations

from typing import Any, Dict, List
from agents.tools.exceptions import ToolError
from agents.tools.base import ToolBase
from utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningError(Exception):
    """Base exception for all reasoning-related errors."""

    def __init__(self, message: str):
        super().__init__(message)
        logger.warning(
            "reasoning_error",
            error_type=self.__class__.__name__,
            message=message,
        )


class ToolSelectionError(ReasoningError):
    """A suitable tool could not be found/validated for a step."""


class ParameterGenerationError(ToolError):
    """Valid parameters for a tool could not be generated."""


class UnknownParameterError(ParameterGenerationError):
    """LLM explicitly marked required parameters as <UNKNOWN>. LLM is encouraged to use <UNKNOWN> for parameters it cannot infer from the given context."""
    
    def __init__(self, tool: ToolBase, unknown_params: List[str], step_text: str, generated_params: Dict[str, Any]):
        self.unknown_params = unknown_params
        message = (
            f"LLM indicated missing data for parameters: {', '.join(unknown_params)} in step '{step_text}'. "
            f"Generated parameters: {generated_params}. Tool '{tool.id}' requires these parameters for successful execution."
        )
        super().__init__(message, tool)


class MissingParameterError(ParameterGenerationError):
    """LLM omitted required parameters entirely."""
    
    def __init__(self, tool: ToolBase, missing_params: List[str], step_text: str, generated_params: Dict[str, Any]):
        self.missing_params = missing_params
        message = (
            f"Generated parameters missing required keys: {', '.join(missing_params)} for step '{step_text}'. "
            f"Generated parameters: {generated_params}. Tool '{tool.id}' requires these parameters for successful execution."
        )
        super().__init__(message, tool)


