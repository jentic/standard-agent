from agents.tools.exceptions import ToolError
from utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningError(Exception):
    """Base exception for all reasoning-related errors."""
    def __init__(self, message: str):
        super().__init__(message)
        
        # Log all reasoning errors at warning level for visibility
        logger.warning("reasoning_error",
            error_type=self.__class__.__name__,
            message=message
        )


class MissingInputError(ReasoningError, KeyError):
    """A required memory key is absent."""


class ToolSelectionError(ReasoningError):
    """A suitable tool could not be found for a step."""


class ParameterGenerationError(ToolError):
    """Valid parameters for a tool could not be generated."""
