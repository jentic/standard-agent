
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolError(Exception):
    """Base exception for all tool-related errors."""
    def __init__(self, message: str, tool):
        self.tool = tool
        self.message = message
        super().__init__(f"'{tool}': {message}")
        
        # Log all tool errors at warning level for visibility
        logger.warning("tool_error", 
            error_type=self.__class__.__name__,
            tool_id=getattr(tool, 'id', str(tool)),
            message=message
        )

class ToolNotFoundError(ToolError):
    """The specified tool ID is not recognized."""

class ToolExecutionError(ToolError):
    """A tool fails to execute for any reason."""

class ToolCredentialsMissingError(ToolExecutionError):
    """A tool fails to execute because of missing credentials"""
