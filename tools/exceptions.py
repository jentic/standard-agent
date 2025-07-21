"""
Generic, tool-related exceptions that can be used across the agent framework.
"""

class ToolExecutionError(Exception):
    """Raised when a tool fails to execute for any reason."""
    pass
