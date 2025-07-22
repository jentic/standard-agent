"""
Generic, tool-related exceptions that can be used across the agent framework.
"""

class ToolExecutionError(Exception):
    """Raised when a tool fails to execute for any reason."""

    def __init__(self, message: str, *, tool_id: str):
        self.tool_id = tool_id
        self.message = message
        super().__init__(f"Tool '{tool_id}': {message}")
