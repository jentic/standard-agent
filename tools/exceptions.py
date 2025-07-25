"""
Generic, tool-related exceptions that can be used across the agent framework.
"""

class ToolExecutionError(Exception):
    """Raised when a tool fails to execute for any reason."""

    def __init__(self, message: str, *, tool_id: str):
        self.tool_id = tool_id
        self.message = message
        super().__init__(f"Tool '{tool_id}': {message}")


class MissingAPIKeyError(Exception):
    """Raised when a required environment variable for tool execution is not set."""

    def __init__(
        self,
        env_var: str,
        *,
        tool_id: str,
        api_name: str | None = None,
        message: str | None = None,
    ) -> None:
        self.env_var = env_var
        self.tool_id = tool_id
        self.api_name = api_name
        base_msg = message or f"Environment variable '{env_var}' is not set with the required API KEY."
        if api_name:
            base_msg += f" (required for API '{api_name}')"
        super().__init__(f"Tool '{tool_id}': {base_msg}")
