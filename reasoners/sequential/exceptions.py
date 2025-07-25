class MissingInputError(KeyError):
    """Raised when a required memory key is absent."""


class ReasoningStepError(RuntimeError):
    """Raised when a reasoning-only step cannot produce a valid result."""


class ToolSelectionError(RuntimeError):
    """Raised when a suitable tool cannot be selected for a step."""

    def __init__(self, message: str, *, tool_id: str | None = None):
        self.tool_id = tool_id
        self.message = message
        super().__init__(message)


class ParameterGenerationError(RuntimeError):
    """Raised when parameters for a tool cannot be generated."""

    def __init__(self, tool_id: str, message: str):
        self.tool_id = tool_id
        self.message = message
        super().__init__(f"Tool '{tool_id}': {message}")

