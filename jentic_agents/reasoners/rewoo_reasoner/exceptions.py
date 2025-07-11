class MissingInputError(KeyError):
    """Raised when a required memory key is absent."""

class ReasoningStepError(RuntimeError):
    """Raised when a reasoning-only step cannot produce a valid result."""


class ToolSelectionError(RuntimeError):
    """Raised when a suitable tool cannot be selected for a step."""


class ParameterGenerationError(RuntimeError):
    """Raised when parameters for a tool cannot be generated."""

