class ImplicitReasoningError(Exception):
    """Raised when an error occurs during Implicit Reasoning."""

class ActionNodeMissingError(ImplicitReasoningError):
    """Raised when the Act component is invoked without a preceding Action node."""

class ThinkFormatError(ImplicitReasoningError):
    """Raised when the Think component returns an invalid or unparsable output."""


