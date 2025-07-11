"""Outbox pattern for handling agent results."""
"""Outbox pattern for handling agent results."""
from .base_outbox import BaseOutbox
from .cli_outbox import CLIOutbox

__all__ = ["BaseOutbox", "CLIOutbox"]

