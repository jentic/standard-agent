"""A simple outbox that logs the result to the console."""
from jentic_agents.reasoners.models import ReasoningResult
from jentic_agents.utils.logger import get_logger
from .base_outbox import BaseOutbox

__all__ = ["LoggingOutbox"]


class LoggingOutbox(BaseOutbox):
    """A simple outbox that logs the result to the console."""

    def __init__(self):
        self._logger = get_logger(self.__class__.__name__)

    def send(self, result: ReasoningResult) -> None:
        """Log the final answer from the reasoning result."""
        self._logger.info(
            "Outbox received result: success=%s, answer='%s'",
            result.success,
            result.final_answer,
        )
