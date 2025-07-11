"""Abstract contract for delivering a reasoner's final result."""
from abc import ABC, abstractmethod

from jentic_agents.reasoners.models import ReasoningResult

__all__ = ["BaseOutbox"]


class BaseOutbox(ABC):
    """Abstract contract for delivering a reasoner's final result."""

    @abstractmethod
    def send(self, result: ReasoningResult) -> None:  # pragma: no cover
        """
        Deliver the reasoning result to a downstream system.

        This method should be implemented by concrete outbox classes to handle
        the delivery of the result, for example, by sending an email,
        posting to a message queue, or calling a webhook.
        """
        raise NotImplementedError
