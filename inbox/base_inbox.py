"""
Abstract base class for inbox systems that deliver goals to agents.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional


class BaseInbox(ABC):
    """
    Abstract base class for goal/task delivery systems.
    
    Provides a stream of goals from various sources (CLI, message queue, 
    cron job, web API, etc.) to decouple goal sources from agent internals.
    """
    
    @abstractmethod
    def get_next_goal(self) -> Optional[str]:
        """
        Get the next available goal/task.
        
        Returns:
            Next goal string, or None if no goals available
        """
        pass
    
    @abstractmethod
    def acknowledge_goal(self, goal: str) -> None:
        """
        Acknowledge that a goal has been processed.
        
        Args:
            goal: The goal that was successfully processed
        """
        pass
    
    @abstractmethod
    def reject_goal(self, goal: str, reason: str) -> None:
        """
        Reject a goal that couldn't be processed.
        
        Args:
            goal: The goal that failed to process
            reason: Reason for rejection
        """
        pass
    
    def goal_stream(self) -> Iterator[str]:
        """
        Generator that yields goals as they become available.
        
        Yields:
            Goal strings as they arrive
        """
        while True:
            goal = self.get_next_goal()
            if goal is not None:
                yield goal
            else:
                break
    
    @abstractmethod
    def has_goals(self) -> bool:
        """
        Check if there are pending goals.
        
        Returns:
            True if goals are available, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Clean up inbox resources.
        """
        pass
