"""
Abstract base class for AI agents that compose reasoner, memory, inbox, and Jentic client.
"""
from abc import ABC, abstractmethod
from typing import Any

from ..reasoners.rewoo_reasoner_contract import BaseReWOOReasoner
from ..reasoners.models import ReasoningResult
from ..memory.base_memory import BaseMemory
from ..inbox.base_inbox import BaseInbox
from ..platform.jentic_client import JenticClient


class BaseAgent(ABC):
    """
    Base class for AI agents that wire together core components.
    
    Composes:
    - Reasoner: Implements the reasoning loop logic
    - Memory: Stores and retrieves information across sessions
    - Inbox: Receives goals/tasks from various sources
    - JenticClient: Interface to Jentic workflows and tools
    
    Concrete agents override I/O methods while inheriting core behavior.
    """
    
    def __init__(
        self,
        reasoner: BaseReWOOReasoner,
        memory: BaseMemory,
        inbox: BaseInbox,
        jentic_client: JenticClient
    ):
        """
        Initialize agent with core components.
        
        Args:
            reasoner: Reasoning loop implementation
            memory: Memory backend for storing information
            inbox: Source of goals/tasks
            jentic_client: Interface to Jentic platform
        """
        self.reasoner = reasoner
        self.memory = memory
        self.inbox = inbox
        self.jentic_client = jentic_client
    
    @abstractmethod
    def spin(self) -> None:
        """
        Main agent loop that processes goals from inbox.
        
        This is the primary entry point for running the agent.
        Concrete implementations define how the agent interacts with users.
        """
        pass
    
    def process_goal(self, goal: str) -> ReasoningResult:
        """
        Process a single goal using the reasoning loop.
        
        Args:
            goal: The objective or question to process
            
        Returns:
            ReasoningResult with the agent's response
        """
        # Store the goal in memory
        self.memory.store("current_goal", goal)
        
        # Execute reasoning loop
        result = self.reasoner.run(goal)
        
        # Store the result in memory
        self.memory.store("last_result", result.model_dump())
        
        return result
    
    @abstractmethod
    def handle_input(self, input_data: Any) -> str:
        """
        Handle input from the user/environment.
        
        Args:
            input_data: Raw input data from the interface
            
        Returns:
            Processed goal string
        """
        pass
    
    @abstractmethod
    def handle_output(self, result: ReasoningResult) -> None:
        """
        Handle output to the user/environment.
        
        Args:
            result: Reasoning result to present
        """
        pass
    
    @abstractmethod
    def should_continue(self) -> bool:
        """
        Determine if the agent should continue processing.
        
        Returns:
            True if agent should keep running, False to stop
        """
        pass
