"""
Interactive CLI agent that reads goals from stdin and outputs to stdout.
"""
import logging
import sys
from typing import Any

from .base_agent import BaseAgent
from ..reasoners.models import ReasoningResult

logger = logging.getLogger(__name__)


class InteractiveCLIAgent(BaseAgent):
    """
    Interactive command-line agent.
    
    Reads goals from the CLI inbox, processes them using the reasoner,
    and outputs results to stdout. Supports graceful exit on "quit".
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the interactive CLI agent."""
        super().__init__(*args, **kwargs)
        self._running = False
    
    def spin(self) -> None:
        """
        Main agent loop that processes goals from inbox.
        
        Continues until the inbox is closed or an exit command is received.
        """
        logger.info("Starting InteractiveCLIAgent")
        print("AI Agent started. Type 'quit' to exit.")
        print("=" * 50)
        
        self._running = True
        
        try:
            while self.should_continue():
                goal = self.inbox.get_next_goal()
                
                if goal is None:
                    # Inbox is closed or no more goals
                    break
                
                try:
                    # Process the goal
                    result = self.process_goal(goal)
                    
                    # Handle output
                    self.handle_output(result)
                    
                    # Acknowledge successful processing
                    self.inbox.acknowledge_goal(goal)
                    
                except Exception as e:
                    error_msg = f"Error processing goal: {str(e)}"
                    logger.error(error_msg)
                    print(f"❌ {error_msg}", file=sys.stderr)
                    
                    # Reject the goal
                    self.inbox.reject_goal(goal, error_msg)
                
                print()  # Add spacing between interactions
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
        
        finally:
            self._running = False
            self.inbox.close()
            logger.info("InteractiveCLIAgent stopped")
    
    def handle_input(self, input_data: Any) -> str:
        """
        Handle input from the user/environment.
        
        For CLI agent, this just returns the input as-is since
        the inbox already handles input parsing.
        
        Args:
            input_data: Raw input data from the interface
            
        Returns:
            Processed goal string
        """
        return str(input_data).strip()
    
    def handle_output(self, result: ReasoningResult) -> None:
        """
        Handle output to the user/environment.
        
        Formats and prints the reasoning result to stdout.
        
        Args:
            result: Reasoning result to present
        """
        if result.success:
            print(f"✅ **Answer:** {result.final_answer}")
            
            if result.tool_calls:
                print(f"\n📋 **Used {len(result.tool_calls)} tool(s) in {result.iterations} iteration(s):**")
                for i, call in enumerate(result.tool_calls, 1):
                    tool_name = call.get('tool_name', call.get('tool_id', 'Unknown'))
                    print(f"  {i}. {tool_name}")
        else:
            print(f"❌ **Failed:** {result.final_answer}")
            if result.error_message:
                print(f"   Error: {result.error_message}")
    
    def should_continue(self) -> bool:
        """
        Determine if the agent should continue processing.
        
        Returns:
            True if agent should keep running, False to stop
        """
        return self._running and self.inbox.has_goals()
    
    def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
