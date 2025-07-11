"""
CLI-based inbox that reads goals from standard input.
"""
import sys
from typing import Optional, TextIO
from .base_inbox import BaseInbox


class CLIInbox(BaseInbox):
    """
    Inbox that reads goals from command line input.
    
    Reads from stdin and treats each line as a separate goal.
    Useful for interactive CLI agents and testing.
    """
    
    def __init__(self, input_stream: Optional[TextIO] = None, prompt: str = "Enter goal: "):
        """
        Initialize CLI inbox.
        
        Args:
            input_stream: Stream to read from (defaults to stdin)
            prompt: Prompt to display when asking for input
        """
        self.input_stream = input_stream or sys.stdin
        self.prompt = prompt
        self._closed = False
        self._current_goal: Optional[str] = None
    
    def get_next_goal(self) -> Optional[str]:
        """
        Get the next goal from user input.
        
        Returns:
            Next goal string, or None if no more input available
        """
        if self._closed:
            return None
        
        try:
            # Display prompt if using stdin
            if self.input_stream == sys.stdin:
                print(self.prompt, end="", flush=True)
            
            line = self.input_stream.readline()
            
            # EOF reached
            if not line:
                self._closed = True
                return None
            
            goal = line.strip()
            
            # Empty line or quit commands
            if not goal or goal.lower() in ('quit', 'exit', 'q'):
                self._closed = True
                return None
            
            self._current_goal = goal
            return goal
            
        except (EOFError, KeyboardInterrupt):
            self._closed = True
            return None
    
    def acknowledge_goal(self, goal: str) -> None:
        """
        Acknowledge that a goal has been processed.
        
        Args:
            goal: The goal that was successfully processed
        """
        # For CLI inbox, acknowledgment is just logging
        # In more complex implementations, this might update a database
        if goal == self._current_goal:
            self._current_goal = None
    
    def reject_goal(self, goal: str, reason: str) -> None:
        """
        Reject a goal that couldn't be processed.
        
        Args:
            goal: The goal that failed to process
            reason: Reason for rejection
        """
        # For CLI inbox, just print the rejection reason
        print(f"Goal rejected: {reason}", file=sys.stderr)
        if goal == self._current_goal:
            self._current_goal = None
    
    def has_goals(self) -> bool:
        """
        Check if there are pending goals.
        
        For CLI inbox, this is true unless we've been closed.
        
        Returns:
            True if goals might be available, False if definitely none
        """
        return not self._closed
    
    def close(self) -> None:
        """
        Clean up inbox resources.
        """
        self._closed = True
        if self.input_stream != sys.stdin:
            self.input_stream.close()
