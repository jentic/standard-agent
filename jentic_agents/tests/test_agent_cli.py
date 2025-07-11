"""
Unit tests for InteractiveCLIAgent.
"""
from io import StringIO
from unittest.mock import Mock, patch

from ..agents.interactive_cli_agent import InteractiveCLIAgent
from ..reasoners.models import ReasoningResult
from ..memory.scratch_pad import ScratchPadMemory
from ..inbox.cli_inbox import CLIInbox
from ..platform.jentic_client import JenticClient


class TestInteractiveCLIAgent:
    """Test cases for InteractiveCLIAgent"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.reasoner = Mock()
        self.memory = ScratchPadMemory()
        self.jentic_client = Mock(spec=JenticClient)
        
        # Create a sample reasoning result
        self.sample_result = ReasoningResult(
            final_answer="Test answer",
            iterations=1,
            tool_calls=[{"tool_id": "test_tool", "tool_name": "Test Tool"}],
            success=True
        )
        
        self.reasoner.run.return_value = self.sample_result
    
    def test_handle_input(self):
        """Test input handling"""
        input_stream = StringIO("test goal\nquit\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        agent = InteractiveCLIAgent(
            reasoner=self.reasoner,
            memory=self.memory,
            inbox=inbox,
            jentic_client=self.jentic_client
        )
        
        result = agent.handle_input("  test input  ")
        assert result == "test input"
    
    @patch('builtins.print')
    def test_handle_output_success(self, mock_print):
        """Test successful output handling"""
        input_stream = StringIO("quit\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        agent = InteractiveCLIAgent(
            reasoner=self.reasoner,
            memory=self.memory,
            inbox=inbox,
            jentic_client=self.jentic_client
        )
        
        agent.handle_output(self.sample_result)
        
        # Verify print was called with success message
        mock_print.assert_called()
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("✅" in call for call in calls)
        assert any("Test answer" in call for call in calls)
    
    @patch('builtins.print')
    def test_handle_output_failure(self, mock_print):
        """Test failure output handling"""
        input_stream = StringIO("quit\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        agent = InteractiveCLIAgent(
            reasoner=self.reasoner,
            memory=self.memory,
            inbox=inbox,
            jentic_client=self.jentic_client
        )
        
        failure_result = ReasoningResult(
            final_answer="Failed to process",
            iterations=1,
            tool_calls=[],
            success=False,
            error_message="Test error"
        )
        
        agent.handle_output(failure_result)
        
        # Verify print was called with failure message
        mock_print.assert_called()
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("❌" in call for call in calls)
        assert any("Failed to process" in call for call in calls)
    
    def test_should_continue(self):
        """Test should_continue logic"""
        input_stream = StringIO("test goal\nquit\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        agent = InteractiveCLIAgent(
            reasoner=self.reasoner,
            memory=self.memory,
            inbox=inbox,
            jentic_client=self.jentic_client
        )
        
        # Initially not running
        assert agent.should_continue() is False
        
        # After starting, should continue
        agent._running = True
        assert agent.should_continue() is True
        
        # After stopping, should not continue
        agent.stop()
        assert agent.should_continue() is False
    
    @patch('builtins.print')
    def test_spin_single_goal(self, mock_print):
        """Test agent spin with single goal"""
        input_stream = StringIO("What's 2+2?\nquit\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        agent = InteractiveCLIAgent(
            reasoner=self.reasoner,
            memory=self.memory,
            inbox=inbox,
            jentic_client=self.jentic_client
        )
        
        agent.spin()
        
        # Verify reasoner was called
        self.reasoner.run.assert_called_once_with("What's 2+2?")
        
        # Verify output was handled
        mock_print.assert_called()
    
    @patch('builtins.print')
    @patch('sys.stderr')
    def test_spin_with_error(self, mock_stderr, mock_print):
        """Test agent spin with processing error"""
        input_stream = StringIO("error goal\nquit\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        agent = InteractiveCLIAgent(
            reasoner=self.reasoner,
            memory=self.memory,
            inbox=inbox,
            jentic_client=self.jentic_client
        )
        
        # Make reasoner raise an exception
        self.reasoner.run.side_effect = Exception("Test error")
        
        agent.spin()
        
        # Verify error was handled
        self.reasoner.run.assert_called_once_with("error goal")
    
    @patch('builtins.print')
    def test_spin_keyboard_interrupt(self, mock_print):
        """Test agent spin with keyboard interrupt"""
        input_stream = StringIO("test goal\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        agent = InteractiveCLIAgent(
            reasoner=self.reasoner,
            memory=self.memory,
            inbox=inbox,
            jentic_client=self.jentic_client
        )
        
        # Mock get_next_goal to raise KeyboardInterrupt
        def mock_get_goal():
            raise KeyboardInterrupt()
        
        inbox.get_next_goal = mock_get_goal
        
        # Should handle gracefully
        agent.spin()
        
        # Verify goodbye message was printed
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Interrupted by user" in call for call in calls)
