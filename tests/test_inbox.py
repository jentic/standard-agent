"""
Unit tests for inbox implementations.
"""
from io import StringIO
from unittest.mock import patch
from ..inbox.cli_inbox import CLIInbox


class TestCLIInbox:
    """Test cases for CLIInbox"""
    
    def test_get_next_goal_from_stream(self):
        """Test getting goals from a stream"""
        input_stream = StringIO("goal 1\ngoal 2\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        assert inbox.get_next_goal() == "goal 1"
        assert inbox.get_next_goal() == "goal 2"
        assert inbox.get_next_goal() is None  # EOF
    
    def test_quit_commands(self):
        """Test quit commands stop the inbox"""
        input_stream = StringIO("goal 1\nquit\ngoal 2\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        assert inbox.get_next_goal() == "goal 1"
        assert inbox.get_next_goal() is None  # quit encountered
        assert inbox.get_next_goal() is None  # should remain closed
    
    def test_exit_commands(self):
        """Test various exit commands"""
        test_cases = ["exit", "EXIT", "q", "Q", "QUIT"]
        
        for exit_cmd in test_cases:
            input_stream = StringIO(f"goal 1\n{exit_cmd}\n")
            inbox = CLIInbox(input_stream=input_stream)
            
            assert inbox.get_next_goal() == "goal 1"
            assert inbox.get_next_goal() is None
    
    def test_empty_lines_ignored(self):
        """Test that empty lines are treated as quit"""
        input_stream = StringIO("goal 1\n\ngoal 2\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        assert inbox.get_next_goal() == "goal 1"
        assert inbox.get_next_goal() is None  # empty line treated as quit
    
    def test_whitespace_stripped(self):
        """Test that whitespace is stripped from goals"""
        input_stream = StringIO("  goal with spaces  \n")
        inbox = CLIInbox(input_stream=input_stream)
        
        assert inbox.get_next_goal() == "goal with spaces"
    
    def test_acknowledge_goal(self):
        """Test goal acknowledgment"""
        input_stream = StringIO("test goal\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        goal = inbox.get_next_goal()
        assert goal == "test goal"
        
        # Should not raise any exceptions
        inbox.acknowledge_goal(goal)
    
    def test_reject_goal(self):
        """Test goal rejection"""
        input_stream = StringIO("test goal\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        goal = inbox.get_next_goal()
        assert goal == "test goal"
        
        # Should not raise any exceptions
        with patch('sys.stderr'):  # Suppress stderr output in tests
            inbox.reject_goal(goal, "Test rejection reason")
    
    def test_has_goals(self):
        """Test has_goals functionality"""
        input_stream = StringIO("goal 1\nquit\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        assert inbox.has_goals() is True
        inbox.get_next_goal()  # goal 1
        assert inbox.has_goals() is True
        inbox.get_next_goal()  # quit
        assert inbox.has_goals() is False
    
    def test_close(self):
        """Test inbox close functionality"""
        input_stream = StringIO("goal 1\ngoal 2\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        assert inbox.has_goals() is True
        inbox.close()
        assert inbox.has_goals() is False
        assert inbox.get_next_goal() is None
    
    def test_goal_stream(self):
        """Test goal stream iterator"""
        input_stream = StringIO("goal 1\ngoal 2\nquit\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        goals = list(inbox.goal_stream())
        assert goals == ["goal 1", "goal 2"]
    
    def test_keyboard_interrupt_handling(self):
        """Test handling of KeyboardInterrupt"""
        input_stream = StringIO("goal 1\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        # Mock readline to raise KeyboardInterrupt
        def mock_readline():
            raise KeyboardInterrupt()
        
        input_stream.readline = mock_readline
        
        assert inbox.get_next_goal() is None
        assert inbox.has_goals() is False
    
    def test_eof_error_handling(self):
        """Test handling of EOFError"""
        input_stream = StringIO("goal 1\n")
        inbox = CLIInbox(input_stream=input_stream)
        
        # Mock readline to raise EOFError
        def mock_readline():
            raise EOFError()
        
        input_stream.readline = mock_readline
        
        assert inbox.get_next_goal() is None
        assert inbox.has_goals() is False
