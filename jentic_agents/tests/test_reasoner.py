"""
Unit tests for StandardReasoner.
"""
from unittest.mock import Mock, patch

from ..reasoners.standard_reasoner import StandardReasoner
from ..platform.jentic_client import JenticClient
from ..utils.llm import BaseLLM


class TestStandardReasoner:
    """Test cases for StandardReasoner"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.jentic_client = Mock(spec=JenticClient)
        self.llm = Mock(spec=BaseLLM)
        
        # Mock LLM response
        self.llm.chat.return_value = "Test response"
        
        self.reasoner = StandardReasoner(
            jentic_client=self.jentic_client,
            llm=self.llm,
            model="gpt-3.5-turbo"  # Use cheaper model for tests
        )
    
    def test_init(self):
        """Test reasoner initialization"""
        assert self.reasoner.jentic_client == self.jentic_client
        assert self.reasoner.llm == self.llm
        assert self.reasoner.model == "gpt-3.5-turbo"
    
    @patch('jentic_agents.reasoners.standard_reasoner.LiteLLMChatLLM')
    def test_init_default_llm(self, mock_llm):
        """Test reasoner initialization with default LLM client"""
        StandardReasoner(jentic_client=self.jentic_client)
        mock_llm.assert_called_once()
    
    def test_plan(self):
        """Test plan generation"""
        context = {"iteration": 1, "observations": [], "failed_attempts": []}
        plan = self.reasoner.plan("Test goal", context)
        
        assert plan == "Test response"
        self.llm.chat.assert_called_once()
    
    def test_select_tool_empty_list(self):
        """Test tool selection with empty tool list"""
        result = self.reasoner.select_tool("test plan", [])
        assert result is None
    
    def test_select_tool_single_tool(self):
        """Test tool selection with single tool"""
        tools = [{"id": "tool1", "name": "Tool 1"}]
        result = self.reasoner.select_tool("test plan", tools)
        assert result == tools[0]
    
    def test_select_tool_multiple_tools(self):
        """Test tool selection with multiple tools"""
        tools = [
            {"id": "tool1", "name": "Tool 1"},
            {"id": "tool2", "name": "Tool 2"}
        ]
        
        # Mock LLM to return tool1
        self.llm.chat.return_value = "tool1"
        
        result = self.reasoner.select_tool("test plan", tools)
        assert result == tools[0]
    
    def test_select_tool_none_response(self):
        """Test tool selection when LLM responds with NONE"""
        tools = [
            {"id": "tool1", "name": "Tool 1"},
            {"id": "tool2", "name": "Tool 2"}
        ]
        
        # Mock LLM to return NONE
        self.llm.chat.return_value = "NONE"
        
        result = self.reasoner.select_tool("test plan", tools)
        assert result is None
    
    def test_act_no_parameters(self):
        """Test action with tool that has no parameters"""
        tool = {"name": "Test Tool", "parameters": {}}
        result = self.reasoner.act(tool, "test plan")
        assert result == {}
    
    def test_act_with_parameters(self):
        """Test action with tool that has parameters"""
        tool = {
            "name": "Test Tool",
            "parameters": {"param1": "string"}
        }
        
        # Mock LLM to return valid JSON
        self.llm.chat.return_value = '{"param1": "value1"}'
        
        result = self.reasoner.act(tool, "test plan")
        assert result == {"param1": "value1"}
    
    def test_act_invalid_json(self):
        """Test action when LLM returns invalid JSON"""
        tool = {
            "name": "Test Tool", 
            "parameters": {"param1": "string"}
        }
        
        # Mock LLM to return invalid JSON
        self.llm.chat.return_value = "invalid json"
        
        result = self.reasoner.act(tool, "test plan")
        assert result == {}
    
    def test_observe_success(self):
        """Test observation of successful action"""
        action_result = {"status": "success", "result": "Test result"}
        observation = self.reasoner.observe(action_result)
        
        assert "successfully" in observation
        assert "Test result" in observation
    
    def test_observe_failure(self):
        """Test observation of failed action"""
        action_result = {"status": "error", "error": "Test error"}
        observation = self.reasoner.observe(action_result)
        
        assert "failed" in observation
        assert "Test error" in observation
    
    def test_evaluate_no_observations(self):
        """Test evaluation with no observations"""
        result = self.reasoner.evaluate("test goal", [])
        assert result is False
    
    def test_evaluate_with_observations(self):
        """Test evaluation with observations"""
        # Mock LLM to return YES
        self.llm.chat.return_value = "YES"
        
        result = self.reasoner.evaluate("test goal", ["observation 1"])
        assert result is True
    
    def test_reflect(self):
        """Test reflection functionality"""
        reflection = self.reasoner.reflect(
            "test goal",
            ["observation 1"], 
            ["failed attempt 1"]
        )
        
        assert reflection == "Test response"
        self.llm.chat.assert_called()
    
    def test_run_max_iterations_reached(self):
        """Test running reasoner when max iterations is reached"""
        # Setup mocks to always fail tool selection
        self.jentic_client.search.return_value = []
        
        # Mock LLM to always return a plan
        self.llm.chat.return_value = "Try to find a tool"
        
        result = self.reasoner.run("Impossible goal", max_iterations=2)
        
        assert result.success is False
        assert result.iterations == 2
        assert len(result.tool_calls) == 0
        assert "iteration limit" in result.final_answer
