"""
Unit tests for JenticClient platform wrapper.
"""
import pytest

from ..platform.jentic_client import JenticClient


class TestJenticClient:
    """Test cases for JenticClient"""
    
    def test_init(self):
        """Test client initialization"""
        client = JenticClient(api_key="test_key")
        assert client.api_key == "test_key"
    
    def test_init_no_key(self):
        """Test client initialization without API key"""
        client = JenticClient()
        assert client.api_key is None
    
    def test_search(self):
        """Test search functionality returns expected mock data"""
        client = JenticClient()
        results = client.search("test query", top_k=3)
        
        assert isinstance(results, list)
        assert len(results) == 2  # Mock returns 2 tools
        assert all(isinstance(result, dict) for result in results)
        assert all("id" in result and "name" in result for result in results)
    
    def test_load_echo_tool(self):
        """Test loading echo tool definition"""
        client = JenticClient()
        tool = client.load("echo_tool_001")
        
        assert tool["id"] == "echo_tool_001"
        assert tool["name"] == "Echo Tool"
        assert "parameters" in tool
        assert tool["executable"] is True
    
    def test_load_ping_tool(self):
        """Test loading ping tool definition"""
        client = JenticClient()
        tool = client.load("ping_tool_001")
        
        assert tool["id"] == "ping_tool_001"
        assert tool["name"] == "Ping Tool"
        assert tool["parameters"] == {}
        assert tool["executable"] is True
    
    def test_load_unknown_tool(self):
        """Test loading unknown tool raises ValueError"""
        client = JenticClient()
        
        with pytest.raises(ValueError, match="Tool not found: unknown_tool"):
            client.load("unknown_tool")
    
    def test_execute_echo_tool(self):
        """Test executing echo tool"""
        client = JenticClient()
        result = client.execute("echo_tool_001", {"message": "Hello, World!"})
        
        assert result["status"] == "success"
        assert result["result"] == "Echo: Hello, World!"
        assert "execution_time" in result
    
    def test_execute_ping_tool(self):
        """Test executing ping tool"""
        client = JenticClient()
        result = client.execute("ping_tool_001", {})
        
        assert result["status"] == "success"
        assert result["result"] == "pong"
        assert "execution_time" in result
    
    def test_execute_unknown_tool(self):
        """Test executing unknown tool raises ValueError"""
        client = JenticClient()
        
        with pytest.raises(ValueError, match="Tool not found: unknown_tool"):
            client.execute("unknown_tool", {})
