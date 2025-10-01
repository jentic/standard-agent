"""Tests for @observe decorator."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from utils.observability.observe import observe, _safe_preview, SECRET_REDACT_KEYS


class TestSafePreview:
    """Tests for _safe_preview function."""
    
    def test_preview_primitives(self):
        """Test that primitives are returned as-is."""
        assert _safe_preview(None) is None
        assert _safe_preview(True) is True
        assert _safe_preview(42) == 42
        assert _safe_preview(3.14) == 3.14
    
    def test_preview_short_string(self):
        """Test that short strings are unchanged."""
        short_str = "hello world"
        assert _safe_preview(short_str) == short_str
    
    def test_preview_long_string_truncated(self):
        """Test that long strings are truncated with ellipsis."""
        long_str = "x" * 1000
        result = _safe_preview(long_str, max_len=100)
        
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")
        assert result[:100] == long_str[:100]
    
    def test_preview_dict_redacts_secrets(self):
        """Test that secret keys are redacted in dicts."""
        data = {
            "api_key": "secret123",
            "username": "john",
            "password": "hunter2"
        }
        
        result = _safe_preview(data)
        
        assert result["api_key"] == "<redacted>"
        assert result["password"] == "<redacted>"
        assert result["username"] == "john"
    
    def test_preview_dict_handles_various_secret_formats(self):
        """Test that various secret key formats are redacted."""
        data = {
            "api_key": "secret",
            "api-key": "secret",
            "apikey": "secret",
            "API_KEY": "secret",
            "access_token": "secret",
            "accessToken": "secret",
            "client_secret": "secret",
            "normal_field": "visible"
        }
        
        result = _safe_preview(data)
        
        # All secret variations should be redacted
        assert result["api_key"] == "<redacted>"
        assert result["api-key"] == "<redacted>"
        assert result["apikey"] == "<redacted>"
        assert result["API_KEY"] == "<redacted>"
        assert result["access_token"] == "<redacted>"
        assert result["accessToken"] == "<redacted>"
        assert result["client_secret"] == "<redacted>"
        # Normal field should be visible
        assert result["normal_field"] == "visible"
    
    def test_preview_list_truncates_with_marker(self):
        """Test that long lists are truncated with ellipsis marker."""
        long_list = list(range(50))
        result = _safe_preview(long_list)
        
        assert len(result) == 21  # 20 items + "..."
        assert result[-1] == "..."
        assert result[:20] == long_list[:20]
    
    def test_preview_dict_truncates_with_marker(self):
        """Test that large dicts are truncated with marker."""
        large_dict = {f"key{i}": f"val{i}" for i in range(50)}
        result = _safe_preview(large_dict)
        
        assert "..." in result
        assert "30 more keys" in result["..."]
        assert len([k for k in result.keys() if k != "..."]) == 20
    
    def test_preview_nested_structures(self):
        """Test that nested structures are previewed recursively."""
        data = {
            "user": {
                "name": "john",
                "password": "secret"
            },
            "items": [1, 2, 3]
        }
        
        result = _safe_preview(data)
        
        assert result["user"]["name"] == "john"
        assert result["user"]["password"] == "<redacted>"
        assert result["items"] == [1, 2, 3]
    
    def test_preview_dataclass(self):
        """Test that dataclasses are converted and previewed."""
        @dataclass
        class User:
            name: str
            api_key: str
        
        user = User(name="john", api_key="secret123")
        result = _safe_preview(user)
        
        assert result["name"] == "john"
        assert result["api_key"] == "<redacted>"


class TestObserveDecorator:
    """Tests for @observe decorator."""
    
    def test_observe_without_otel_runs_function(self):
        """Test that function runs normally when OpenTelemetry not available."""
        @observe
        def my_func(x, y):
            return x + y
        
        result = my_func(2, 3)
        assert result == 5
    
    def test_observe_with_llm_flag(self):
        """Test that @observe(llm=True) works."""
        @observe(llm=True)
        def llm_call(messages):
            return {"text": "response"}
        
        result = llm_call([{"role": "user", "content": "hello"}])
        assert result["text"] == "response"
    
    def test_observe_with_root_flag(self):
        """Test that @observe(root=True) works."""
        @observe(root=True)
        def root_func():
            return "done"
        
        result = root_func()
        assert result == "done"
    
    def test_observe_preserves_function_name(self):
        """Test that decorator preserves function __name__ and __doc__."""
        @observe
        def my_function():
            """My docstring."""
            return 42
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."
    
    def test_observe_with_exceptions(self):
        """Test that exceptions are propagated correctly."""
        @observe
        def failing_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            failing_func()
    
    @patch('opentelemetry.trace.get_tracer')
    def test_observe_creates_span(self, mock_get_tracer):
        """Test that observe creates a span when OTel is available."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer
        
        @observe
        def test_func():
            return "result"
        
        result = test_func()
        
        assert result == "result"
        mock_tracer.start_as_current_span.assert_called_once()
        # Span name should include module and function name
        span_name = mock_tracer.start_as_current_span.call_args[0][0]
        assert "test_func" in span_name
    
    @patch('opentelemetry.trace.get_tracer')
    def test_observe_sets_attributes(self, mock_get_tracer):
        """Test that observe sets span attributes."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer
        
        @observe
        def test_func(x):
            return x * 2
        
        result = test_func(21)
        
        assert result == 42
        # Should set input, output, and duration_ms attributes
        assert mock_span.set_attribute.called
        call_args = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        assert any("input" in str(arg) or "output" in str(arg) or "duration_ms" in str(arg) for arg in call_args)
