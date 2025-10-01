"""Tests for Langfuse exporter."""

import os
import pytest
from unittest.mock import patch

from utils.observability.exporters.langfuse import create_langfuse_exporter


class TestLangfuseExporter:
    """Tests for create_langfuse_exporter function."""
    
    def test_create_exporter_with_valid_env_vars(self):
        """Test exporter creation with all required env vars set."""
        env_vars = {
            "LANGFUSE_PUBLIC_KEY": "pk-test-12345",
            "LANGFUSE_SECRET_KEY": "sk-test-67890",
            "LANGFUSE_HOST": "https://cloud.langfuse.com"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exporter = create_langfuse_exporter()
            
            assert exporter is not None
            assert exporter._endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
            assert "authorization" in exporter._headers
            assert exporter._headers["authorization"].startswith("Basic ")
    
    def test_create_exporter_strips_trailing_slash(self):
        """Test that trailing slash is removed from host."""
        env_vars = {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
            "LANGFUSE_HOST": "https://cloud.langfuse.com/"  # Note trailing slash
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exporter = create_langfuse_exporter()
            
            assert exporter._endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    
    def test_missing_public_key_raises_error(self):
        """Test that missing LANGFUSE_PUBLIC_KEY raises ValueError."""
        env_vars = {
            "LANGFUSE_SECRET_KEY": "sk-test",
            "LANGFUSE_HOST": "https://cloud.langfuse.com"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_langfuse_exporter()
            
            assert "LANGFUSE_PUBLIC_KEY" in str(exc_info.value)
    
    def test_missing_secret_key_raises_error(self):
        """Test that missing LANGFUSE_SECRET_KEY raises ValueError."""
        env_vars = {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_HOST": "https://cloud.langfuse.com"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_langfuse_exporter()
            
            assert "LANGFUSE_SECRET_KEY" in str(exc_info.value)
    
    def test_missing_host_raises_error(self):
        """Test that missing LANGFUSE_HOST raises ValueError."""
        env_vars = {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_langfuse_exporter()
            
            assert "LANGFUSE_HOST" in str(exc_info.value)
    
    def test_missing_all_vars_lists_all_in_error(self):
        """Test that missing all vars lists all of them in error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_langfuse_exporter()
            
            error_msg = str(exc_info.value)
            assert "LANGFUSE_PUBLIC_KEY" in error_msg
            assert "LANGFUSE_SECRET_KEY" in error_msg
            assert "LANGFUSE_HOST" in error_msg
    
    def test_basic_auth_header_format(self):
        """Test that authorization header uses Basic auth with base64 encoding."""
        env_vars = {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
            "LANGFUSE_HOST": "https://test.com"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exporter = create_langfuse_exporter()
            
            # Check that authorization header exists and is properly formatted
            auth_header = exporter._headers["authorization"]
            assert auth_header.startswith("Basic ")
            
            # The token should be base64 encoded "pk-test:sk-test"
            import base64
            expected_token = base64.b64encode(b"pk-test:sk-test").decode()
            assert auth_header == f"Basic {expected_token}"
