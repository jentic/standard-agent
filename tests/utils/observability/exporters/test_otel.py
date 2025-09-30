"""Tests for generic OTLP exporter."""

import os
import pytest
from unittest.mock import patch

from utils.observability.exporters.otel import create_otel_exporter


class TestOTelExporter:
    """Tests for create_otel_exporter function."""
    
    def test_create_exporter_with_valid_endpoint(self):
        """Test exporter creation with OTEL_EXPORTER_OTLP_ENDPOINT set."""
        env_vars = {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exporter = create_otel_exporter()
            
            assert exporter is not None
            # OTLPSpanExporter reads from env vars internally
            assert hasattr(exporter, '_endpoint')
    
    def test_create_exporter_with_endpoint_and_headers(self):
        """Test exporter creation with both endpoint and headers."""
        env_vars = {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
            "OTEL_EXPORTER_OTLP_HEADERS": "authorization=Bearer token123"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exporter = create_otel_exporter()
            
            assert exporter is not None
    
    def test_missing_endpoint_raises_error(self):
        """Test that missing OTEL_EXPORTER_OTLP_ENDPOINT raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_otel_exporter()
            
            assert "OTEL_EXPORTER_OTLP_ENDPOINT" in str(exc_info.value)
    
    def test_empty_endpoint_raises_error(self):
        """Test that empty OTEL_EXPORTER_OTLP_ENDPOINT raises ValueError."""
        env_vars = {
            "OTEL_EXPORTER_OTLP_ENDPOINT": ""
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_otel_exporter()
            
            assert "OTEL_EXPORTER_OTLP_ENDPOINT" in str(exc_info.value)
    
    def test_various_endpoint_formats(self):
        """Test that various endpoint formats are accepted."""
        endpoints = [
            "http://localhost:4318",
            "https://api.honeycomb.io",
            "http://jaeger:4318/v1/traces",
            "https://tempo.example.com:443",
        ]
        
        for endpoint in endpoints:
            env_vars = {"OTEL_EXPORTER_OTLP_ENDPOINT": endpoint}
            
            with patch.dict(os.environ, env_vars, clear=True):
                exporter = create_otel_exporter()
                assert exporter is not None
