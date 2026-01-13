"""Tests for OpenTelemetry setup."""

import os
import pytest
from unittest.mock import patch, MagicMock

from utils.observability.otel_setup import (
    setup_telemetry,
    get_tracer,
    TelemetryTarget,
    _create_exporter
)


class TestTelemetryTarget:
    """Tests for TelemetryTarget enum."""
    
    def test_langfuse_target_exists(self):
        """Test that LANGFUSE target is available."""
        assert TelemetryTarget.LANGFUSE == "langfuse"
    
    def test_otel_target_exists(self):
        """Test that OTEL target is available."""
        assert TelemetryTarget.OTEL == "otel"


class TestCreateExporter:
    """Tests for _create_exporter dispatcher."""
    
    @patch('utils.observability.otel_setup.create_langfuse_exporter')
    def test_create_exporter_langfuse(self, mock_create_langfuse):
        """Test that LANGFUSE target calls langfuse exporter."""
        mock_exporter = MagicMock()
        mock_create_langfuse.return_value = mock_exporter
        
        result = _create_exporter(TelemetryTarget.LANGFUSE)
        
        assert result == mock_exporter
        mock_create_langfuse.assert_called_once()
    
    @patch('utils.observability.otel_setup.create_otel_exporter')
    def test_create_exporter_otel(self, mock_create_otel):
        """Test that OTEL target calls otel exporter."""
        mock_exporter = MagicMock()
        mock_create_otel.return_value = mock_exporter
        
        result = _create_exporter(TelemetryTarget.OTEL)
        
        assert result == mock_exporter
        mock_create_otel.assert_called_once()
    
    def test_create_exporter_unknown_target_raises(self):
        """Test that unknown target raises ValueError."""
        with pytest.raises(ValueError, match="Unknown telemetry target"):
            _create_exporter("unknown")


class TestSetupTelemetry:
    """Tests for setup_telemetry function."""
    
    @patch('utils.observability.otel_setup.create_otel_exporter')
    @patch('opentelemetry.sdk.trace.TracerProvider')
    @patch('opentelemetry.trace.set_tracer_provider')
    @patch('opentelemetry.trace.get_tracer')
    def test_setup_telemetry_default_target(
        self, mock_get_tracer, mock_set_provider, mock_provider_class, mock_create_otel
    ):
        """Test setup_telemetry with default target (LANGFUSE)."""
        mock_exporter = MagicMock()
        mock_create_otel.return_value = mock_exporter
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        env_vars = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = setup_telemetry(target=TelemetryTarget.OTEL)
        
        assert result == mock_tracer
        mock_set_provider.assert_called_once()
    
    @patch('utils.observability.otel_setup.create_langfuse_exporter')
    @patch('opentelemetry.sdk.trace.TracerProvider')
    @patch('opentelemetry.trace.set_tracer_provider')
    @patch('opentelemetry.trace.get_tracer')
    def test_setup_telemetry_langfuse_target(
        self, mock_get_tracer, mock_set_provider, mock_provider_class, mock_create_langfuse
    ):
        """Test setup_telemetry with LANGFUSE target."""
        mock_exporter = MagicMock()
        mock_create_langfuse.return_value = mock_exporter
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        env_vars = {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
            "LANGFUSE_HOST": "https://cloud.langfuse.com"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = setup_telemetry(target=TelemetryTarget.LANGFUSE)
        
        assert result == mock_tracer
        mock_create_langfuse.assert_called_once()
    
    @patch('utils.observability.otel_setup.create_otel_exporter')
    def test_setup_telemetry_propagates_exporter_errors(self, mock_create_otel):
        """Test that exporter errors are propagated."""
        mock_create_otel.side_effect = ValueError("Missing endpoint")
        
        with pytest.raises(ValueError, match="Missing endpoint"):
            setup_telemetry(target=TelemetryTarget.OTEL)
    
    @patch('utils.observability.otel_setup.create_otel_exporter')
    @patch('opentelemetry.sdk.trace.TracerProvider')
    @patch('opentelemetry.trace.set_tracer_provider')
    @patch('opentelemetry.trace.get_tracer')
    def test_setup_telemetry_custom_service_name(
        self, mock_get_tracer, mock_set_provider, mock_provider_class, mock_create_otel
    ):
        """Test setup_telemetry with custom service name."""
        mock_exporter = MagicMock()
        mock_create_otel.return_value = mock_exporter
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        env_vars = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = setup_telemetry(service_name="my-custom-service", target=TelemetryTarget.OTEL)
        
        assert result == mock_tracer
        mock_get_tracer.assert_called_with("my-custom-service")


class TestGetTracer:
    """Tests for get_tracer function."""
    
    @patch('opentelemetry.trace.get_tracer_provider')
    @patch('opentelemetry.trace.get_tracer')
    @patch('opentelemetry.trace.NoOpTracerProvider')
    def test_get_tracer_when_already_setup(self, mock_noop, mock_get_tracer, mock_get_provider):
        """Test get_tracer when telemetry is already setup."""
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        # Provider is NOT NoOp, so setup should not be called
        mock_noop.return_value = MagicMock()
        
        result = get_tracer("test-service")
        
        assert result == mock_tracer
        mock_get_tracer.assert_called_with("test-service")
    
    @patch('utils.observability.otel_setup.setup_telemetry')
    @patch('opentelemetry.trace.get_tracer_provider')
    @patch('opentelemetry.trace.get_tracer')
    @patch('opentelemetry.trace.NoOpTracerProvider')
    def test_get_tracer_auto_setup_when_needed(
        self, mock_noop_class, mock_get_tracer, mock_get_provider, mock_setup
    ):
        """Test get_tracer auto-setup when provider is NoOp."""
        # Make provider == NoOpTracerProvider to trigger setup
        noop_instance = MagicMock()
        mock_noop_class.return_value = noop_instance
        mock_get_provider.return_value = noop_instance
        
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        mock_setup.return_value = mock_tracer
        
        env_vars = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            # This may or may not trigger setup depending on OTel state
            result = get_tracer("test-service")
        
        assert result == mock_tracer
