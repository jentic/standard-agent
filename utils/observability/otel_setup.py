"""Simple OpenTelemetry setup for Standard Agent."""

from __future__ import annotations

import os
from enum import Enum

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from .exporters import create_langfuse_exporter, create_otel_exporter

class TelemetryTarget(str, Enum):
    """Supported telemetry export targets."""
    LANGFUSE = "langfuse"
    OTEL = "otel"


def setup_telemetry(service_name: str = "standard-agent", target: TelemetryTarget = TelemetryTarget.LANGFUSE) -> trace.Tracer:
    """Setup OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service for telemetry
        target: Explicit target for telemetry export (defaults to otel)
    
    Environment variables:
        - Langfuse: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
        - OTel: OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_HEADERS
    
    Raises:
        ValueError: If target is specified but required env vars are missing
    
    Returns:
        Tracer ready for use
    """
    # Setup tracing
    resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", service_name)})
    trace_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(trace_provider)
    
    # Setup exporter based on target
    exporter = _create_exporter(target)
    processor = BatchSpanProcessor(exporter)
    trace_provider.add_span_processor(processor)
    
    return trace.get_tracer(service_name)


def _create_exporter(target: TelemetryTarget) -> OTLPSpanExporter:
    """Create OTLP exporter based on target."""
    if target == TelemetryTarget.LANGFUSE:
        return create_langfuse_exporter()
    elif target == TelemetryTarget.OTEL:
        return create_otel_exporter()
    else:
        raise ValueError(f"Unknown telemetry target: {target}")


def get_tracer(service_name: str = "standard-agent") -> trace.Tracer:
    """Get a tracer, setting up telemetry if needed."""
    if trace.get_tracer_provider() == trace.NoOpTracerProvider():
        setup_telemetry(service_name)
    return trace.get_tracer(service_name)