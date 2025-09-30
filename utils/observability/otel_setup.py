"""Simple OpenTelemetry setup for Standard Agent."""

from __future__ import annotations

import os
import base64
from enum import Enum
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

class TelemetryTarget(str, Enum):
    """Supported telemetry export targets."""
    LANGFUSE = "langfuse"
    OTEL = "otel"


def setup_telemetry(service_name: str = "standard-agent", target: TelemetryTarget = TelemetryTarget.OTEL) -> trace.Tracer:
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
    if exporter:
        processor = BatchSpanProcessor(exporter)
        trace_provider.add_span_processor(processor)
    
    return trace.get_tracer(service_name)


def _create_exporter(target: TelemetryTarget) -> Optional[OTLPSpanExporter]:
    """Create OTLP exporter based on target."""
    if target == TelemetryTarget.LANGFUSE:
        return _create_langfuse_exporter(required=True)
    elif target == TelemetryTarget.OTEL:
        return _create_otel_exporter(required=True)
    else:
        raise ValueError(f"Unknown telemetry target: {target}")


def _create_langfuse_exporter(required: bool = False) -> Optional[OTLPSpanExporter]:
    """Create exporter for Langfuse using LANGFUSE_* env vars."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") 
    host = os.getenv("LANGFUSE_HOST")
    
    # Check for missing required variables
    missing = [name for name, val in [("LANGFUSE_PUBLIC_KEY", public_key), ("LANGFUSE_SECRET_KEY", secret_key), ("LANGFUSE_HOST", host)] if not val]
    
    if missing:
        if required:
            raise ValueError(f"Langfuse target specified but missing environment variables: {', '.join(missing)}")
        return None
    
    # Build Langfuse OTLP endpoint and auth
    endpoint = host.rstrip("/") + "/api/public/otel/v1/traces"
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers = {"authorization": f"Basic {token}"}
    
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _create_otel_exporter(required: bool = False) -> Optional[OTLPSpanExporter]:
    """Create standard OTel exporter using OTEL_* env vars.
    
    The OTLPSpanExporter reads OTEL_EXPORTER_OTLP_* env vars automatically.
    We only validate they exist if this exporter is explicitly required.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    
    if required and not endpoint:
        raise ValueError("OTel target specified but missing environment variable: OTEL_EXPORTER_OTLP_ENDPOINT")
    
    # OTLPSpanExporter() reads from OTEL_EXPORTER_OTLP_* env vars automatically
    return OTLPSpanExporter() if endpoint else None


def get_tracer(service_name: str = "standard-agent") -> trace.Tracer:
    """Get a tracer, setting up telemetry if needed."""
    if trace.get_tracer_provider() == trace.NoOpTracerProvider():
        setup_telemetry(service_name)
    return trace.get_tracer(service_name)