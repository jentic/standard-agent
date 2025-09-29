"""Simple OpenTelemetry setup for Standard Agent."""

from __future__ import annotations

import os
import base64
from enum import Enum
from typing import Optional, Tuple

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

class TelemetryTarget(str, Enum):
    """Supported telemetry export targets."""
    LANGFUSE = "langfuse"
    OTEL = "otel"


def setup_telemetry(service_name: str = "standard-agent", target: TelemetryTarget = TelemetryTarget.OTEL) -> Tuple[trace.Tracer, metrics.Meter]:
    """Setup OpenTelemetry tracing and metrics.
    
    Args:
        service_name: Name of the service for telemetry
        target: Explicit target for telemetry export (defaults to otel)
    
    Environment variables:
        - Langfuse: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
        - OTel: OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_HEADERS
    
    Raises:
        ValueError: If target is specified but required env vars are missing
    
    Returns:
        Tuple of (tracer, meter) ready for use
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
    
    # Setup metrics (basic, no exporter)
    metrics.set_meter_provider(MeterProvider())
    
    return trace.get_tracer(service_name), metrics.get_meter(service_name)


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
    
    missing_vars = []
    if not public_key:
        missing_vars.append("LANGFUSE_PUBLIC_KEY")
    if not secret_key:
        missing_vars.append("LANGFUSE_SECRET_KEY")
    if not host:
        missing_vars.append("LANGFUSE_HOST")
    
    if missing_vars:
        if required:
            raise ValueError(f"Langfuse target specified but missing environment variables: {', '.join(missing_vars)}")
        return None
    
    # Build Langfuse OTLP endpoint and auth
    endpoint = host.rstrip("/") + "/api/public/otel/v1/traces"
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers = {"authorization": f"Basic {token}"}
    
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _create_otel_exporter(required: bool = False) -> Optional[OTLPSpanExporter]:
    """Create standard OTel exporter using OTEL_* env vars."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    
    if not (endpoint or headers):
        if required:
            raise ValueError("OTel target specified but missing environment variables: OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_HEADERS")
        return None
    
    return OTLPSpanExporter()


def get_tracer(service_name: str = "standard-agent") -> trace.Tracer:
    """Get a tracer, setting up telemetry if needed."""
    if trace.get_tracer_provider() == trace.NoOpTracerProvider():
        setup_telemetry(service_name)
    return trace.get_tracer(service_name)


def get_meter(service_name: str = "standard-agent") -> metrics.Meter:
    """Get a meter, setting up telemetry if needed."""
    return metrics.get_meter(service_name)