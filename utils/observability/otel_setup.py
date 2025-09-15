"""OpenTelemetry setup for Standard Agent evaluation (OTLP exporter via env)."""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict
import base64

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None


def setup_telemetry(service_name: str = "standard-agent-eval") -> Tuple[trace.Tracer, metrics.Meter]:
    """Setup OpenTelemetry with OTLP exporter configured via env.

    Example (Langfuse Cloud):
      - OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/otlp
      - OTEL_EXPORTER_OTLP_HEADERS=authorization=Bearer <base64(PUBLIC_KEY:SECRET_KEY)>
      - OTEL_SERVICE_NAME=standard-agent-eval (or pass service_name)
    """
    global _tracer, _meter
    if _tracer is not None:
        return _tracer, _meter  # type: ignore[return-value]

    # Configure tracing provider with resource (service.name)
    resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", service_name)})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Attach OTLP exporter (env-driven config or LANGFUSE_* convenience)
    try:
        endpoint_env = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")

        exporter: Optional[OTLPSpanExporter] = None
        if endpoint_env or headers_env:
            # Respect explicit OTEL_* env configuration
            exporter = OTLPSpanExporter()
        else:
            # Convenience: if LANGFUSE_* present, derive OTLP endpoint + headers
            lf_public = os.getenv("LANGFUSE_PUBLIC_KEY")
            lf_secret = os.getenv("LANGFUSE_SECRET_KEY")
            lf_host = os.getenv("LANGFUSE_HOST")
            if lf_public and lf_secret and lf_host:
                # Langfuse OTLP HTTP endpoint must include /v1/traces for HTTP exporter
                endpoint = lf_host.rstrip("/") + "/api/public/otel/v1/traces"
                token = base64.b64encode(f"{lf_public}:{lf_secret}".encode("utf-8")).decode("utf-8")
                headers: Dict[str, str] = {"authorization": f"Basic {token}"}
                exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)

        if exporter is None:
            # Fall back to default (no export) if nothing configured
            exporter = OTLPSpanExporter()

        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
    except Exception:
        # If misconfigured, spans won't be exported; keep tracing in-memory.
        pass

    # Initialize metrics provider (no exporter wired for now)
    metrics.set_meter_provider(MeterProvider())

    _tracer = trace.get_tracer(service_name)
    _meter = metrics.get_meter(service_name)
    return _tracer, _meter


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        tracer, _ = setup_telemetry()
        return tracer
    return _tracer


def get_meter() -> metrics.Meter:
    if _meter is None:
        _, meter = setup_telemetry()
        return meter
    return _meter


