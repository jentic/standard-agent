"""OpenTelemetry + Langfuse setup for Standard Agent evaluation."""

from __future__ import annotations

import os
from typing import Optional, Tuple

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider

_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None


def setup_telemetry(service_name: str = "standard-agent-eval") -> Tuple[trace.Tracer, metrics.Meter]:
    """Setup OpenTelemetry with Langfuse backend using env vars.

    Required envs (Langfuse Cloud or self-host):
      - LANGFUSE_PUBLIC_KEY
      - LANGFUSE_SECRET_KEY
      - LANGFUSE_HOST (e.g., https://cloud.langfuse.com)
    """
    global _tracer, _meter
    if _tracer is not None:
        return _tracer, _meter  # type: ignore[return-value]

    # Configure tracing provider
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Langfuse SDK does not expose an OpenTelemetry SpanProcessor in Python.
    # We'll emit spans via OpenTelemetry locally and (optionally) wrap calls with
    # the Langfuse SDK observe() decorator in instrumentation.
    lf_public = os.getenv("LANGFUSE_PUBLIC_KEY")
    lf_secret = os.getenv("LANGFUSE_SECRET_KEY")
    lf_host = os.getenv("LANGFUSE_HOST")
    if not (lf_public and lf_secret and lf_host):
        print("WARNING: LANGFUSE_* env vars not set; Langfuse export disabled")

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


