"""Generic OTLP exporter using standard OpenTelemetry environment variables.

This exporter works with any OTLP-compatible backend (Jaeger, Tempo, etc.)
by using the standard OpenTelemetry environment variables for configuration.

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    OTEL_EXPORTER_OTLP_HEADERS: Optional headers (e.g., for authentication)
    OTEL_EXPORTER_OTLP_PROTOCOL: Protocol (http/protobuf or grpc, defaults to http/protobuf)

Example:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer token123"

For backends requiring authentication (like Honeycomb), set headers:
    export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=your-api-key"
"""

from __future__ import annotations

import os
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def create_otel_exporter() -> OTLPSpanExporter:
    """Create a generic OTLP exporter using standard OpenTelemetry env vars.

    The OTLPSpanExporter automatically reads OTEL_EXPORTER_OTLP_* environment
    variables for configuration (ENDPOINT, HEADERS, PROTOCOL, TIMEOUT, etc.).

    Returns:
        Configured OTLPSpanExporter

    Raises:
        ValueError: If OTEL_EXPORTER_OTLP_ENDPOINT is not configured
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not endpoint:
        raise ValueError(
            "OTLP exporter requires OTEL_EXPORTER_OTLP_ENDPOINT environment variable"
        )

    # OTLPSpanExporter() reads from OTEL_EXPORTER_OTLP_* env vars automatically
    return OTLPSpanExporter()
