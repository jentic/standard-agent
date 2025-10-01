"""Telemetry exporters for different observability backends.

This module provides pluggable exporter implementations for Standard Agent.
Each exporter knows how to configure and create an OTLP span exporter for
a specific backend (Langfuse, Datadog, Jaeger, etc.).

To add a new exporter:
1. Create a new file in this directory (e.g., `mybackend.py`)
2. Implement a `create_mybackend_exporter(required: bool) -> Optional[OTLPSpanExporter]`
3. Export it from this __init__.py
4. Add your target to TelemetryTarget enum in otel_setup.py
5. Update the dispatcher in otel_setup._create_exporter()

See langfuse.py or otel.py for examples.
"""

from .langfuse import create_langfuse_exporter
from .otel import create_otel_exporter

__all__ = ["create_langfuse_exporter", "create_otel_exporter"]
