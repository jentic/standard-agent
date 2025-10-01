"""Langfuse OTLP exporter with automatic endpoint and authentication setup.

This exporter configures OpenTelemetry to send traces to Langfuse Cloud or
self-hosted Langfuse instances using the OTLP protocol.

Environment Variables:
    LANGFUSE_PUBLIC_KEY: Your Langfuse public API key
    LANGFUSE_SECRET_KEY: Your Langfuse secret API key
    LANGFUSE_HOST: Langfuse instance URL (e.g., https://cloud.langfuse.com)

Example:
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://cloud.langfuse.com
"""

from __future__ import annotations

import os
import base64

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def create_langfuse_exporter() -> OTLPSpanExporter:
    """Create an OTLP exporter configured for Langfuse.
    
    Returns:
        Configured OTLPSpanExporter
    
    Raises:
        ValueError: If required environment variables are missing
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") 
    host = os.getenv("LANGFUSE_HOST")
    
    # Check for missing required variables
    missing = [name for name, val in [("LANGFUSE_PUBLIC_KEY", public_key), ("LANGFUSE_SECRET_KEY", secret_key), ("LANGFUSE_HOST", host)] if not val]
    
    if missing:
        raise ValueError(
            f"Langfuse exporter requires environment variables: {', '.join(missing)}"
        )
    
    # Build Langfuse OTLP endpoint and auth
    endpoint = host.rstrip("/") + "/api/public/otel/v1/traces"
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers = {"authorization": f"Basic {token}"}
    
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)
