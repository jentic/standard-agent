"""Observability utilities for Standard Agent.

This module provides vendor-neutral tracing infrastructure:
- @observe decorator for automatic span creation
- OpenTelemetry setup with Langfuse integration
- Token usage tracking and aggregation
"""

from .observe import observe

# Lazy imports to avoid requiring OpenTelemetry
def setup_telemetry(*args, **kwargs):
    """Set up OpenTelemetry tracing."""
    from .otel_setup import setup_telemetry as _setup_telemetry
    return _setup_telemetry(*args, **kwargs)

def get_tracer(*args, **kwargs):
    """Get OpenTelemetry tracer."""
    from .otel_setup import get_tracer as _get_tracer
    return _get_tracer(*args, **kwargs)

# Import TelemetryTarget only when needed
def __getattr__(name):
    if name == "TelemetryTarget":
        from .otel_setup import TelemetryTarget
        return TelemetryTarget
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["observe", "setup_telemetry", "get_tracer", "TelemetryTarget"]
