"""
Evaluation framework package for Standard-Agent.

Provides:
- Pluggable tracing with an `observe` decorator
- Per-instance hooks to enable instrumentation without core edits
- Storage utilities for run records
- Aggregation helpers and a minimal runner (CLI)
"""

__all__ = [
    "storage",
    "metrics",
    "dataset",
]



