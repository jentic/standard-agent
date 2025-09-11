from __future__ import annotations

from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional


def _default_span_name(fn: Callable[..., Any]) -> str:
    """Compute a readable default span name from the function's qualname.

    Example: module.Class.method or module.function
    """
    module = getattr(fn, "__module__", None) or ""
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "anonymous"))
    if module:
        return f"{module}.{qualname}"
    return qualname


def observe(_fn: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
    """Minimal, vendor-neutral tracing decorator.

    Usage:
      @observe
      def fn(...): ...

      @observe()
      async def afn(...): ...

    - No arguments required. It auto-names spans using module + qualname.
    - Works whether OpenTelemetry is installed/configured or not.
      If OTel is unavailable, it's a no-op with near-zero overhead.
    - Records exceptions to the span and sets error status when OTel is present.
    """

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = _default_span_name(fn)

        if iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    # Lazy import to avoid hard dependency when observability is not installed
                    from opentelemetry import trace  # type: ignore
                    from opentelemetry.trace import Status, StatusCode  # type: ignore

                    tracer = trace.get_tracer("standard-agent")
                    with tracer.start_as_current_span(span_name) as span:  # type: ignore[assignment]
                        try:
                            return await fn(*args, **kwargs)
                        except Exception as e:  # pragma: no cover - passthrough with metadata
                            try:
                                span.record_exception(e)  # type: ignore[attr-defined]
                                span.set_status(Status(StatusCode.ERROR))  # type: ignore[attr-defined]
                            finally:
                                pass
                            raise
                except Exception:
                    # If OpenTelemetry isn't available or any tracing issue occurs, just run the function
                    return await fn(*args, **kwargs)

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                from opentelemetry import trace  # type: ignore
                from opentelemetry.trace import Status, StatusCode  # type: ignore

                tracer = trace.get_tracer("standard-agent")
                with tracer.start_as_current_span(span_name) as span:  # type: ignore[assignment]
                    try:
                        return fn(*args, **kwargs)
                    except Exception as e:  # pragma: no cover - passthrough with metadata
                        try:
                            span.record_exception(e)  # type: ignore[attr-defined]
                            span.set_status(Status(StatusCode.ERROR))  # type: ignore[attr-defined]
                        finally:
                            pass
                        raise
            except Exception:
                # No OTel at runtime (or misconfigured): act as a no-op decorator
                return fn(*args, **kwargs)

        return sync_wrapper

    # Support both @observe and @observe() forms
    if callable(_fn):
        return _decorate(_fn)
    return _decorate


