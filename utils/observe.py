from __future__ import annotations

from functools import wraps
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Optional
import time
import os
import hashlib


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

        def _maybe_truncate(value: Any, limit: int = 200) -> Any:
            if isinstance(value, str):
                return value if len(value) <= limit else value[:limit] + "…"
            return value

        if iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    # Lazy import to avoid hard dependency when observability is not installed
                    from opentelemetry import trace  # type: ignore
                    from opentelemetry.trace import Status, StatusCode  # type: ignore

                    tracer = trace.get_tracer("standard-agent")
                    t0 = time.perf_counter()
                    with tracer.start_as_current_span(span_name) as span:  # type: ignore[assignment]
                        try:
                            # Best-effort: bind args to capture a 'goal' param if present
                            try:
                                bound = signature(fn).bind_partial(*args, **kwargs)
                                goal_val = bound.arguments.get("goal")
                                if isinstance(goal_val, str):
                                    # Record full goal as requested (no truncation)
                                    span.set_attribute("sa.goal", goal_val)  # type: ignore[attr-defined]
                            except Exception:
                                pass

                            result = await fn(*args, **kwargs)

                            # Attach basic outcome fields if present
                            try:
                                if hasattr(result, "success"):
                                    span.set_attribute("sa.result_success", bool(getattr(result, "success")))  # type: ignore[attr-defined]
                                if hasattr(result, "iterations"):
                                    span.set_attribute("sa.result_iterations", int(getattr(result, "iterations")))  # type: ignore[attr-defined]
                                if hasattr(result, "final_answer"):
                                    fa = getattr(result, "final_answer")
                                    if isinstance(fa, str):
                                        # Always store a short preview
                                        span.set_attribute("sa.result_final_preview", _maybe_truncate(fa))  # type: ignore[attr-defined]
                                        # Optionally store a bounded full result
                                        if os.getenv("SA_OBSERVE_FULL_RESULT", "").strip().lower() in ("1", "true", "yes", "on"):  # pragma: no cover
                                            cap = 8192
                                            truncated = fa if len(fa) <= cap else fa[:cap]
                                            span.set_attribute("sa.result_final_full", truncated)  # type: ignore[attr-defined]
                                            span.set_attribute("sa.result_final_full_len", len(fa))  # type: ignore[attr-defined]
                                            span.set_attribute("sa.result_final_full_truncated", len(fa) > cap)  # type: ignore[attr-defined]
                                            try:
                                                digest = hashlib.sha256(fa.encode("utf-8")).hexdigest()
                                                span.set_attribute("sa.result_final_sha256", digest)  # type: ignore[attr-defined]
                                            except Exception:
                                                pass
                            except Exception:
                                pass

                            return result
                        except Exception as e:  # pragma: no cover - passthrough with metadata
                            try:
                                span.record_exception(e)  # type: ignore[attr-defined]
                                span.set_status(Status(StatusCode.ERROR))  # type: ignore[attr-defined]
                            finally:
                                pass
                            raise
                        finally:
                            try:
                                duration_ms = int((time.perf_counter() - t0) * 1000)
                                span.set_attribute("sa.duration_ms", duration_ms)  # type: ignore[attr-defined]
                            except Exception:
                                pass
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
                t0 = time.perf_counter()
                with tracer.start_as_current_span(span_name) as span:  # type: ignore[assignment]
                    try:
                        # Best-effort: bind args to capture a 'goal' param if present
                        try:
                            bound = signature(fn).bind_partial(*args, **kwargs)
                            goal_val = bound.arguments.get("goal")
                            if isinstance(goal_val, str):
                                # Record full goal as requested (no truncation)
                                span.set_attribute("sa.goal", goal_val)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                        result = fn(*args, **kwargs)

                        # Attach basic outcome fields if present
                        try:
                            if hasattr(result, "success"):
                                span.set_attribute("sa.result_success", bool(getattr(result, "success")))  # type: ignore[attr-defined]
                            if hasattr(result, "iterations"):
                                span.set_attribute("sa.result_iterations", int(getattr(result, "iterations")))  # type: ignore[attr-defined]
                            if hasattr(result, "final_answer"):
                                fa = getattr(result, "final_answer")
                                if isinstance(fa, str):
                                    # Always store a short preview
                                    span.set_attribute("sa.result_final_preview", _maybe_truncate(fa))  # type: ignore[attr-defined]
                                    # Optionally store a bounded full result
                                    if os.getenv("SA_OBSERVE_FULL_RESULT", "").strip().lower() in ("1", "true", "yes", "on"):  # pragma: no cover
                                        cap = 8192
                                        truncated = fa if len(fa) <= cap else fa[:cap]
                                        span.set_attribute("sa.result_final_full", truncated)  # type: ignore[attr-defined]
                                        span.set_attribute("sa.result_final_full_len", len(fa))  # type: ignore[attr-defined]
                                        span.set_attribute("sa.result_final_full_truncated", len(fa) > cap)  # type: ignore[attr-defined]
                                        try:
                                            digest = hashlib.sha256(fa.encode("utf-8")).hexdigest()
                                            span.set_attribute("sa.result_final_sha256", digest)  # type: ignore[attr-defined]
                                        except Exception:
                                            pass
                        except Exception:
                            pass

                        return result
                    except Exception as e:  # pragma: no cover - passthrough with metadata
                        try:
                            span.record_exception(e)  # type: ignore[attr-defined]
                            span.set_status(Status(StatusCode.ERROR))  # type: ignore[attr-defined]
                        finally:
                            pass
                        raise
                    finally:
                        try:
                            duration_ms = int((time.perf_counter() - t0) * 1000)
                            span.set_attribute("sa.duration_ms", duration_ms)  # type: ignore[attr-defined]
                        except Exception:
                            pass
            except Exception:
                # No OTel at runtime (or misconfigured): act as a no-op decorator
                return fn(*args, **kwargs)

        return sync_wrapper

    # Support both @observe and @observe() forms
    if callable(_fn):
        return _decorate(_fn)
    return _decorate


