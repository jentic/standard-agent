from __future__ import annotations

from functools import wraps
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Optional
import time
import os
import hashlib
import json


def _default_span_name(fn: Callable[..., Any]) -> str:
    """Compute a readable default span name from the function's qualname.

    Example: module.Class.method or module.function
    """
    module = getattr(fn, "__module__", None) or ""
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "anonymous"))
    if module:
        return f"{module}.{qualname}"
    return qualname


def observe(_fn: Optional[Callable[..., Any]] = None, *, llm: bool = False) -> Callable[..., Any]:
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

        def _is_sensitive(name: str) -> bool:
            lowered = name.lower()
            return any(k in lowered for k in ("key", "secret", "token", "password", "auth", "credential"))

        def _result_bool(result: Any, names: tuple[str, ...]) -> Optional[bool]:
            for n in names:
                try:
                    if isinstance(result, dict) and n in result:
                        v = result.get(n)
                    else:
                        v = getattr(result, n)
                    if isinstance(v, bool):
                        return v
                except Exception:
                    continue
            return None

        def _result_int(result: Any, names: tuple[str, ...]) -> Optional[int]:
            for n in names:
                try:
                    if isinstance(result, dict) and n in result:
                        v = result.get(n)
                    else:
                        v = getattr(result, n)
                    if isinstance(v, int):
                        return v
                    if isinstance(v, float):
                        return int(v)
                except Exception:
                    continue
            return None

        def _iter_result_fields(result: Any) -> list[tuple[str, Any]]:
            try:
                if isinstance(result, dict):
                    return list(result.items())
                data = getattr(result, "__dict__", None)
                if isinstance(data, dict):
                    return list(data.items())
            except Exception:
                pass
            return []

        def _is_reasoning_result(obj: Any) -> bool:
            try:
                # Lazy import to avoid hard dependency/cycles
                from agents.reasoner.base import ReasoningResult  # type: ignore
                return isinstance(obj, ReasoningResult)
            except Exception:
                # Fallback duck-typing if import unavailable
                try:
                    if isinstance(obj, dict):
                        return "final_answer" in obj
                    return hasattr(obj, "final_answer")
                except Exception:
                    return False

        def _set_output(span: Any, result: Any, is_llm: bool) -> None:
            # Only set generic 'output' on non-LLM spans
            if is_llm:
                return
            try:
                if _is_reasoning_result(result):
                    fa = None
                    try:
                        fa = result.get("final_answer") if isinstance(result, dict) else getattr(result, "final_answer")
                    except Exception:
                        pass
                    if isinstance(fa, str):
                        span.set_attribute("output", fa)  # type: ignore[attr-defined]
                        return
                if isinstance(result, dict):
                    try:
                        span.set_attribute("output", json.dumps(result, ensure_ascii=False, default=str))  # type: ignore[attr-defined]
                    except Exception:
                        span.set_attribute("output", str(result))  # type: ignore[attr-defined]
                    return
                if isinstance(result, str):
                    span.set_attribute("output", result)  # type: ignore[attr-defined]
                    return
                # Generic objects and sequences (e.g., Deque[Step])
                span.set_attribute("output", str(result))  # type: ignore[attr-defined]
            except Exception:
                pass

        def _capture_result_fields(span: Any, result: Any) -> None:
            try:
                items = _iter_result_fields(result)
                if not items:
                    return
                captured = 0
                for name, value in items:
                    # Skip the core stable fields; they are handled separately
                    if name in {"success", "iterations", "turns", "steps", "final_answer"}:
                        continue
                    if _is_sensitive(name):
                        span.set_attribute(f"sa.result.fields.{name}", "[REDACTED]")  # type: ignore[attr-defined]
                    else:
                        span.set_attribute(f"sa.result.fields.{name}", _maybe_truncate(str(value)))  # type: ignore[attr-defined]
                    captured += 1
                span.set_attribute("sa.result_fields_captured", captured)  # type: ignore[attr-defined]
            except Exception:
                pass

        def _render_messages(messages: Any, cap_chars: int = 2000, cap_items: int = 8) -> str:
            try:
                if not isinstance(messages, list):
                    return _maybe_truncate(str(messages), cap_chars)
                parts: list[str] = []
                total = 0
                for i, m in enumerate(messages):
                    if i >= cap_items:
                        parts.append("…")
                        break
                    if isinstance(m, dict):
                        role = str(m.get("role", ""))
                        content = str(m.get("content", ""))
                    else:
                        role = getattr(m, "role", "")
                        content = getattr(m, "content", "")
                        content = str(content)
                    snippet = f"{role}: {content}"
                    if total + len(snippet) > cap_chars:
                        remaining = max(0, cap_chars - total)
                        snippet = snippet[:remaining] + "…"
                        parts.append(snippet)
                        break
                    parts.append(snippet)
                    total += len(snippet)
                return "\n".join(parts)
            except Exception:
                return _maybe_truncate(repr(messages), cap_chars)

        # If the function is async, return it unchanged (no tracing) to keep the
        # implementation simple. We can add async support later when needed.
        if iscoroutinefunction(fn):
            return fn

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
                            # Populate generic 'input' for non-LLM spans
                            if not llm and isinstance(goal_val, str):
                                span.set_attribute("input", goal_val)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                        result = fn(*args, **kwargs)

                        # Attach basic outcome fields if present
                        try:
                            success_val = _result_bool(result, ("success",))
                            if success_val is not None:
                                span.set_attribute("sa.result_success", success_val)  # type: ignore[attr-defined]

                            iter_val = _result_int(result, ("iterations", "turns", "steps"))
                            if iter_val is not None:
                                span.set_attribute("sa.result_iterations", iter_val)  # type: ignore[attr-defined]
                            # ReasoningResult preview if available
                            try:
                                fa = getattr(result, "final_answer") if hasattr(result, "final_answer") else None
                                if isinstance(fa, str):
                                    span.set_attribute("sa.result_final_preview", _maybe_truncate(fa))  # type: ignore[attr-defined]
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
                            # Populate generic 'output' according to type rules
                            _set_output(span, result, llm)
                            _capture_result_fields(span, result)
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


