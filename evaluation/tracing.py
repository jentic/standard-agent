from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# Per-run context (tokens, timers, identifiers)
current_run: ContextVar[Optional[Dict[str, Any]]] = ContextVar("current_run", default=None)


class Span:
    def __init__(self, name: str, tracer: "Tracer", attrs: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.tracer = tracer
        self.attrs = dict(attrs or {})
        self.start_time_monotonic = 0.0

    def __enter__(self) -> "Span":
        self.start_time_monotonic = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        duration_ms = int((time.monotonic() - self.start_time_monotonic) * 1000)
        self.set_attr("duration_ms", duration_ms)
        self.tracer.on_span_end(self)

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value


class Tracer:
    def start_span(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> Span:
        return Span(name=name, tracer=self, attrs=attrs)

    def on_span_end(self, span: Span) -> None:  # pragma: no cover - interface hook
        pass


class JsonTracer(Tracer):
    """Simple tracer that writes spans as JSON lines.

    Intended for local development and lightweight telemetry without external deps.
    """

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_span_end(self, span: Span) -> None:
        try:
            record = {
                "span_id": str(uuid.uuid4()),
                "name": span.name,
                "attrs": span.attrs,
                "ts": time.time(),
            }
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Best-effort tracing; never crash the app due to telemetry
            pass


def observe(tracer: Tracer, attrs_fn: Optional[Callable[..., Dict[str, Any]]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to create a span around a function call using the provided tracer.

    - Adds duration_ms to span attrs
    - Leaves input/output capture to the caller (via attrs_fn) to avoid large payloads by default
    - Maintains per-run context in `current_run` if unset
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run_state = current_run.get()
            created_state = False
            if run_state is None:
                run_state = {
                    "run_id": str(uuid.uuid4()),
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                }
                current_run.set(run_state)
                created_state = True

            attrs = {}
            if attrs_fn:
                try:
                    attrs = attrs_fn(*args, **kwargs) or {}
                except Exception:
                    attrs = {}
            attrs.setdefault("run_id", run_state["run_id"]) 

            with tracer.start_span(func.__name__, attrs=attrs) as span:
                result = func(*args, **kwargs)
                # Optionally attach lightweight output preview
                try:
                    preview = str(result)
                    if len(preview) > 1000:
                        preview = preview[:1000] + "…"
                    span.set_attr("result_preview", preview)
                except Exception:
                    pass

            if created_state:
                # End-of-run cleanup; callers can read current_run before it resets
                current_run.set(None)
            return result

        wrapper.__name__ = getattr(func, "__name__", "wrapped")
        wrapper.__doc__ = getattr(func, "__doc__", None)
        return wrapper

    return decorator


__all__ = [
    "Tracer",
    "Span",
    "JsonTracer",
    "observe",
    "current_run",
]


