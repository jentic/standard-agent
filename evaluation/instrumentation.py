"""Non-intrusive instrumentation for Standard Agent using OpenTelemetry (OTel-only)."""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from .otel_setup import get_tracer

# Per-run context for metrics aggregation
current_run: ContextVar[Optional[Dict[str, Any]]] = ContextVar("current_run", default=None)


def instrument_agent(agent: Any, llm: Any) -> None:
    """Apply non-intrusive instrumentation to agent and its LLM.

    - Wraps agent.solve to create an OTel span and measure duration
    - Hooks LLM.usage_callback to accumulate token counts
    """

    if hasattr(agent, "solve") and not getattr(agent.solve, "_otel_instrumented", False):
        original_solve = agent.solve

        def solve_wrapper(goal: str) -> Any:
            tracer = get_tracer()
            run_id = str(uuid.uuid4())
            existing = current_run.get() or {}
            run_ctx = dict(existing)
            run_ctx.update({
                "run_id": run_id,
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "start": time.monotonic(),
            })
            current_run.set(run_ctx)

            with tracer.start_as_current_span("agent_solve") as span:
                span.set_attribute("run_id", run_id)
                span.set_attribute("goal", goal[:500])
                span.set_attribute("goal_length", len(goal))
                # If the runner pre-populated metadata, attach as attributes
                for meta_key in ("dataset_id", "item_id", "agent_name", "config_hash"):
                    if meta_key in run_ctx and run_ctx[meta_key] is not None:
                        span.set_attribute(meta_key, str(run_ctx[meta_key]))
                try:
                    # Call the original solve method within the OTel span
                    result = original_solve(goal)
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute("error", True)
                    raise
                finally:
                    duration_ms = (time.monotonic() - run_ctx["start"]) * 1000
                    span.set_attribute("duration_ms", duration_ms)
                    # Store for runner JSONL
                    run_ctx["duration_ms"] = duration_ms
                    # Capture trace id for linking JSONL → dashboard
                    try:
                        t_id = span.get_span_context().trace_id
                        run_ctx["trace_id"] = f"{t_id:032x}"
                    except Exception:
                        pass

                    total_tokens = run_ctx["tokens_prompt"] + run_ctx["tokens_completion"]
                    if total_tokens > 0:
                        span.set_attribute("tokens_prompt", run_ctx["tokens_prompt"])
                        span.set_attribute("tokens_completion", run_ctx["tokens_completion"])
                        span.set_attribute("tokens_total", total_tokens)

                # Attach a small result preview if possible
                try:
                    preview = str(result)
                    if len(preview) > 1000:
                        preview = preview[:1000] + "..."
                    span.set_attribute("result_preview", preview)
                except Exception:
                    pass

                # Try to capture success attribute
                try:
                    success = getattr(result, "success", None)
                    if isinstance(success, bool):
                        span.set_attribute("success", success)
                except Exception:
                    pass

                return result

        solve_wrapper._otel_instrumented = True
        agent.solve = solve_wrapper  # type: ignore[assignment]

    # Always attach usage callback for token accounting (LLM checks getattr internally)
    if not getattr(llm, "_otel_instrumented", False):
        try:
            setattr(llm, "usage_callback", _token_callback)
        except Exception:
            pass
        llm._otel_instrumented = True  # type: ignore[attr-defined]


def _token_callback(prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
    run_ctx = current_run.get()
    if not run_ctx:
        return
    if prompt_tokens is not None:
        run_ctx["tokens_prompt"] += int(prompt_tokens)
    if completion_tokens is not None:
        run_ctx["tokens_completion"] += int(completion_tokens)


def get_current_run_metrics() -> Dict[str, Any]:
    run_ctx = current_run.get() or {}
    tokens_prompt = int(run_ctx.get("tokens_prompt", 0))
    tokens_completion = int(run_ctx.get("tokens_completion", 0))
    total = tokens_prompt + tokens_completion
    return {
        "run_id": run_ctx.get("run_id"),
        "duration_ms": run_ctx.get("duration_ms", 0),
        "tokens_prompt": tokens_prompt if tokens_prompt > 0 else None,
        "tokens_completion": tokens_completion if tokens_completion > 0 else None,
        "tokens_total": total if total > 0 else None,
    }


