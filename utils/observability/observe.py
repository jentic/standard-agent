"""Simple, minimal tracing decorator for Standard Agent."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any, Callable, Optional
from contextvars import ContextVar
from dataclasses import is_dataclass, asdict
import json
import re
import time


def observe(_fn: Optional[Callable[..., Any]] = None, *, llm: bool = False, root: bool = False) -> Callable[..., Any]:
    """Minimal tracing decorator.
    
    Usage:
        @observe
        def my_function(): ...
        
        @observe(llm=True) 
        def llm_call(): ...
        
        @observe(root=True)
        def agent_solve(): ...
    
    - Auto-names spans from function module.qualname
    - Records timing, exceptions, basic I/O
    - Tracks token usage when llm=True
    - Aggregates total tokens when root=True
    - No-op if OpenTelemetry unavailable
    """
    
    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Create span name from module.Class.method
        module = getattr(fn, "__module__", "") or ""
        qualname = getattr(fn, "__qualname__", fn.__name__)
        span_name = f"{module}.{qualname}" if module else qualname
        
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                from opentelemetry import trace
                tracer = trace.get_tracer("standard-agent")
            except Exception:
                # NoOp:  No OTel available - just run the function
                return fn(*args, **kwargs)
            
            start_time = time.perf_counter()
            
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Start token accumulator for root spans
                    if root:
                        _start_token_accumulator(span)
                    
                    # Capture basic input
                    _capture_input(span, fn, args, kwargs, llm)
                    
                    # Run the function
                    result = fn(*args, **kwargs)
                    
                    # Capture output based on function type
                    if llm:
                        _capture_llm_output(span, result)
                    else:
                        _capture_output(span, result)
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise
                    
                finally:
                    duration_ms = int((time.perf_counter() - start_time) * 1000)
                    span.set_attribute("goal_duration_ms", duration_ms)
                    
                    # Finalize token accumulator for root spans
                    if root:
                        _finalize_token_accumulator(span)
        
        return wrapper
    
    # Support both @observe and @observe() forms
    if callable(_fn):
        return _decorate(_fn)
    return _decorate


def _capture_input(span: Any, fn: Callable, args: tuple, kwargs: dict, llm: bool) -> None:
    """Capture function inputs dynamically with safe previews and caps."""
    try:
        bound = signature(fn).bind_partial(*args, **kwargs)

        # LLM path: capture messages only (longer cap for prompt visibility)
        if llm:
            messages = bound.arguments.get("messages")
            if messages is not None:
                try:
                    msg_json = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    msg_json = str(messages)
                LLM_INPUT_MAX = 12288  # ~12KB
                span.set_attribute("input", msg_json[:LLM_INPUT_MAX])
            return


        # Targeted secret redaction (do not match generic 'key')
        secret_key_re = re.compile(
            r"^(api(?:_|-)?key|access(?:_|-)?token|refresh(?:_|-)?token|client(?:_|-)?secret|secret|password|authorization|bearer|cookie|set(?:_|-)?cookie|private(?:_|-)?key|ssh(?:_|-)?key)$",
            re.IGNORECASE,
        )

        def preview(value: Any, *, scalar_limit: int = 512, collection_limit: int = 20) -> Any:
            try:
                if value is None:
                    return None
                if isinstance(value, (bool, int, float)):
                    return value
                if isinstance(value, str):
                    return value[:scalar_limit]
                if isinstance(value, dict):
                    out = {}
                    for k, v in list(value.items())[:collection_limit]:
                        key_str = str(k)
                        pv = preview(v, scalar_limit=scalar_limit, collection_limit=collection_limit)
                        if secret_key_re.match(key_str):
                            if isinstance(pv, str):
                                pv = "<redacted>"
                            else:
                                pv = "<redacted>"
                        out[key_str] = pv
                    return out
                if isinstance(value, (list, tuple, set)):
                    items = list(value)
                    return [preview(v, scalar_limit=scalar_limit, collection_limit=collection_limit) for v in items[:collection_limit]]
                if is_dataclass(value):
                    return preview(asdict(value), scalar_limit=scalar_limit, collection_limit=collection_limit)
                # step-like summary
                if hasattr(value, "text") or hasattr(value, "output_key"):
                    data = {}
                    if hasattr(value, "text"):
                        data["text"] = str(getattr(value, "text"))[:scalar_limit]
                    if hasattr(value, "output_key"):
                        ok = getattr(value, "output_key")
                        data["output_key"] = str(ok)[:scalar_limit] if ok is not None else None
                    return data if data else repr(value)[:scalar_limit]
                # tool-like summary
                if hasattr(value, "id") or hasattr(value, "name"):
                    data = {}
                    if hasattr(value, "id"):
                        data["id"] = str(getattr(value, "id"))[:scalar_limit]
                    if hasattr(value, "name"):
                        data["name"] = str(getattr(value, "name"))[:scalar_limit]
                    if not data and hasattr(value, "get_summary") and callable(getattr(value, "get_summary")):
                        try:
                            data["summary"] = str(value.get_summary())[:scalar_limit]
                        except Exception:
                            pass
                    return data if data else repr(value)[:scalar_limit]
                # pydantic-like
                if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
                    try:
                        return preview(value.model_dump(), scalar_limit=scalar_limit, collection_limit=collection_limit)
                    except Exception:
                        pass
                if hasattr(value, "dict") and callable(getattr(value, "dict")):
                    try:
                        return preview(value.dict(), scalar_limit=scalar_limit, collection_limit=collection_limit)
                    except Exception:
                        pass
                return repr(value)[:scalar_limit]
            except Exception:
                return repr(value)[:scalar_limit]

        inputs_preview: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            # Exclude instance/class references which add no signal and can be huge
            if str(name) in {"self", "cls"}:
                continue
            # Redact obvious secret keys at top-level dicts
            if isinstance(value, dict):
                sanitized = {}
                for k, v in value.items():
                    k_str = str(k)
                    if secret_key_re.match(k_str):
                        sanitized[k_str] = "<redacted>"
                    else:
                        sanitized[k_str] = v
                inputs_preview[str(name)] = preview(sanitized)
            else:
                inputs_preview[str(name)] = preview(value)

        try:
            input_json = json.dumps(inputs_preview, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            input_json = str(inputs_preview)

        MAX_INPUT_CHARS = 6144
        clipped = input_json[:MAX_INPUT_CHARS]
        if len(input_json) > MAX_INPUT_CHARS:
            clipped = clipped + "...(truncated)"
        span.set_attribute("input", clipped)
    except Exception:
        pass


def _capture_output(span: Any, result: Any) -> None:
    """Capture regular outputs."""
    try:
        if hasattr(result, 'success'):
            span.set_attribute("result_success", bool(result.success))
        if hasattr(result, 'iterations'):
            span.set_attribute("total_iterations", int(result.iterations))
        if hasattr(result, 'final_answer'):
            span.set_attribute("output", str(result.final_answer)[:8124])
        else:
            span.set_attribute("output", str(result)[:8124])

    except Exception:
        pass


def _capture_llm_output(span: Any, result: Any) -> None:
    """Capture LLM function outputs and handle token tracking."""
    try:
        # Handle LLMResponse with tokens
        try:
            from agents.llm.base_llm import BaseLLM
            if isinstance(result, BaseLLM.LLMResponse):
                span.set_attribute("output", result.text)
                
                # Set per-call token attributes
                if isinstance(result.prompt_tokens, int):
                    span.set_attribute("sa.tokens.prompt", result.prompt_tokens)
                if isinstance(result.completion_tokens, int):
                    span.set_attribute("sa.tokens.completion", result.completion_tokens)
                if isinstance(result.total_tokens, int):
                    span.set_attribute("sa.tokens.total", result.total_tokens)
                    _accumulate_tokens(result.total_tokens)
                return
        except Exception:
            pass
        
        # Fallback for non-LLMResponse
        span.set_attribute("output", str(result))
    except Exception:
        pass


# Token accumulation for root spans
_token_accumulator: ContextVar[Optional[int]] = ContextVar("tokens", default=None)
_token_owner: ContextVar[Optional[int]] = ContextVar("owner", default=None)


def _start_token_accumulator(span: Any) -> None:
    """Start token accumulation for root solve spans."""
    try:
        if _token_accumulator.get() is None:
            _token_accumulator.set(0)
            _token_owner.set(id(span))
    except Exception:
        pass


def _accumulate_tokens(tokens: int) -> None:
    """Add tokens to the current accumulator."""
    try:
        current = _token_accumulator.get()
        if isinstance(current, int):
            _token_accumulator.set(current + tokens)
    except Exception:
        pass


def _finalize_token_accumulator(span: Any) -> None:
    """Emit total tokens and clear accumulator."""
    try:
        if _token_owner.get() == id(span):
            total = _token_accumulator.get()
            if isinstance(total, int) and total > 0:
                span.set_attribute("sa.tokens.total", total)
            _token_accumulator.set(None)
            _token_owner.set(None)
    except Exception:
        pass