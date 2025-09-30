"""Simple, minimal tracing decorator for Standard Agent."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any, Callable, Optional
from contextvars import ContextVar
from dataclasses import is_dataclass, asdict
import json
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
                # No OTel available - just run the function
                return fn(*args, **kwargs)
            
            start_time = time.perf_counter()
            
            with tracer.start_as_current_span(span_name) as span:
                try:
                    if root:
                        _start_token_accumulator(span)
                    
                    _capture_input(span, fn, args, kwargs, llm)
                    result = fn(*args, **kwargs)
                    
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
                    span.set_attribute("duration_ms", duration_ms)
                    if root:
                        _finalize_token_accumulator(span)
        
        return wrapper
    
    # Support both @observe and @observe() forms
    if callable(_fn):
        return _decorate(_fn)
    return _decorate


def _capture_input(span: Any, fn: Callable, args: tuple, kwargs: dict, llm: bool) -> None:
    """Capture function inputs with smart previews and redaction."""
    try:
        bound = signature(fn).bind_partial(*args, **kwargs)

        # LLM path: capture messages only (longer cap for prompt visibility)
        if llm:
            messages = bound.arguments.get("messages")
            if messages:
                msg_str = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                span.set_attribute("input", msg_str[:12288])  # 12KB for prompts
            return

        # BASIC redaction candidate keys
        SECRET_KEYS = {"api_key", "apikey", "access_token", "accesstoken", "refresh_token",
                       "client_secret", "secret", "password", "authorization", "bearer",
                       "cookie", "set_cookie", "private_key", "ssh_key"}
        
        def _safe_preview(val: Any, max_len: int = 512) -> Any:
            """Create safe preview of any value."""
            if val is None or isinstance(val, (bool, int, float)):
                return val
            if isinstance(val, str):
                return val[:max_len]
            if isinstance(val, dict):
                return {
                    str(k): ("<redacted>" if str(k).lower().replace("_", "").replace("-", "") in SECRET_KEYS
                            else _safe_preview(v, max_len))
                        for k, v in list(val.items())[:20]
                }
            if isinstance(val, (list, tuple)):
                return [_safe_preview(v, max_len) for v in list(val)[:20]]
            if is_dataclass(val):
                return _safe_preview(asdict(val), max_len)
            if hasattr(val, "model_dump"):
                return _safe_preview(val.model_dump(), max_len)
            return repr(val)[:max_len]

        # Build input dict excluding self/cls
        inputs = {name: _safe_preview(value) for name, value in bound.arguments.items() if name not in {"self", "cls"}}
        
        input_str = json.dumps(inputs, ensure_ascii=False, separators=(",", ":"), default=str)
        span.set_attribute("input", input_str[:6144] + ("..." if len(input_str) > 6144 else ""))
    except Exception:
        pass


def _capture_output(span: Any, result: Any) -> None:
    """Capture non-LLM outputs with structured attributes."""
    try:
        # ReasoningResult-like: capture structured fields
        if hasattr(result, 'final_answer'):
            output = result.final_answer or getattr(result, 'transcript', '')
            span.set_attribute("output", str(output)[:8192])
            if hasattr(result, 'success'):
                span.set_attribute("result_success", bool(result.success))
            if hasattr(result, 'iterations'):
                span.set_attribute("total_iterations", int(result.iterations))
        else:
            # Generic output
            span.set_attribute("output", str(result)[:8192])
    except Exception:
        pass


def _capture_llm_output(span: Any, result: Any) -> None:
    """Capture LLM outputs and track tokens."""
    try:
        from agents.llm.base_llm import BaseLLM
        
        if isinstance(result, BaseLLM.LLMResponse):
            span.set_attribute("output", result.text)
            if isinstance(result.prompt_tokens, int):
                span.set_attribute("tokens.prompt", result.prompt_tokens)
            if isinstance(result.completion_tokens, int):
                span.set_attribute("tokens.completion", result.completion_tokens)
            if isinstance(result.total_tokens, int):
                span.set_attribute("tokens.total", result.total_tokens)
                _accumulate_tokens(result.total_tokens)
        else:
            span.set_attribute("output", str(result)[:8192])
    except Exception:
        pass


# ── Token Accumulation ──────────────────────────────────────────────────────
# Root spans start a token counter; child LLM calls increment it; root finalizes total.

_tokens: ContextVar[Optional[int]] = ContextVar("tokens", default=None)
_owner: ContextVar[Optional[int]] = ContextVar("owner", default=None)


def _start_token_accumulator(span: Any) -> None:
    """Initialize token counter for root span."""
    if _tokens.get() is None:
        _tokens.set(0)
        _owner.set(id(span))


def _accumulate_tokens(token_count: int) -> None:
    """Add tokens from an LLM call."""
    current = _tokens.get()
    if isinstance(current, int):
        _tokens.set(current + token_count)


def _finalize_token_accumulator(span: Any) -> None:
    """Write total tokens to root span and reset."""
    if _owner.get() == id(span):
        total = _tokens.get()
        if isinstance(total, int) and total > 0:
            span.set_attribute("tokens.total", total)
        _tokens.set(None)
        _owner.set(None)