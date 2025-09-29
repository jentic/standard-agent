"""Simple, minimal tracing decorator for Standard Agent."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any, Callable, Optional
from contextvars import ContextVar
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
                    span.set_attribute("sa.duration_ms", duration_ms)
                    
                    # Finalize token accumulator for root spans
                    if root:
                        _finalize_token_accumulator(span)
        
        return wrapper
    
    # Support both @observe and @observe() forms
    if callable(_fn):
        return _decorate(_fn)
    return _decorate


def _capture_input(span: Any, fn: Callable, args: tuple, kwargs: dict, llm: bool) -> None:
    """Capture function inputs - goal for regular functions, messages for LLM."""
    try:
        bound = signature(fn).bind_partial(*args, **kwargs)
        
        if llm:
            # For LLM calls, capture messages
            messages = bound.arguments.get("messages")
            if messages:
                span.set_attribute("input", str(messages)[:8192])
        else:
            # For regular functions, capture goal if present
            goal = bound.arguments.get("goal")
            if isinstance(goal, str):
                span.set_attribute("sa.goal", goal)
                span.set_attribute("input", goal)
    except Exception:
        pass


def _capture_output(span: Any, result: Any) -> None:
    """Capture regular function outputs."""
    try:
        # Handle common result attributes
        if hasattr(result, 'success'):
            span.set_attribute("sa.result_success", bool(result.success))
        if hasattr(result, 'iterations'):
            span.set_attribute("sa.result_iterations", int(result.iterations))
        if hasattr(result, 'final_answer'):
            preview = str(result.final_answer)[:200]
            span.set_attribute("sa.result_final_preview", preview)
        
        # Set generic output
        span.set_attribute("output", str(result)[:1000])
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