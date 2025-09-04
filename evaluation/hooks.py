from __future__ import annotations

from types import MethodType
from typing import Any, Optional

from .tracing import Tracer, observe, current_run


def enable_instrumentation(agent: Any, llm: Optional[Any], tracer: Tracer) -> None:
    """Wrap per-instance methods to capture spans/metrics without core edits.

    - Wraps agent.solve if present
    - Wraps llm.completion if provided and present
    - Idempotent: sets _eval_wrapped flags on instances
    """
    if hasattr(agent, "_eval_wrapped") and getattr(agent, "_eval_wrapped"):
        return

    if hasattr(agent, "solve"):
        original_solve = agent.solve

        def _attrs_fn(goal: str, *args, **kwargs):
            return {"goal_preview": goal[:1000] if isinstance(goal, str) else str(goal)}

        def _wrap_solve(*args, **kwargs):
            # Initialize per-run token aggregation
            run = current_run.get()
            if run is None:
                run = {"run_id": None}
                current_run.set(run)
            run.setdefault("tokens_prompt_sum", 0)
            run.setdefault("tokens_completion_sum", 0)
            run.setdefault("token_na", False)
            return original_solve(*args, **kwargs)

        wrapped = observe(tracer, attrs_fn=_attrs_fn)(_wrap_solve)
        # Rebind to this instance
        agent.solve = MethodType(wrapped, agent)

    setattr(agent, "_eval_wrapped", True)

    if llm is None or (hasattr(llm, "_eval_wrapped") and getattr(llm, "_eval_wrapped")):
        return

    # Wrap LLM completion to allow token accounting or span capture later
    if hasattr(llm, "completion"):
        original_completion = llm.completion

        def _usage_callback(ptok, ctok):
            run = current_run.get()
            if run is None:
                return
            if ptok is None or ctok is None:
                run["token_na"] = True
                return
            run["tokens_prompt_sum"] = int(run.get("tokens_prompt_sum", 0)) + int(ptok)
            run["tokens_completion_sum"] = int(run.get("tokens_completion_sum", 0)) + int(ctok)

        # attach callback to llm instance
        setattr(llm, "usage_callback", _usage_callback)

        def completion_wrapper(*args, **kwargs):
            return original_completion(*args, **kwargs)

        llm.completion = MethodType(completion_wrapper, llm)
        setattr(llm, "_eval_wrapped", True)


__all__ = ["enable_instrumentation"]


