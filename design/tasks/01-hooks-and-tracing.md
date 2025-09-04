## Task 01: Hooks and Tracing (Pluggable Tracer)

### Scope
Implement non-intrusive decorators to instrument `StandardAgent.solve` and LLM calls for latency and token metrics using a provider-agnostic tracing layer. No core edits in `agents/*` for v1; all instrumentation must be opt-in from runner/tests.

### Deliverables
- `evaluation/tracing.py` with:
  - Tracer interface: `start_span(name, attrs=None) -> Span` with `set_attr(key, value)`.
  - `observe(tracer, attrs_fn=None)` decorator using contextvars for per-run state.
  - `JsonTracer` implementation writing spans to JSONL.
- `evaluation/hooks.py` with:
  - `enable_instrumentation(agent, llm, tracer)` that wraps per-instance methods.
  - `observe_agent_run` and `observe_llm_completion` helpers bound to a tracer.
- Safe no-op behavior when no tracer provided.
- Unit tests covering decorators on dummy functions.

### Detailed Design
- Entry points
  - Wrap `StandardAgent.solve(goal: str) -> ReasoningResult` with `observe(tracer)`.
  - Wrap LLM completion call path: `LiteLLM.completion(...)` with `observe(tracer)` or a targeted wrapper that extracts usage.
- Context propagation
  - `contextvars.ContextVar` named `current_run` holding:
    ```python
    {"run_id": str, "tokens_prompt": int, "tokens_completion": int, "start_monotonic": float}
    ```
- Token extraction
  - Prefer provider `usage` in responses; fallback to estimator if configured.
- Idempotency
  - `enable_instrumentation` guards with `_eval_wrapped` attributes on instances.

### Example
```python
# runner excerpt
from evaluation.tracing import JsonTracer
from evaluation.hooks import enable_instrumentation

tracer = JsonTracer(output_path="./eval_runs/spans.jsonl")
agent = ReWOOAgent(model="gpt-4o")
enable_instrumentation(agent, llm=agent.llm, tracer=tracer)
result = agent.solve("goal text")
```

### Tests
- Span created for wrapped `solve`, `duration_ms` > 0, goal/result recorded.
- Token counters increase when LLM completion returns usage.
- Double `enable_instrumentation` is a no-op.

### Acceptance Criteria
- Real per-instance wrapping works without core changes; spans/metrics written via `JsonTracer`.
