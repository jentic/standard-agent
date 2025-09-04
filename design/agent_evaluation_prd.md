## Agent Evaluation Framework PRD (Standard-Agent)

### 1. Problem Statement
We need a non-intrusive evaluation framework to compare agents and configurations built with `Standard-Agent` on an apples-to-apples basis. The first-pass metrics per run: Success (boolean), Time to Complete, and Tokens Consumed (ptok + ctok). Metrics must be persisted and aggregated over a labeled dataset (Goal + Expected Result) to compute success/failure rates and other analytics.

### 2. Goals and Non-Goals
- Goals
  - Instrument `StandardAgent` runs without invasive code changes, ideally via decorators.
  - Collect per-run metrics: success, latency, tokens.
  - Persist run-level metrics with sufficient context (agent id, config, seed, dataset item id).
  - Provide aggregation utilities (per agent/config, per dataset) to compute success rate and distributions.
  - Optional: Surface traces and metrics in a UI dashboard.
- Non-Goals (for first pass)
  - Ground-truth validators for arbitrary tasks (will rely on `ReasoningResult.success`).
  - Complex outcome grading; simple boolean success only.
  - Importing Deepeval in core; core remains provider-agnostic.

### 3. Key Metrics (v1)
- Success: use `ReasoningResult.success` (boolean).
- Time to Complete: wall-clock time of `StandardAgent.solve(goal)`.
- Tokens Consumed: sum of prompt and completion tokens across the run (ptok+ctok) using the LLM wrapper counters.

### 4. High-Level Approach
- Use a provider-agnostic `evaluation/tracing.py` with our own `@observe`-like decorator that:
  - Starts/ends spans around target functions.
  - Uses contextvars to accumulate per-run totals (tokens, timing).
  - Records inputs/outputs with truncation and redaction.
- Plug in backends via adapters:
  - `JsonTracer` (default, no deps): write spans/run summaries to JSONL.
  - Optional adapters: `OpenTelemetryTracer`, `DeepEvalTracer` (used only from runner if installed).
- Wrap the actual instance methods in the runner (per-instance), not in core.
- Persist run records to JSONL (and optionally SQLite later) with a clean schema.

### 5. Tracing/Backend Integration Plan
- Core does NOT import Deepeval.
- Runner selects tracer backend based on config/env:
  - Default: `JsonTracer`.
  - If `OTEL_EXPORTER_OTLP_ENDPOINT` present: use `OpenTelemetryTracer`.
  - If Deepeval is installed and explicitly requested: use `DeepEvalTracer` adapter.
- Our `observe(tracer, attrs_fn=None)` decorator is used to wrap:
  - `StandardAgent.solve` (per-instance) for latency and IO capture.
  - LLM completion method (per-instance) for token usage accumulation.

### 6. Architecture & Components
- evaluation/
  - tracing.py: pluggable tracer interface, `observe` decorator, contextvars store.
  - hooks.py: helper to enable per-instance wrapping of agent/llm methods.
  - storage.py: persistence (JSONL, optional SQLite later).
  - metrics.py: aggregation utilities (success rate, avg latency, token distributions).
  - dataset.py: schema and loader for gold dataset items.
  - runner.py: CLI to run agents on datasets and collect results.

### 7. Data Model
- RunRecord (JSON serializable):
  - run_id: uuid
  - dataset_id: str | None
  - item_id: str | None
  - agent_name: str
  - agent_version: str | None
  - config_hash: str (hash of selected config, prompts, model, temperature, etc.)
  - timestamp_utc: str
  - goal: str
  - expected: Optional[str | dict]
  - result: Optional[str | dict]
  - success: bool
  - time_ms: int
  - tokens_prompt: int
  - tokens_completion: int
  - tokens_total: int
  - trace_ids: Optional[list[str]] (if Deepeval span ids)
  - extra: dict (freeform)

### 8. Instrumentation Details (v1)
- Latency: wall-clock via `time.monotonic()` around `agent.solve`.
- Tokens: extend `LiteLLM` to expose per-call token counts; accumulate in a thread-local or per-run context span.
- Success: read from `ReasoningResult.success` post-run.

### 9. Storage
- Default: JSONL at `./eval_runs/<dataset_or_experiment>.jsonl` with one record per run.
- Optional: SQLite for queryable storage (future work).

### 10. Aggregation & Reporting
- Aggregations:
  - Success rate = sum(success)/N.
  - Failure rate = 1 - success rate.
  - Avg/median time_ms and tokens_total.
  - Breakdown by agent_name, config_hash, dataset_id.
- CLI commands:
  - `eval run --dataset <path> --agent <profile> --config <cfg.json>`
  - `eval aggregate --runs <path.jsonl> [--by agent_name,config_hash]`

### 11. Example Hook Usage (sketch)
```python
# evaluation/hooks.py
from evaluation.tracing import observe, JsonTracer
import time

tracer = JsonTracer(output_path="./eval_runs/spans.jsonl")

def observe_agent_run(func):
    return observe(tracer)(func)
```

### 12. Risks & Mitigations
- Relying on `ReasoningResult.success` may mislabel outcomes — acceptable for v1; plan validator interface in v2.
- Token counting accuracy varies by model/provider — use provider responses where available; otherwise estimate via tiktoken as fallback.
- Multiple tracing backends add complexity — default to JsonTracer; make others opt-in.

### 13. Milestones
- M1: Tracing layer + JSONL persistence for v1 metrics; CLI runner minimal.
- M2: Aggregation utilities; per-agent/config reports.
- M3: Optional adapters (OpenTelemetry, Deepeval) and SQLite backend.

### 14. Acceptance Criteria (v1)
- Run a dataset of ≥10 goals against ≥2 agent configs.
- Persist per-run records including success, time_ms and tokens_total.
- Aggregate success rate and mean time/tokens per config.
- No changes required to core `agents/*` files beyond optional opt-in wrappers in example runners.
