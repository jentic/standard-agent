## Agent Evaluation Framework PRD (Standard-Agent)

### 1. Problem Statement
We need a non-intrusive evaluation framework to compare agents and configurations built with `Standard-Agent` on an apples-to-apples basis. The first-pass metrics per run: Success (boolean), Time to Complete, Tokens Consumed (ptok + ctok), and Steps Taken. Metrics must be persisted and aggregated over a labeled dataset (Goal + Expected Result) to compute success/failure rates and other analytics.

### 2. Goals and Non-Goals
- Goals
  - Instrument `StandardAgent` runs without invasive code changes, ideally via decorators.
  - Collect per-run metrics: success, latency, tokens, steps.
  - Persist run-level metrics with sufficient context (agent id, config, seed, dataset item id).
  - Provide aggregation utilities (per agent/config, per dataset) to compute success rate and distributions.
  - Optional: Surface traces and metrics in a UI dashboard.
- Non-Goals (for first pass)
  - Ground-truth validators for arbitrary tasks (will rely on `ReasoningResult.success`).
  - Complex outcome grading; simple boolean success only.
  - Advanced dashboards; we’ll prefer Deepeval/Confident integrations if feasible.

### 3. Key Metrics (v1)
- Success: use `ReasoningResult.success` (boolean).
- Time to Complete: wall-clock time of `StandardAgent.solve(goal)`.
- Tokens Consumed: sum of prompt and completion tokens across the run (ptok+ctok) using the LLM wrapper counters.
- Steps Taken: count of reasoning/tool steps (reasoner-dependent; ReWOO: plan+exec+reflect steps; ReACT: think/act loops).

### 4. High-Level Approach
- Use decorators to hook into key functions:
  - At the entry/exit of `StandardAgent.solve` to time the run and emit a run span.
  - Inside LLM calls (e.g., `LiteLLM.completion`) to accumulate token usage.
  - Inside reasoner loops to increment step counts.
- Prefer Deepeval’s `@observe` + `update_current_span` for tracing and metric capture with minimal intrusion. Wrap our functions externally where possible to avoid edits to core files.
- Persist run records to a simple backend (JSONL or SQLite) with a clean schema. Support later export to Deepeval/Confident if needed.

### 5. Deepeval Feasibility and Integration Plan
- Decorator: `@observe` to create spans on instrumented functions, with `update_current_span(test_case=LLMTestCase(...))` to attach IO.
- Custom Metrics: Implement `BaseMetric` subclasses for latency, steps, and token totals if needed, or compute them via tracing spans and store as custom fields.
- Dashboard: Use Deepeval’s Confident AI integration to surface custom metrics if available; otherwise, store locally and visualize later.
- Non-intrusiveness: Provide wrapper decorators in a separate module (`evaluation/hooks.py`) and apply them in example runners or via a lightweight monkeypatch in tests.

### 6. Architecture & Components
- evaluation/
  - hooks.py: decorators using Deepeval tracing to capture spans and metrics.
  - storage.py: persistence (JSONL, and optional SQLite with SQLAlchemy).
  - metrics.py: metric aggregation utilities (success rate, avg latency, token distributions).
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
  - steps: int
  - trace_ids: Optional[list[str]] (if Deepeval span ids)
  - extra: dict (freeform)

### 8. Instrumentation Details (v1)
- Latency: wall-clock via `time.monotonic()` around `agent.solve`.
- Tokens: extend `LiteLLM` to expose per-call token counts; accumulate in a thread-local or per-run context span.
- Steps: increment on each reasoner step entry (ReWOO plan, execute, reflect; ReACT loop iteration). Use decorator on `Reasoner.run` plus helper counters.
- Success: read from `ReasoningResult.success` post-run.

### 9. Storage
- Default: JSONL at `./eval_runs/<dataset_or_experiment>.jsonl` with one record per run.
- Optional: SQLite for queryable storage (future work).

### 10. Aggregation & Reporting
- Aggregations:
  - Success rate = sum(success)/N.
  - Failure rate = 1 - success rate.
  - Avg/median time_ms, tokens_total, steps.
  - Breakdown by agent_name, config_hash, dataset_id.
- CLI commands:
  - `eval run --dataset <path> --agent <profile> --config <cfg.json>`
  - `eval aggregate --runs <path.jsonl> [--by agent_name,config_hash]`

### 11. Example Hook Usage (sketch)
```python
# evaluation/hooks.py
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
import time

def observe_agent_run(func):
    @observe()
    def wrapper(self, goal: str, *args, **kwargs):
        start = time.monotonic()
        result = func(self, goal, *args, **kwargs)
        duration_ms = int((time.monotonic() - start) * 1000)
        update_current_span(test_case=LLMTestCase(input=goal, actual_output=str(result)))
        # also stash duration_ms in a span attribute; tokens/steps aggregated elsewhere
        return result
    return wrapper
```

### 12. Risks & Mitigations
- Relying on `ReasoningResult.success` may mislabel outcomes — acceptable for v1; plan validator interface in v2.
- Deepeval dashboard custom metrics may require Confident AI account/features — store locally regardless; surface to dashboard when available.
- Token counting accuracy varies by model/provider — use provider responses where available; otherwise estimate via tiktoken as fallback.

### 13. Milestones
- M1: Hooks + JSONL persistence for v1 metrics; CLI runner minimal.
- M2: Aggregation utilities; per-agent/config reports.
- M3: Optional dashboard surfacing via Deepeval; SQLite backend.

### 14. Acceptance Criteria (v1)
- Run a dataset of ≥10 goals against ≥2 agent configs.
- Persist per-run records including success, time_ms, tokens_total, steps.
- Aggregate success rate and mean time/tokens/steps per config.
- No changes required to core `agents/*` files beyond optional opt-in wrappers in example runners.
