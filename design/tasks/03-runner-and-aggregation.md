## Task 03: Runner and Aggregation

### Scope
Implement a CLI runner to execute agents on a dataset and aggregate results; provide aggregation utilities. v1 is sequential execution (no concurrency), with deterministic config hashing and robust error handling. The runner selects a tracer backend (default JsonTracer) and wraps methods per-instance.

### Deliverables
- `evaluation/runner.py` with CLI commands:
  - `eval run --dataset <path> --agent <profile> --config <cfg.json> [--output <jsonl>] [--limit N] [--seed S] [--tracer json|otel|deepeval]`
  - `eval aggregate --runs <path.jsonl> [--by agent_name,config_hash] [--json <out.json>]`
- `evaluation/metrics.py` with aggregation functions (success rate, avg/median latency and tokens, min/max, p50/p90).
- Simple README snippet showing usage.

### Tracer Selection
- `--tracer json` (default): `JsonTracer` with spans file path.
- `--tracer otel`: `OpenTelemetryTracer` if env `OTEL_EXPORTER_OTLP_ENDPOINT` is set, else error.
- `--tracer deepeval`: use adapter if `deepeval` is installed, else error.

### Execution Flow (eval run)
1. Parse args, load dataset, load config, compute `config_hash`.
2. Build agent by profile; create tracer per `--tracer`.
3. Enable instrumentation per-instance: `enable_instrumentation(agent, llm=agent.llm, tracer=tracer)`.
4. For each item (respecting `--limit`):
   - Start timer and set a new `run_id` in context.
   - Call `agent.solve(goal)`; capture `ReasoningResult`.
   - Compute latency and read token totals from context/tracer.
   - Build `RunRecord` and append via `JsonlStorage` to output path.
   - On exception, record `success=False`, populate `errors`, continue.
5. Print a brief summary and the output file path.

### Aggregation (eval aggregate)
As previously specified: compute success rate, latency and token stats, group by dimensions, output table or JSON.

### Tests
- JSON tracer end-to-end: dataset of 3 items produces 3 run records and spans file; aggregates match expectations.
- Invalid tracer name yields a clear error.
- Per-instance wrapping is idempotent and does not affect other agents in the same process.

### Acceptance Criteria
- Running against a small dataset produces a JSONL of run records, with spans captured by the selected tracer.
- Aggregation prints success rate and latency/token stats per agent/config.
- Helpful error messages and non-zero exit codes on fatal IO/parse errors.
