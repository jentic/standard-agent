## Task 02: Storage and Schema

### Scope
Design and implement persistent storage for per-run metrics as JSONL (v1), with an extensible, versioned schema and safe writes. Spans from `JsonTracer` are stored separately (optional), while `RunRecord` is canonical for aggregation.

### Deliverables
- `evaluation/storage.py` with:
  - `RunRecord` dataclass (or Pydantic model if allowed) and validation helpers.
  - `JsonlStorage` to append and iterate run records with atomic writes.
  - Path utilities and filename conventions.
- Tests for schema validation and IO behavior.

### Schema (v1)
- `schema_version`: "1.0"
- `run_id`: str (uuid)
- `dataset_id`: Optional[str]
- `item_id`: Optional[str]
- `agent_name`: str
- `agent_version`: Optional[str]
- `config_hash`: str
- `timestamp_utc`: ISO8601 string, UTC
- `goal`: str
- `expected`: Optional[str | dict]
- `result`: Optional[str | dict]
- `success`: bool
- `time_ms`: int >= 0
- `tokens_prompt`: int >= 0
- `tokens_completion`: int >= 0
- `tokens_total`: int >= 0 (redundant convenience = prompt + completion)
- `trace_ids`: Optional[list[str]]  # ids from tracer backend if available
- `agent_config`: Optional[dict] (redacted)
- `errors`: Optional[list[str]]
- `runtime_env`: Optional[dict]
- `extra`: dict (freeform)

### File Layout
- Base directory: `./eval_runs/`
- Default filename: `<dataset_or_experiment>__<YYYY-MM-DD>__<shortcfg>.jsonl`
- Provide utility `make_output_path(dataset_id: str, config_hash: str) -> Path`

### Implementation Details
- Validation, atomic append, serialization, redaction as previously specified.
- Optional spans file (from `JsonTracer`) written to `./eval_runs/spans/<run_id>.jsonl` or a single rolling file; not required for aggregation.

### Tests
- Writing 10 records and reading them back via `iter_records()` yields identical field values and non-decreasing timestamps.
- Missing required fields cause a validation error.
- Negative tokens/time are rejected.
- Filenames created by `make_output_path` match the convention and include the short config hash.

### Acceptance Criteria
- JSONL storage tolerates concurrent appends from multiple processes without corruption (best-effort via append+fsync).
- Schema version included and parsed; unknown fields ignored gracefully for forward-compat.
- Default path `./eval_runs/<dataset>.jsonl` used when none provided.
