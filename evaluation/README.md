Evaluation CLI (experimental)

Run agents on a dataset and collect per-run metrics (success, latency, tokens) without modifying core code. Tokens are None when provider usage is unavailable.

Quick start
```bash
python -m evaluation.runner run --dataset path/to/dataset.jsonl --agent rewoo --config config.json --limit 5
python -m evaluation.runner aggregate --runs eval_runs/<dataset>__<date>__<hash>.jsonl
```

Dataset formats
- JSONL: one object per line with fields: `id`, `goal`, optional `expected`, `metadata`.
- CSV: columns `id`, `goal`, optional `expected`.

Notes
- The runner wraps the agent and LLM per-instance to record latency and token usage via a callback installed on the LLM. Core library files remain unchanged.
- Tokens are set to None for the whole run if any LLM call in the run lacks usage.
- Spans are written to `eval_runs/spans.jsonl` via JsonTracer by default.


