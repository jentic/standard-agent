# Evaluation (OpenTelemetry + Langfuse)

This layer instruments Standard-Agent using OpenTelemetry and streams spans to Langfuse (Cloud or self-hosted). JSONL run records are still written for aggregation.

## What is captured
- `time_ms`: wall-clock duration of `agent.solve(goal)`
- Tokens: `tokens_prompt`, `tokens_completion`, `tokens_total` per run (NA if provider doesn’t return usage)
- `success`: `ReasoningResult.success`
- Span attributes: `dataset_id`, `item_id`, `agent_name`, `config_hash`, `goal`, `duration_ms`, tokens
- `trace_ids` in JSONL for linking to Langfuse traces

## Setup
1) Install observability deps: `pip install -e ".[observability]"`
2) Env (Cloud):
```
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```
3) Run:
```
python -m evaluation.runner run --dataset evaluation/datasets/smoke_nyt.jsonl --agent react --config evaluation/configs/smoke_config.json --limit 1
```

## Files
- `evaluation/otel_setup.py` – OTel init
- `evaluation/instrumentation.py` – wraps `agent.solve`, attaches LLM token callback
- `evaluation/runner.py` – loads .env, instruments agent/LLM, writes JSONL, carries `trace_ids`

## Notes
- We replaced custom tracing/deepeval with standard OTel + Langfuse SDK.
- To remove JSONL later, drop `JsonlStorage` from `runner.py`.
