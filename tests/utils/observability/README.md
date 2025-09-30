# Observability

Observability is **completely optional** and **opt-in**.  
If you don’t install it, everything degrades gracefully to a no-op.  
If you do, you get **traces, metrics, and token usage** captured automatically—framework-agnostic and exportable to your favorite OpenTelemetry backend.

---

## ✨ Why Use This?
- **Zero-risk opt-in** → no dependencies? No problem. Your code still runs as-is.  
- **Open standards** → powered by [OpenTelemetry](https://opentelemetry.io/), export anywhere (Langfuse, Jaeger, Honeycomb, etc).  
- **Composable exporters** → add new backends with just a few lines of code.  
- **LLM-aware tracing** → built-in token counting + aggregation across agent calls.  
- **Ergonomic API** → decorate functions with `@observe` and you’re done.

---

## 🚀 Quick Start

Install with observability extras:

```bash
pip install -e ".[observability]"
Initialize telemetry once in your entrypoint:

python
Copy code
from utils.observability import observe, setup_telemetry, TelemetryTarget

setup_telemetry(
    service_name="standard-agent",
    target=TelemetryTarget.LANGFUSE  # or TelemetryTarget.OTEL
)
Decorate functions and LLM calls:

python
Copy code
@observe
def compute(x, y):
    return x + y

@observe(llm=True)
def call_llm(messages):
    return llm.completion(messages)

@observe(root=True)
def solve(goal):
    ...
```

✅ That’s it—your traces will flow to the configured backend.

## 🏗️ Architecture
- utils/observability/observe.py → @observe decorator (falls back to no-op if OTel isn’t installed).
- utils/observability/otel_setup.py → setup_telemetry(service_name, target) wires up the OTel SDK + exporter.
- utils/observability/exporters/ → pluggable exporter factories:
  - langfuse.py → uses LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST. 
  - otel.py → standard OTel env vars (OTEL_EXPORTER_OTLP_ENDPOINT, etc).

⚠️ If setup_telemetry is never called, spans are dropped silently. No config? No errors.


## 📊 Token Tracking

- @observe(llm=True) → records prompt_tokens, completion_tokens, and total_tokens.
- @observe(root=True) → aggregates tokens across all child calls into a single total_tokens attribute.
- Perfect for tracking LLM usage per run or per agent workflow.

## 🔌 Adding a New Exporter

1. Add create_\<name\>_exporter() in utils/observability/exporters/\<name\>.py.
2. Export it from utils/observability/exporters/__init__.py.
3. Add TelemetryTarget.<NAME> in otel_setup.py and dispatch in _create_exporter.


## 💡 Contributing

We’d love your help!
Whether it’s adding exporters, improving docs, or shaping better LLM observability, contributions are welcome.

👉 Open an issue, submit a PR, or just share your ideas.