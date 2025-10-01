# Observability

Observability is **completely optional** and **opt-in**.  
If you don’t install it, everything degrades gracefully to a no-op.  
If you do, you get **traces, metrics, and token usage** captured automatically—framework-agnostic and exportable to your favorite OpenTelemetry backend.

---

## Why Use This?
- **Zero-risk opt-in** → no dependencies? No problem. Your code still runs as-is.  
- **Open standards** → powered by [OpenTelemetry](https://opentelemetry.io/), export anywhere (Langfuse, Jaeger, Honeycomb, etc).  
- **Composable exporters** → add new backends with just a few lines of code.  
- **LLM-aware tracing** → built-in token counting + aggregation across agent calls.  
- **Ergonomic API** → decorate functions with `@observe` and you’re done.

---

## Quick Start

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

## Core Concepts

### 1. `@observe` Decorator
- Use `@observe` to wrap functions.  
- Falls back to no-op if OpenTelemetry isn’t installed.  
- Special modes:  
  - `@observe(llm=True)` → records `prompt_tokens`, `completion_tokens`, `total_tokens`.  
  - `@observe(root=True)` → aggregates tokens across child calls (great for per-run or per-agent tracking).  

---

### 2. Telemetry Setup (`otel_setup.py`)
- Call `setup_telemetry(service_name, target)` once in your entrypoint.  
- Wires the OTel SDK and chosen exporter.  
- ⚠️ If you skip this, spans are dropped silently (no errors).  

---

### 3. Exporters (`exporters/`)
- Pluggable factories for sending data to backends:  
  - `langfuse.py` → uses `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.  
  - `otel.py` → uses standard OTel env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`, etc).  


   **Add a new exporter by:**  
   1. Creating `create_<name>_exporter()` in `exporters/<name>.py`.  
   2. Exporting it in `exporters/__init__.py`.  
   3. Extending `TelemetryTarget` in `otel_setup.py`.  


## 💡 Contributing

We’d love your help!
Whether it’s adding exporters, improving docs, or shaping better LLM observability, contributions are welcome.

👉 Open an issue, submit a PR, or just share your ideas.