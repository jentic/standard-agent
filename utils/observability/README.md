# Observability  

Observability is **optional and opt-in**.  
Skip the install → everything no-ops.  
Enable it → you get **traces, metrics, and LLM insights** out of the box, ready for any OpenTelemetry backend.  

---
## Why Use This?

- **Zero overhead** → no OTel? Your code still runs fine.  
- **Open standards** → powered by [OpenTelemetry](https://opentelemetry.io/), export anywhere that supports OTEL (Langfuse, Jaeger, Honeycomb…).  
- **One-liner API** → add tracing with a simple `@observe`.
- **LLM-aware** → token usage + aggregation built in.  
- **Extensible** → drop in your own exporters in a few lines.  

---

##  Quick Start  

Install with observability extras:  

```bash
pip install -e ".[observability]"
```

Initialize telemetry once in your entrypoint:
```bash
from utils.observability import observe, setup_telemetry, TelemetryTarget

setup_telemetry(service_name="standard-agent",target=TelemetryTarget.LANGFUSE)  # or TelemetryTarget.OTEL
```

Add tracing with a single decorator `@observe`:
```bash
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

That’s it — traces flow automatically to your configured backend.

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


### Contributing  

Contributions are welcome — exporters, docs, ideas, or improvements.  
👉 Open an issue, submit a PR, or share feedback!  
