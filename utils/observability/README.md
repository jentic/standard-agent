# Observability  

Observability is **optional and opt-in**.  
Skip the install → everything no-ops.  
Enable it → you get **traces, metrics, and LLM insights** out of the box, ready for any OpenTelemetry backend.  

---

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

### The `@observe` Decorator  

The core API for observability is the `@observe` decorator:  

- **Default** → Wrap any function, capture execution time + metadata.  
- **`llm=True`** → Adds LLM-aware tracing, including token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`).  
- **`root=True`** → Aggregates tokens + spans from all child calls under a single parent trace (ideal for workflows or agent runs).  

This makes it effortless to track both system performance and LLM usage.  

---

### Architecture  

- **`observe.py`** → `@observe` decorator (no-ops if OTel missing).  
- **`otel_setup.py`** → `setup_telemetry(service, target)` wires up the OTel SDK + exporter.  
- **`exporters/`** → plug-and-play factories:  
  - `langfuse.py` → `LANGFUSE_*` env vars  
  - `otel.py` → `OTEL_*` env vars  

⚠️ If `setup_telemetry` isn’t called, spans are dropped silently (no config, no errors).  

---

### Add a New Exporter  

1. Create `create_<name>_exporter()` in `exporters/<name>.py`.  
2. Export it via `exporters/__init__.py`.  
3. Add `TelemetryTarget.<NAME>` in `otel_setup.py`.  

---

### Contributing  

Contributions are welcome — exporters, docs, ideas, or improvements.  
👉 Open an issue, submit a PR, or share feedback!  
