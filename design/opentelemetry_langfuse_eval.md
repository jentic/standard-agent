# OpenTelemetry + Langfuse Migration Plan

**Goal:** Replace our custom evaluation framework with industry-standard OpenTelemetry + Langfuse for better maintainability, features, and UI.

**Timeline:** ~2 hours total
**Approach:** Clean rewrite, no backward compatibility needed

---

## 📋 **Migration Checklist**

### **Phase 1: Setup & Dependencies (30 minutes)**
- [ ] Install OpenTelemetry packages
- [ ] Install and configure Langfuse (self-hosted)
- [ ] Verify Langfuse dashboard is accessible
- [ ] Remove old dependencies from requirements

### **Phase 2: Replace Custom Tracing (45 minutes)**
- [ ] Delete custom framework files
- [ ] Create OpenTelemetry setup module
- [ ] Replace custom decorators with OTel spans
- [ ] Modify runner.py to use OpenTelemetry
- [ ] Update StandardAgent.solve instrumentation

### **Phase 3: Integration & Testing (30 minutes)**
- [ ] Test basic span creation
- [ ] Verify spans appear in Langfuse dashboard
- [ ] Test custom metrics and attributes
- [ ] Run smoke test with existing dataset
- [ ] Verify token counting works

### **Phase 4: Cleanup (15 minutes)**
- [ ] Remove Streamlit dashboard files
- [ ] Clean up imports and references
- [ ] Update evaluation README
- [ ] Document new usage patterns

---

## 🔧 **Phase 1: Setup & Dependencies**

### **1.1 Install OpenTelemetry Packages**

```bash
pip install \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-instrumentation \
    langfuse
```

### **1.2 Setup Langfuse (Self-Hosted)**

Create `docker-compose.yml` in project root:

```yaml
# docker-compose.yml
version: '3.8'
services:
  langfuse-server:
    image: langfuse/langfuse:latest
    depends_on:
      - db
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres
      - NEXTAUTH_SECRET=mysecret
      - SALT=mysalt
      - NEXTAUTH_URL=http://localhost:3000
      - TELEMETRY_ENABLED=true
      - LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false

  db:
    image: postgres:15
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=postgres
    ports:
      - "5432:5432"
    volumes:
      - langfuse_db_data:/var/lib/postgresql/data

volumes:
  langfuse_db_data:
    driver: local
```

**Start Langfuse:**
```bash
docker-compose up -d
```

**Verify setup:**
- Open http://localhost:3000
- Create account and project
- Note down API keys from project settings

### **1.3 Environment Configuration**

Add to `.env`:
```bash
# Langfuse configuration
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

---

## 🗑️ **Phase 2: Replace Custom Tracing**

### **2.1 Delete Custom Framework Files**

**Files to delete:**
```bash
rm evaluation/tracing.py          # 132 lines of custom spans/tracers
rm evaluation/hooks.py            # Complex method wrapping logic
rm evaluation/dashboard.py        # Streamlit dashboard
rm evaluation/requirements-dashboard.txt
```

### **2.2 Create OpenTelemetry Setup Module**

**Create `evaluation/otel_setup.py`:**

```python
"""OpenTelemetry + Langfuse setup for Standard Agent evaluation."""

import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from langfuse.opentelemetry import LangfuseSpanProcessor
from typing import Optional

# Global tracer and meter instances
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None

def setup_telemetry(
    service_name: str = "standard-agent-eval",
    langfuse_host: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_public_key: Optional[str] = None
) -> tuple[trace.Tracer, metrics.Meter]:
    """Setup OpenTelemetry with Langfuse backend."""
    global _tracer, _meter
    
    if _tracer is not None:
        return _tracer, _meter
    
    # Get config from env if not provided
    langfuse_host = langfuse_host or os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    langfuse_secret_key = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    
    # Setup tracing
    trace_provider = TracerProvider()
    trace.set_tracer_provider(trace_provider)
    
    # Add Langfuse span processor
    if langfuse_secret_key and langfuse_public_key:
        langfuse_processor = LangfuseSpanProcessor(
            host=langfuse_host,
            secret_key=langfuse_secret_key,
            public_key=langfuse_public_key
        )
        trace_provider.add_span_processor(langfuse_processor)
    else:
        print("WARNING: Langfuse credentials not found, traces will not be exported")
    
    # Setup metrics
    metric_provider = MeterProvider()
    metrics.set_meter_provider(metric_provider)
    
    # Create tracer and meter
    _tracer = trace.get_tracer(service_name)
    _meter = metrics.get_meter(service_name)
    
    return _tracer, _meter

def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    if _tracer is None:
        tracer, _ = setup_telemetry()
        return tracer
    return _tracer

def get_meter() -> metrics.Meter:
    """Get the global meter instance.""" 
    if _meter is None:
        _, meter = setup_telemetry()
        return meter
    return _meter

# Convenience decorators
def trace_agent_operation(operation_name: str):
    """Decorator to trace agent operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(operation_name) as span:
                # Add basic attributes
                span.set_attribute("operation", operation_name)
                span.set_attribute("function", func.__name__)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Add result preview
                try:
                    result_str = str(result)
                    if len(result_str) > 1000:
                        result_str = result_str[:1000] + "..."
                    span.set_attribute("result_preview", result_str)
                except Exception:
                    pass
                
                return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
```

### **2.3 Create Agent Instrumentation Module**

**Create `evaluation/instrumentation.py`:**

```python
"""Non-intrusive instrumentation for Standard Agent evaluation."""

import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional
from opentelemetry import trace
from .otel_setup import get_tracer, get_meter

# Per-run context for metrics aggregation
current_run: ContextVar[Dict[str, Any]] = ContextVar('current_run', default=None)

def instrument_agent(agent, llm) -> None:
    """Apply non-intrusive instrumentation to agent and LLM."""
    
    # Wrap agent.solve method
    if not hasattr(agent.solve, '_otel_instrumented'):
        original_solve = agent.solve
        agent.solve = _create_solve_wrapper(original_solve)
        agent.solve._otel_instrumented = True
    
    # Set up LLM token callback
    if hasattr(llm, 'usage_callback') and not hasattr(llm, '_otel_instrumented'):
        llm.usage_callback = _token_callback
        llm._otel_instrumented = True

def _create_solve_wrapper(original_solve):
    """Create instrumented wrapper for agent.solve method."""
    tracer = get_tracer()
    meter = get_meter()
    
    # Metrics
    solve_counter = meter.create_counter(
        "agent_solve_total",
        description="Total number of agent solve calls"
    )
    solve_duration = meter.create_histogram(
        "agent_solve_duration_ms",
        description="Duration of agent solve calls in milliseconds"
    )
    solve_tokens = meter.create_histogram(
        "agent_solve_tokens",
        description="Token usage per solve call"
    )
    
    def wrapper(goal: str) -> Any:
        # Initialize run context
        run_id = str(uuid.uuid4())
        run_context = {
            "run_id": run_id,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "start_time": time.time()
        }
        current_run.set(run_context)
        
        # Create span
        with tracer.start_as_current_span("agent_solve") as span:
            # Set span attributes
            span.set_attribute("run_id", run_id)
            span.set_attribute("goal", goal[:500])  # Truncate long goals
            span.set_attribute("goal_length", len(goal))
            
            start_time = time.monotonic()
            
            try:
                # Call original method
                result = original_solve(goal)
                
                # Calculate duration
                duration_ms = (time.monotonic() - start_time) * 1000
                run_context["duration_ms"] = duration_ms
                
                # Set result attributes
                span.set_attribute("success", getattr(result, 'success', True))
                span.set_attribute("duration_ms", duration_ms)
                
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "..."
                span.set_attribute("result_preview", result_str)
                
                # Record metrics
                solve_counter.add(1, {
                    "success": str(getattr(result, 'success', True)),
                    "agent_type": type(original_solve.__self__).__name__
                })
                solve_duration.record(duration_ms)
                
                # Token metrics
                total_tokens = run_context["tokens_prompt"] + run_context["tokens_completion"]
                if total_tokens > 0:
                    span.set_attribute("tokens_prompt", run_context["tokens_prompt"])
                    span.set_attribute("tokens_completion", run_context["tokens_completion"])
                    span.set_attribute("tokens_total", total_tokens)
                    solve_tokens.record(total_tokens)
                
                return result
                
            except Exception as e:
                # Record error
                span.record_exception(e)
                span.set_attribute("error", True)
                solve_counter.add(1, {
                    "success": "false",
                    "error": type(e).__name__
                })
                raise
    
    return wrapper

def _token_callback(prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
    """Callback to accumulate token usage in current run context."""
    run_context = current_run.get()
    if run_context is None:
        return
    
    if prompt_tokens is not None:
        run_context["tokens_prompt"] += prompt_tokens
    if completion_tokens is not None:
        run_context["tokens_completion"] += completion_tokens

def get_current_run_metrics() -> Dict[str, Any]:
    """Get metrics for the current run."""
    run_context = current_run.get()
    if run_context is None:
        return {}
    
    return {
        "run_id": run_context.get("run_id"),
        "duration_ms": run_context.get("duration_ms", 0),
        "tokens_prompt": run_context.get("tokens_prompt", 0),
        "tokens_completion": run_context.get("tokens_completion", 0),
        "tokens_total": run_context.get("tokens_prompt", 0) + run_context.get("tokens_completion", 0)
    }
```

### **2.4 Update Runner to Use OpenTelemetry**

**Modify `evaluation/runner.py`:**

Replace the imports section:
```python
# OLD imports to remove:
# from .tracing import JsonTracer, current_run
# from .hooks import enable_instrumentation

# NEW imports:
from .otel_setup import setup_telemetry
from .instrumentation import instrument_agent, get_current_run_metrics, current_run
```

Replace the instrumentation section in `cmd_run`:
```python
# OLD code to remove:
# tracer = JsonTracer(spans_path)
# enable_instrumentation(agent, llm, tracer)

# NEW code:
# Setup OpenTelemetry + Langfuse
tracer, meter = setup_telemetry("standard-agent-eval")
print(f"✓ OpenTelemetry configured, traces will be sent to {os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}")

# Instrument agent and LLM
instrument_agent(agent, llm)
print("✓ Agent and LLM instrumented")
```

Replace the metrics collection section:
```python
# OLD code to remove:
# run_state = current_run.get() or {}
# time_ms = run_state.get("time_ms", 0)
# tokens_prompt = run_state.get("ptok", None)
# tokens_completion = run_state.get("ctok", None)

# NEW code:
metrics = get_current_run_metrics()
time_ms = metrics.get("duration_ms", 0)
tokens_prompt = metrics.get("tokens_prompt") if metrics.get("tokens_prompt", 0) > 0 else None
tokens_completion = metrics.get("tokens_completion") if metrics.get("tokens_completion", 0) > 0 else None
```

---

## 🧪 **Phase 3: Integration & Testing**

### **3.1 Basic Smoke Test**

**Create `evaluation/test_otel.py`:**

```python
#!/usr/bin/env python3
"""Smoke test for OpenTelemetry + Langfuse integration."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.otel_setup import setup_telemetry, trace_agent_operation
from opentelemetry import trace

def test_basic_tracing():
    """Test basic span creation."""
    print("🧪 Testing basic OpenTelemetry tracing...")
    
    # Setup telemetry
    tracer, meter = setup_telemetry("test-service")
    
    # Create a test span
    with tracer.start_as_current_span("test_operation") as span:
        span.set_attribute("test_attr", "test_value")
        span.set_attribute("number_attr", 42)
        print("✓ Created test span with attributes")
    
    print("✓ Basic tracing test passed")

@trace_agent_operation("test_decorated_function")
def test_decorated_function():
    """Test the decorator."""
    return "test result"

def test_decorator():
    """Test the trace decorator."""
    print("🧪 Testing trace decorator...")
    result = test_decorated_function()
    assert result == "test result"
    print("✓ Decorator test passed")

def test_langfuse_connection():
    """Test connection to Langfuse."""
    print("🧪 Testing Langfuse connection...")
    
    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY")
    
    if not langfuse_secret or not langfuse_public:
        print("⚠️  Langfuse credentials not found, skipping connection test")
        return
    
    try:
        import requests
        response = requests.get(f"{langfuse_host}/api/public/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Langfuse is accessible at {langfuse_host}")
        else:
            print(f"❌ Langfuse returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to Langfuse: {e}")

if __name__ == "__main__":
    print("🚀 Starting OpenTelemetry smoke tests...\n")
    
    test_basic_tracing()
    test_decorator()
    test_langfuse_connection()
    
    print("\n✅ All tests completed!")
```

**Run the test:**
```bash
python evaluation/test_otel.py
```

### **3.2 Test with Actual Agent**

**Run evaluation with single item:**
```bash
python -m evaluation.runner run \
    --dataset evaluation/datasets/smoke_nyt.jsonl \
    --agent react \
    --config evaluation/configs/smoke_config.json \
    --limit 1
```

**Expected output:**
```
✓ OpenTelemetry configured, traces will be sent to http://localhost:3000
✓ Agent and LLM instrumented
Processing item 1/1: nyt-1
✓ Run completed: success=True, duration=1234ms, tokens=567
Wrote records to: eval_runs/smoke_nyt__2025-09-04__5a5df609.jsonl
```

### **3.3 Verify Langfuse Dashboard**

1. **Open Langfuse:** http://localhost:3000
2. **Check Traces tab:** Should see spans for agent_solve
3. **Check Metrics:** Should see token usage and duration
4. **Verify Attributes:** run_id, goal, success, tokens, etc.

---

## 🧹 **Phase 4: Cleanup**

### **4.1 Remove Old Files**

```bash
# Remove old dashboard files (already deleted in Phase 2)
# Clean up any remaining references
grep -r "tracing.py" evaluation/ || echo "✓ No references to old tracing.py"
grep -r "hooks.py" evaluation/ || echo "✓ No references to old hooks.py"
```

### **4.2 Update Documentation**

**Update `evaluation/README.md`:**

```markdown
# Standard Agent Evaluation Framework

## Overview

This framework provides non-intrusive evaluation and observability for Standard Agent using industry-standard OpenTelemetry + Langfuse.

## Features

- ✅ **OpenTelemetry Integration** - Industry standard tracing and metrics
- ✅ **Langfuse Dashboard** - Professional UI for LLM observability  
- ✅ **Non-intrusive** - No changes to core Standard Agent code
- ✅ **Custom Metrics** - Token usage, success rates, duration tracking
- ✅ **Self-hosted** - Complete data ownership

## Quick Start

1. **Setup Langfuse:**
   ```bash
   docker-compose up -d
   ```

2. **Configure environment:**
   ```bash
   export LANGFUSE_SECRET_KEY=sk-lf-...
   export LANGFUSE_PUBLIC_KEY=pk-lf-...
   export LANGFUSE_HOST=http://localhost:3000
   ```

3. **Run evaluation:**
   ```bash
   python -m evaluation.runner run \
       --dataset path/to/dataset.jsonl \
       --agent react \
       --config path/to/config.json
   ```

4. **View results:** Open http://localhost:3000

## Architecture

- **OpenTelemetry** - Handles tracing, metrics, and instrumentation
- **Langfuse** - LLM-aware observability backend and dashboard
- **Non-intrusive** - Uses decorators and callbacks, no core code changes

## Custom Metrics

The framework automatically tracks:
- **Success Rate** - Based on ReasoningResult.success
- **Duration** - Wall-clock time for agent.solve()
- **Token Usage** - Prompt, completion, and total tokens
- **Custom Attributes** - Goal length, agent type, run metadata
```

### **4.3 Clean Up Dependencies**

**Update main requirements or pyproject.toml:**
```toml
# Remove old dependencies:
# streamlit
# plotly

# Add new dependencies:
opentelemetry-api = "^1.20.0"
opentelemetry-sdk = "^1.20.0" 
opentelemetry-exporter-otlp = "^1.20.0"
langfuse = "^2.0.0"
```

---

## ✅ **Success Criteria**

**Migration is complete when:**

- [ ] **All traces appear in Langfuse dashboard**
- [ ] **Custom metrics are captured** (tokens, duration, success)
- [ ] **No custom tracing code remains** (deleted old files)
- [ ] **Smoke test passes** with real agent evaluation
- [ ] **Documentation is updated** with new usage patterns
- [ ] **Dependencies are cleaned up** (removed streamlit, added otel)

**Performance expectations:**
- **Faster startup** - No custom framework initialization
- **Lower memory usage** - Optimized OpenTelemetry implementation  
- **Better reliability** - Battle-tested libraries instead of custom code

**Code reduction:**
- **Before:** ~500 lines of custom tracing/hooks/dashboard code
- **After:** ~100 lines of OpenTelemetry setup and instrumentation
- **Net reduction:** 80% less code to maintain

---

## 🚀 **Next Steps After Migration**

Once migration is complete, you can:

1. **Explore Langfuse features** - Datasets, A/B testing, cost tracking
2. **Add more custom metrics** - Model performance, tool usage, etc.
3. **Set up alerts** - For success rate drops or performance regressions
4. **Scale to multiple agents** - Compare different agent configurations
5. **Team collaboration** - Share dashboard with other developers

The migration gives you a production-ready evaluation system with minimal maintenance overhead.

