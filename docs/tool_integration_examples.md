# Tool Integration Examples

This guide walks through three ways to plug custom tools into the Standard Agent framework. Each example
corresponds to real code under `examples/tools/integration/` and is exercised by automated tests so you can trust
that the snippets work end to end.

> ℹ️ If you are new to the tool interfaces, start by reviewing `agents/tools/base.py`. The `ToolBase` class describes
> what an individual tool exposes, while `JustInTimeToolingBase` defines the provider that discovers, loads, and
> executes tools at runtime.

- [Example 1 – Local temperature converter](#example-1--local-temperature-converter)
- [Example 2 – REST weather lookup](#example-2--rest-weather-lookup)
- [Example 3 – Allow-listed shell commands](#example-3--allow-listed-shell-commands)

All three follow the same pattern:

1. Describe the tool (`ToolBase` implementation) so the LLM knows the capability and parameters.
2. Expose the tool through a provider (`JustInTimeToolingBase`). The provider decides when the tool is returned from
   `search`, how it is hydrated in `load`, and how to execute it safely.
3. Wire the provider into `StandardAgent` together with the LLM, memory, and reasoner of your choice.

---

## Example 1 – Local temperature converter

**Goal:** keep the entire computation inside the agent process. This is ideal for pure functions, feature toggles, or
anything that does not need I/O.

**Code reference:** [`examples/tools/integration/local_temperature.py`](../examples/tools/integration/local_temperature.py)

### Step-by-step (local tool)

1. **Describe the tool.** `TemperatureConversionTool` states what the tool does and which parameters (`value`,
   `from_unit`, `to_unit`) are expected.
2. **Expose the tool.** `LocalTemperatureTools.search` only returns the converter for relevant queries. `execute`
   performs the unit conversion in pure Python.
3. **Use the tool in an agent.**

```python
from agents.llm.litellm import LiteLLM
from agents.memory.dict_memory import DictMemory
from agents.reasoner.react import ReACTReasoner
from agents.standard_agent import StandardAgent

from examples.tools.integration.local_temperature import LocalTemperatureTools

llm = LiteLLM(model="gpt-4o-mini")
tools = LocalTemperatureTools()
memory = DictMemory()
reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)

agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner)
```

**Test coverage:** `tests/examples/test_tool_integration_examples.py::test_local_temperature_tool_executes` ensures the
conversion returns the expected value.

---

## Example 2 – REST weather lookup

**Goal:** forward parameters to an HTTP API and map the response back into structured data the agent can reason about.

**Code reference:** [`examples/tools/integration/weather_api.py`](../examples/tools/integration/weather_api.py)

### Step-by-step (REST tool)

1. **Describe the REST call.** `WeatherAPITool` documents the required `location` argument and optional `units`.
2. **Create a provider that knows how to talk to the API.** `WeatherAPIClient` injects a `requests.Session` (or any
   drop-in implementation) and handles authentication, timeouts, and response parsing.
3. **Execute safely.** `execute` builds the query string, raises if `location` is missing, and returns a normalized
   dictionary with `location`, `temperature`, `conditions`, and the raw payload for debugging.

```python
from agents.llm.litellm import LiteLLM
from agents.memory.dict_memory import DictMemory
from agents.reasoner.rewoo import ReWOOReasoner
from agents.standard_agent import StandardAgent

from examples.tools.integration.weather_api import WeatherAPIClient

llm = LiteLLM(model="gpt-4o-mini")
tools = WeatherAPIClient(base_url="https://api.example.com", api_key="YOUR_KEY")
memory = DictMemory()
reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)

agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner)
```

**Testing strategy:** the suite fakes the HTTP client so no real network access is required. See
`tests/examples/test_tool_integration_examples.py::test_weather_api_tool_executes`.

---

## Example 3 – Allow-listed shell commands

**Goal:** expose carefully curated shell commands (or other system integrations) while keeping guardrails in place.

**Code reference:** [`examples/tools/integration/shell_command.py`](../examples/tools/integration/shell_command.py)

### Step-by-step (system tool)

1. **Define the metadata.** `ShellCommandTool` declares that the tool executes a command with optional arguments and a
   timeout.
2. **Create a provider with safety checks.** `ShellCommandTools` accepts an allow-list and a `runner` callable so you
   can inject a stub for testing. The helper `format_command` renders a user-friendly command string for prompts.
3. **Execute with a controlled runtime.** Only commands present in the allow-list are executed via `subprocess.run` with
   captured stdout/stderr.

```python
from agents.llm.litellm import LiteLLM
from agents.memory.dict_memory import DictMemory
from agents.reasoner.react import ReACTReasoner
from agents.standard_agent import StandardAgent

from examples.tools.integration.shell_command import ShellCommandTools

llm = LiteLLM(model="gpt-4o-mini")
tools = ShellCommandTools(allow_list=["uptime", "echo"])
memory = DictMemory()
reasoner = ReACTReasoner(llm=llm, tools=tools, memory=memory)

agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner)
```

**Test coverage:** `tests/examples/test_tool_integration_examples.py::test_shell_command_tool_executes_when_allowed`
verifies the allow-list logic and output handling.

---

## Tips for your own integrations

- Keep provider constructors injectable (e.g., accept an HTTP session or subprocess runner). This makes them testable and
  easier to reuse in other environments.
- Return structured dictionaries from `execute`. The reasoners and summarizer prompts work best when they receive clean
  JSON-like objects rather than large strings.
- Lean on the agent’s memory for state. You can store intermediate results or rate-limiting information in the
  `MutableMapping` passed to `StandardAgent`.
- Add tests! All examples above have focused unit tests that validate the happy path and failure modes without relying on
  network or system side effects.
