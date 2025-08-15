# Standard Agent 🛠️ — Composable Agents


[![Discord](https://img.shields.io/badge/JOIN%20OUR%20DISCORD-COMMUNITY-7289DA?style=plastic&logo=discord&logoColor=white)](https://discord.gg/yrxmDZWMqB)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-40c463.svg)](https://github.com/jentic/standard-agent/blob/HEAD/CODE_OF_CONDUCT.md)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/jentic/standard-agent/blob/HEAD/LICENSE)


- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Layout](#project-layout)
- [Core Runtime Objects](#core-runtime-objects)
- [Extending the Library](#extending-the-library)
- [Roadmap](#roadmap)

> **Join our community!** Connect with contributors and users on [Discord](https://discord.gg/yrxmDZWMqB) to discuss ideas, ask questions, and collaborate on the Standard Agent repository.

## Architecture Overview

Standard Agent is a simple, modular library for building AI agents—with a composable core and plug‑in components.

![Standard Agent architecture](docs/assets/standard_agent_architecture.png)


## Quick Start

### Installation

```bash
# Clone and set up the project
git clone <repository-url>
cd standard_agent

# Install dependencies
make install

# Activate the virtual environment
source .venv/bin/activate

# Run the agent
python examples/cli_rewoo_api_agent.py
```
### Configuration

Before running the agent, you need to create a `.env` file in the root of the project to store your API keys and other secrets. The application will automatically load these variables.

**Quick Setup:**
1. Copy the provided template: `cp .env.example .env`
2. Edit the `.env` file and replace placeholder values with your actual API keys
3. At minimum, you need one LLM provider key to get started
4. Add JENTIC_API_KEY for out-of-the-box tool access (recommended)

See [.env.example](./.env.example) for the complete configuration template with detailed comments and setup instructions.

**Key Requirements:**
- **LLM Model**: `LLM_MODEL` - Choose your preferred model
- **LLM Provider**: At least one API key (Anthropic, OpenAI, or Google)
- **Tool Provider**: `JENTIC_AGENT_API_KEY` for instant access to 1500+ tools (get yours at [app.jentic.com](https://app.jentic.com))


### Usage Examples

We provide two ways to use the agent library: a quick-start method using a pre-built agent, and a more advanced method that shows how to build an agent from scratch.

#### 1. Quick Start: Running a Pre-built Agent

This is the fastest way to get started. The `ReWOOAgent` and `ReACTAgent` classes provide `StandardAgent` instances that are already configured with a reasoner, LLM, tools, and memory.

```python
# examples/cli_api_agent.py
import os
from dotenv import load_dotenv
from agents.prebuilt import ReWOOAgent, ReACTAgent
from examples._cli_helpers import read_user_goal, print_result

# Load API keys from .env file
load_dotenv()

# 1. Get the pre-built agent.
# Choose a prebuilt profile (ReWOO or ReACT)
agent = ReWOOAgent(model=os.getenv("LLM_MODEL"))
# agent = ReACTAgent(model=os.getenv("LLM_MODEL"))

# 2. Run the agent's main loop.
print("🤖 Agent is ready. Press Ctrl+C to exit.")
while True:
    goal_text = None
    try:
        goal = read_user_goal()
        if not goal:
            continue
        
        result = agent.solve(goal)
        print_result(result)

    except KeyboardInterrupt:
        print("\n🤖 Bye!")
        break
```

#### 2. Custom Agent Composition: Build Your Own

The real power of Standard Agent comes from its **composable architecture**. Every component is swappable, allowing you to create custom agents tailored to your specific needs. Here's how to build agents from scratch by mixing and matching components.

```python
# main_build_your_own_agent.py
import os
from dotenv import load_dotenv

# Import the core agent class
from agents.standard_agent import StandardAgent

# Import different implementations for each layer
from agents.llm.litellm import LiteLLM
from agents.tools.jentic import JenticClient
from agents.memory.dict_memory import DictMemory

# Import reasoner components
from agents.reasoner.rewoo import ReWOOReasoner

from examples._cli_helpers import read_user_goal, print_result

load_dotenv()

# Step 1: Choose and configure your components
llm = LiteLLM(model="gpt-4")
tools = JenticClient()
memory = DictMemory()

# Step 2: Pick a reasoner profile (single-file implementation)
custom_reasoner = ReWOOReasoner(llm=llm, tools=tools, memory=memory)

# Step 3: Wire everything together in the StandardAgent
agent = StandardAgent(
    llm=llm,
    tools=tools,
    memory=memory,
    reasoner=custom_reasoner
)

# Step 4: Use your custom agent
print("🤖 Custom Agent is ready!")
while True:
    goal_text = None
    try:
        goal = read_user_goal()
        if not goal:
            continue
            
        result = agent.solve(goal)
        print_result(result)
        
    except KeyboardInterrupt:
        print("\n🤖 Bye!")
        break
```
---

**💡 Why This Matters**

This composition approach means you can:

- **Start simple** with pre-built agents like `ReWOOAgent`
- **Gradually customize** by swapping individual components
- **Experiment easily** with different LLMs, reasoning strategies, or tool providers
- **Extend incrementally** by implementing new components that follow the same interfaces
- **Mix and match** components from different sources without breaking existing code

The key insight is that each component follows well-defined interfaces (`BaseLLM`, `BaseMemory`, `JustInTimeToolingBase`, etc.), so they can be combined in any configuration that makes sense for your use case.

### Project Layout

```
.
├── agents/
│   ├── standard_agent.py           # The main agent class orchestrating all components
│   ├── prebuilt.py                 # Factory functions for pre-configured agents (e.g., ReWOO)
│   ├── llm/                        # LLM wrappers (e.g., LiteLLM)
│   ├── memory/                     # Memory backends (e.g., in-memory dictionary)
│   ├── tools/                      # Tool integrations (e.g., Jentic client)
│   └── reasoner/                   # Core reasoning and execution logic
│       ├── base.py                 # Base classes and interfaces for reasoners
│       ├── rewoo.py                # ReWOO (Plan → Execute → Reflect)
│       └── react.py                # ReACT (Think → Act)
│   ├── goal_preprocessor/          # [OPTIONAL] Goal preprocessor
│
├── utils/
│   └── logger.py                   # Logging configuration
│
├── examples/                       # Runnable scripts and helper snippets
│
├── tests/                          # Unit and integration tests
├── Makefile                        # Commands for installation, testing, etc.
├── requirements.txt                # Project dependencies
└── config.json                     # Agent configuration file
```

### Core Runtime Objects

| Layer            | Class / Protocol                                                     | Notes                                                             |
|------------------|----------------------------------------------------------------------|-------------------------------------------------------------------|
| **Agent**        | `StandardAgent`                                                      | Owns Reasoner, LLM, Memory, and Tools                             |
| **Reasoners**    | `ReWOOReasoner`, `ReACTReasoner`                                      | Each orchestrates a different reasoning strategy (profile).       |
| **Memory**       | `MutableMapping`                                                         | A key-value store accessible to all components.                   |
| **Tools**        | `JustInTimeToolingBase`                                              | Abstracts external actions (APIs, shell commands, etc.).          |
| **LLM Wrapper**  | `BaseLLM`                                                            | Provides a uniform interface for interacting with different LLMs. |
| **Goal Preprocessor** | `BaseGoalPreprocessor`                                            | [Optional] Preprocess goals before reasoning                      |


### Reasoner Profiles

The library ships two reasoner profiles:

- **ReWOOReasoner** (`agents/reasoner/rewoo.py`): Plan → Execute → Reflect
- **ReACTReasoner** (`agents/reasoner/react.py`): Think → Act 

Each profile exposes a `run(goal: str) -> ReasoningResult` and produces a `transcript`. The agent synthesizes the final answer from the transcript.

Note on the reasoning spectrum:

- "Explicit" reasoners externalize structure (plans, intermediate steps) for transparency and control. ReWOO is more explicit.
- "Implicit" reasoners decide the next move with minimal externalized structure for speed and brevity. ReACT is more implicit.

We welcome contributions of new reasoning strategies anywhere on this spectrum. If you add a profile, keep it as a single-file `BaseReasoner` implementation and define its prompts in YAML under `agents/prompts/reasoners/`.

### Extending the Library

The library is designed to be modular. Here are some common extension points:

| Need                               | How to Implement                                                                                                                                                                     |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Different reasoning strategy**   | Create a new `BaseReasoner` implementation (e.g., `TreeSearchReasoner`) and inject it into `StandardAgent`.                                                                          |
| **New tool provider**              | Create a class that inherits from `JustInTimeToolingBase`, implement its methods, and pass it to your `StandardAgent`.                                                               |
| **Persistent memory**              | Create a class that implements the `MutableMapping` interface (e.g., using Redis), and pass it to your `StandardAgent`.                                                              |
| **New Planners, Executors, etc.**  | Create your own implementations of `Plan`, `ExecuteStep`, `Reflect`, or `SummarizeResult` to invent new reasoning capabilities, then compose them in a `SequentialReasoner`. |
| **Pre-process or validate goals**  | Create a class that inherits from `BaseGoalPreprocessor` and pass it to `StandardAgent`. Use this to resolve conversational ambiguities, check for malicious intent, or sanitize inputs. |


## Roadmap

- Additional pre-built reasoner implementations (ReAct, ToT, Graph-of-Thought)
- More out of the box composable parts to enable custom agents or reasoner implementations
- Web dashboard (live agent state + logs)
- Vector-store memory with RAG planning
- Slack / Discord integration
- Redis / VectorDB memory
- Async agent loop & concurrency-safe inboxes
- Ideas are welcome! [Open an issue](https://github.com/jentic/standard-agent/issues) or [submit a pull request](https://github.com/jentic/standard-agent/pulls).
