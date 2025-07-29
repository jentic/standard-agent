# Standard Agent 🛠️ — Composable Agents

A **modular framework** for building AI agents that can plan, act, and **autonomously recover from failures**.
It ships with a ready-to-use *ReWOO* reasoning stack and the Jentic tool platform out of the box, but every layer is swappable.

- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Layout](#project-layout)
- [Core Runtime Objects](#core-runtime-objects)
- [Extending the Framework](#extending-the-framework)
- [Roadmap](#roadmap)

### Quick Start

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
python main.py
```

### Configuration

Before running the agent, you need to create a `.env` file in the root of the project to store your API keys and other secrets. The application will automatically load these variables.

Create a file named `.env` and add the following content, replacing the placeholder values with your actual keys:

```dotenv
# Jentic Platform API Key
JENTIC_API_KEY="your-jentic-api-key-here"

# LLM Provider API Keys (use the one for your chosen model)
OPENAI_API_KEY="your-openai-api-key-here"
ANTHROPIC_API_KEY="your-anthropic-api-key-here"
GEMINI_API_KEY="your-google-gemini-api-key-here"

# Tool-Specific Secrets (add as needed)
DISCORD_BOT_TOKEN="your-discord-bot-token-here"
```

**Note:** The `JENTIC_API_KEY` and at least one LLM provider key are essential for the agent to function.

You can obtain a jentic key by running the following line from your project directory with the virtual enviroment active:
``` bash
jentic register --email '<your_email>'
```

### Usage Examples

We provide two ways to use the agent framework: a quick-start method using a pre-built agent, and a more advanced method that shows how to build an agent from scratch.

#### 1. Quick Start: Running a Pre-built Agent

This is the fastest way to get started. The `ReWOOAgent` class provides a `StandardAgent` instance that is already configured with a powerful reasoner, LLM, tools, and memory.

```python
# main.py
import os
from dotenv import load_dotenv
from agents.prebuilt import ReWOOAgent
from utils.cli import read_user_goal, print_result

# Load API keys from .env file
load_dotenv()

# 1. Get the pre-built agent.
agent = ReWOOAgent(model=os.getenv("LLM_MODEL", "claude-sonnet-4"))

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
from agents.reasoner.sequential.reasoner import SequentialReasoner
from agents.reasoner.sequential.planners.bullet_list import BulletListPlan
from agents.reasoner.sequential.executors.rewoo import ReWOOExecuteStep
from agents.reasoner.sequential.reflectors.rewoo import ReWOOReflect
from agents.reasoner.sequential.summarizer.default import DefaultSummarizeResult

from utils.cli import read_user_goal, print_result

load_dotenv()

# Step 1: Choose and configure your components
llm = LiteLLM(model="gpt-4")
tools = JenticClient()
memory = DictMemory()

# Step 2: Build a custom reasoner by composing sequential components
custom_reasoner = SequentialReasoner(
    llm=llm,
    tools=tools, 
    memory=memory,
    plan=BulletListPlan(llm=llm),
    execute_step=ReWOOExecuteStep(llm=llm, tools=tools, memory=memory),
    reflect=ReWOOReflect(llm=llm, tools=tools, memory=memory, max_retries=5),  # More retries
    summarize_result=DefaultSummarizeResult(llm=llm)
)

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
│       ├── prebuilt.py             # Pre-composed, ready-to-use reasoner implementations
│       └── sequential/             # A step-by-step reasoner (Plan -> Execute -> Reflect)
│           ├── reasoner.py         # Orchestrates the sequential reasoning loop
│           ├── planners/           # Components for generating plans
│           ├── executors/          # Components for executing single steps of a plan
│           ├── reflectors/         # Components for analyzing failures and self-healing
│           └── summarizer/         # Components for summarizing final results
│
├── utils/
│   ├── cli.py                      # Command-line interface helpers
│   └── logger.py                   # Logging configuration
│
├── tests/                          # Unit and integration tests
├── main.py                         # Main entry point for running the agent
├── Makefile                        # Commands for installation, testing, etc.
├── requirements.txt                # Project dependencies
└── config.json                     # Agent configuration file
```

---

## ✨  Key Principles
| Principle | What it means in Standard Agent                                                                     |
|-----------|-----------------------------------------------------------------------------------------------------|
| **Composition** | Small, focused components are wired together at runtime.                                            |
| **Explicit DI** | LLM, Memory and Tools are injected once by the `StandardAgent` and broadcast to all sub-components. |
| **Swappable everything** | Swap reasoning strategies, memory back-ends or tool providers without touching agent logic.         |
| **Zero-boilerplate CLI** | A fully working CLI agent is  ~40 lines of glue code.                                               |
| **Self-healing** | Reflector components analyse errors, edit the plan and retry automatically.                         |

---

### Core Runtime Objects

| Layer            | Class / Protocol                                                | Notes                                                                    |
|------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------|
| **Agent**        | `StandardAgent`                                                 | Owns LLM, Memory, and Tools; injects them into a Reasoner.               |
| **Reasoners**    | `SequentialReasoner`, `TreeSearchReasoner`, etc.                | Each orchestrates a different reasoning algorithm.                       |
| **Memory**       | `BaseMemory`                               | A key-value store accessible to all components.                          |
| **Tools**        | `JustInTimeToolingBase`                  | Abstracts external actions (APIs, shell commands, etc.).                 |
| **Inbox / Outbox** | `BaseInbox` / `BaseOutbox`                                      | Decouples I/O, allowing the agent to run in any environment.             |
| **LLM Wrapper**  | `BaseLLM`                                     | Provides a uniform interface for interacting with different LLMs.        |

### The Sequential Reasoner

The `SequentialReasoner` is the default reasoning engine. It follows a classic **Plan -> Execute -> Reflect** loop, and its logic is broken down into four distinct, swappable components:

- **Plan**: Takes the user's goal and generates a step-by-step plan.
  - *Example*: `BulletListPlan`
- **Step Executor**: Executes a single step from the plan, often by calling a tool.
  - *Example*: `ReWOOExecuteStep`
- **Reflector**: If a step fails, this component analyzes the error and decides how to recover (e.g., retry, change the plan).
  - *Example*: `ReWOOReflector`
- **Summarizer**: Once the plan is complete, this component synthesizes the final answer for the user.
  - *Example*: `DefaultSummarizeResult`

This design allows you to customize the reasoning process by mixing and matching different implementations for each stage.

### Extending the Framework

The framework is designed to be modular. Here are some common extension points:

| Need                               | How to Implement                                                                                                                                                                     |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Different reasoning strategy**   | Create a new `BaseReasoner` implementation (e.g., `TreeSearchReasoner`) and inject it into `StandardAgent`.                                                                          |
| **Custom sequential logic**        | Create new `Plan`, `ExecuteStep`, `Reflect`, or `SummarizeResult` components and compose your own `SequentialReasoner`.                                                              |
| **New tool provider**              | Create a class that inherits from `JustInTimeToolingBase`, implement its methods, and pass it to your `StandardAgent`.                                                               |
| **Persistent memory**              | Create a class that implements the `MutableMapping` interface (e.g., using Redis), and pass it to your `StandardAgent`.                                                              |
| **New Planners, Executors, etc.**  | Create your own implementations of `Plan`, `ExecuteStep`, `Reflect`, or `SummarizeResult` to invent new reasoning capabilities, then compose them in a `SequentialReasoner`. |

## 🔮 Roadmap

- Async agent loop & concurrency-safe inboxes
- Additional pre-built reasoner implementations (ReAct, ToT, Graph-of-Thought)
- More out of the box composable parts to enable custom agents or reasoner implementations
- Web dashboard (live agent state + logs)
- Vector-store memory with RAG planning
- Slack / Discord integration
- Redis / VectorDB memory
- Ideas are welcome! [Open an issue](https://github.com/jentic/standard-agent/issues) or [submit a pull request](https://github.com/jentic/standard-agent/pulls).
