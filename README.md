# Standard Agent 🛠️ — Composable Agents

A **modular framework** for building AI agents that can plan, act, and **autonomously recover from failures**.  
It ships with a ready-to-use *ReWOO* reasoning stack and the Jentic tool platform out of the box, but every layer is swappable
[![Discord](https://img.shields.io/badge/JOIN%20OUR%20DISCORD-COMMUNITY-7289DA?style=plastic&logo=discord&logoColor=white)](https://discord.gg/yrxmDZWMqB)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-40c463.svg)](CODE_OF_CONDUCT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Layout](#project-layout)
- [Core Runtime Objects](#core-runtime-objects)
- [Extending the Framework](#extending-the-framework)
- [Roadmap](#roadmap)



> **Join our community!** Connect with contributors and users on [Discord](https://discord.gg/yrxmDZWMqB) to discuss ideas, ask questions, and collaborate on the OAK repository.


## Quick Start

### Get Your Jentic API Key

To use any Jentic product such as the Jentic SDK or MCP Plugin, you must first obtain a Jentic API Key. The easiest way is using the Jentic CLI. You can _optionally_ include an email address for higher rate limits and for early access to new features.

```sh
jentic register --email '<your_email>'
```

This will print your API Key and an export command to set it in your environment:

```sh
export JENTIC_API_KEY=<your-jentic-uuid>
```

Alternatively, you can use curl to register and obtain your API Key:

```sh
curl -X POST https://api.jentic.com/api/v1/auth/register \
     -H "Content-Type: application/json" \
     -d '{"email": "<your_email>"}'
```

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

### Usage Examples

We provide two ways to use the agent framework: a quick-start method using a pre-built agent, and a more advanced method that shows how to build an agent from scratch.

#### 1. Quick Start: Running a Pre-built Agent

This is the fastest way to get started. The `get_rewoo_agent` factory provides a `StandardAgent` instance that is already configured with a powerful reasoner, LLM, tools, and memory.

```python
# main.py
import os, time
from agents.prebuilt_agents import get_rewoo_agent
from inbox.cli_inbox import CLIInbox
from outbox.cli_outbox import CLIOutbox

# 1. Get the pre-built agent. API keys are loaded from your .env file.
agent = get_rewoo_agent(model=os.getenv("LLM_MODEL", "claude-sonnet-4"))

# 2. Set up the inbox and outbox for command-line interaction.
inbox = CLIInbox(prompt="🤖 Enter your goal: ")
outbox = CLIOutbox()

# 3. Run the agent's main loop.
print("Agent is ready. Press Ctrl+C to exit.")
while True:
    agent.tick(inbox, outbox)
    time.sleep(1.0)
```

#### 2. Build Your Own Agent & Reasoner

This example demonstrates the framework's true flexibility. It shows how to construct a `SequentialReasoner` from its individual components and then wire it into a `StandardAgent`.

```python
# main_build_your_own_agent.py
import os, time
from agents.standard_agent import StandardAgent
from llm.lite_llm import LiteLLMChatLLM
from tools.jentic_toolkit.jentic_tool_iface import JenticToolInterface
from memory.scratch_pad import ScratchPadMemory
from inbox.cli_inbox import CLIInbox
from outbox.cli_outbox import CLIOutbox

# Import the reasoner and its components
from reasoners.sequential.reasoner import SequentialReasoner
from reasoners.sequential.planners.bullet_list_planner import BulletListPlanner
from reasoners.sequential.step_executors.rewoo_step_executor import ReWOOStepExecutor
from reasoners.sequential.reflectors.rewoo_reflector import ReWOOReflector
from reasoners.sequential.answer_builder.final_answer_builder import FinalAnswerBuilder

# 1. Manually assemble the agent's high-level components.
llm = LiteLLMChatLLM(model=os.getenv("LLM_MODEL", "claude-sonnet-4"))
tools = JenticToolInterface()  # Will use JENTIC_API_KEY from .env
memory = ScratchPadMemory()

# 2. Compose the SequentialReasoner from its parts.
reasoner = SequentialReasoner(
  planner=BulletListPlanner(),
  step_executor=ReWOOStepExecutor(),
  reflector=ReWOOReflector(max_retries=2),
  answer_builder=FinalAnswerBuilder(),
)

# 3. Instantiate the StandardAgent with your custom-built reasoner.
agent = StandardAgent(llm=llm, tools=tools, memory=memory, reasoner=reasoner)

# 4. Set up the inbox and outbox.
inbox = CLIInbox(prompt="🤖 Enter your goal: ")
outbox = CLIOutbox()

# 5. Run the agent's main loop.
print("Custom agent is ready. Press Ctrl+C to exit.")
while True:
  agent.tick(inbox, outbox)
  time.sleep(1.0)
```

### Project Layout

```
.
├── agents/                         # High-level agent orchestration
│   ├── models.py                   # Defines Goal, AgentState
│   └── standard_agent.py           # The StandardAgent
│   └── prebuilt_agents.py          # Prebuilt agents like ReWOO etc
│
├── reasoners/                      # Core reasoning logic
│   └── prebuilt_reasoners.py       # Precomposed ready to use reasoners
│   ├── models.py                   # Defines Step, ReasonerState, etc.
│   └── sequential/                 # Implementation of a sequential reasoner
│       ├── reasoner.py             # Orchestrates the plan -> execute -> reflect loop
│       ├── interface.py            # Defines Planner, StepExecutor, Reflector contracts
│       ├── planners/               # Concrete Planner implementations (e.g., BulletListPlanner)
│       ├── step_executors/         # Concrete StepExecutor implementations (e.g., ReWOOStepExecutor)
│       └── reflectors/             # Concrete Reflector implementations (e.g., ReWOOReflector)
│       └── answer_builder/         # Concrete AnswerBuilder implementations (e.g., FinalAnswerBuilder)
│
├── tools/                          # Abstractions for actions the agent can take
│   ├── interface.py                # Defines the core ToolInterface contract
│   ├── exceptions.py               # Defines ToolExecutionError
│   └── jentic_toolkit/             # Concrete implementation for the Jentic platform
│       ├── jentic_client.py        # Low-level wrapper for the Jentic SDK
│       └── jentic_tool_iface.py    # Maps Jentic tools to the ToolInterface
│
├── llm/                            # Wrappers for different Language Model providers
│   ├── base_llm.py                 # Defines the abstract BaseLLM interface
│   └── lite_llm.py                 # Concrete implementation using the LiteLLM library
│
├── memory/                         # Pluggable memory backends for the agent
│   ├── base_memory.py              # Defines the abstract BaseMemory interface
│   └── scratch_pad.py              # A simple, in-memory dictionary implementation
│
├── inbox/                          # How the agent receives goals 
│   ├── base_inbox.py
│   └── cli_inbox.py
│
├── outbox/                         # How the agent delivers results 
│   ├── base_outbox.py
│   └── cli_outbox.py
│
└── main.py                         # Example entry point to run a CLI-based agent
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
| **Tools**        | `ToolInterface`                          | Abstracts external actions (APIs, shell commands, etc.).                 |
| **Inbox / Outbox** | `BaseInbox` / `BaseOutbox`                                      | Decouples I/O, allowing the agent to run in any environment.             |
| **LLM Wrapper**  | `BaseLLM`                                     | Provides a uniform interface for interacting with different LLMs.        |

### The Sequential Reasoner

The `SequentialReasoner` is the default reasoning engine. It follows a classic **Plan -> Execute -> Reflect** loop, and its logic is broken down into four distinct, swappable components:

- **Planner**: Takes the user's goal and generates a step-by-step plan.
  - *Example*: `BulletListPlanner`
- **Step Executor**: Executes a single step from the plan, often by calling a tool.
  - *Example*: `ReWOOStepExecutor`
- **Reflector**: If a step fails, this component analyzes the error and decides how to recover (e.g., retry, change the plan).
  - *Example*: `ReWOOReflector`
- **Answer Builder**: Once the plan is complete, this component synthesizes the final answer for the user.
  - *Example*: `FinalAnswerBuilder`

This design allows you to customize the reasoning process by mixing and matching different implementations for each stage.

### Extending the Framework

The framework is designed to be modular. Here are some common extension points:

| Need                               | How to Implement                                                                                             |
|------------------------------------|--------------------------------------------------------------------------------------------------------------|
| **Different reasoning strategy**   | Create a new `BaseReasoner` implementation (e.g., `TreeSearchReasoner`) and inject it into `StandardAgent`.      |
| **Custom planner**                 | Sub-class `BasePlanner`, place it in `reasoners/sequential/planners/`, and wire it into your `SequentialReasoner`. |
| **Slack / Discord integration**    | Implement a `SlackOutbox` by sub-classing `BaseOutbox` and pass it to the `agent.tick()` method.             |
| **Redis / VectorDB memory**        | Implement a `RedisMemory` by sub-classing `BaseMemory` and inject it into the `StandardAgent`.                 |
| **Local shell tools**              | Create a `ShellToolInterface` that implements the `ToolInterface` contract and inject it into the `StandardAgent`. |

## 🔮 Roadmap 

- Async agent loop & concurrency-safe inboxes
- Additional pre-built reasoners (ReAct, ToT, Graph-of-Thought)
- More out of the box composable parts to enable custom agents or reasoners
- Web dashboard (live agent state + logs)
- Vector-store memory with RAG planning
- Slack / Discord integration
- Redis / VectorDB memory
- Ideas are welcome! [Open an issue](https://github.com/jentic/standard-agent/issues) or [submit a pull request](https://github.com/jentic/standard-agent/pulls).


