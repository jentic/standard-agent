# ActBots: A Framework for Self-Healing AI Agents

A **clean, modular Python framework** for building robust, autonomous AI agents. ActBots provides a powerful reasoning engine that can dynamically plan, execute, and **reflect on failures** to heal itself, ensuring reliable task completion using the Jentic tool platform.

## 🎯 Project Goals

- **Autonomous & Self-Healing**: Build agents that can recover from tool errors and unexpected issues without human intervention.
- **Modular Architecture**: Enforce a clean separation between the agent's reasoning, memory, and the tools it uses.
- **Extensible by Design**: Easily swap reasoning strategies or add new tool providers without rewriting the core logic.
- **Production Ready**: Emphasize comprehensive testing, strict type hints, and clear dependency management.

## 🏗️ Architecture

```
jentic_agents/
│
├─ agents/                             # High-level agent orchestration
│   ├─ base_agent.py                   # Abstract agent interface
│   └─ interactive_cli_agent.py        # Concrete CLI-based agent
│
├─ reasoners/                          # Core reasoning and self-healing logic
│   ├─ rewoo_reasoner_contract.py      # Abstract ReWOO Reasoner contract
│   └─ rewoo_reasoner/                 # ReWOO implementation
│       └─ core.py                     # The ReWOOReasoner with reflection logic
│
├─ tools/                              # The generic tool abstraction layer
│   ├─ interface.py                    # Defines the abstract ToolInterface contract
│   └─ models.py                       # Canonical Tool data model
│
├─ platform/                           # Concrete implementations for external services
│   ├─ jentic_client.py                # Low-level Jentic SDK wrapper
│   └─ jentic_tool_iface.py            # Jentic implementation of the ToolInterface
│
├─ memory/                             # Pluggable memory backends
│   ├─ base_memory.py                  # Abstract memory interface
│   └─ scratch_pad.py                  # Simple in-memory key-value store
│
├─ inbox/                              # Goal/task delivery systems
│   ├─ base_inbox.py                   # Abstract inbox interface
│   └─ cli_inbox.py                    # CLI input inbox

```

## 🧠 Core Components

### Agents
Agents are the top-level orchestrators that wire together all other components.
- **`BaseAgent`**: The abstract interface defining the agent's main `spin()` loop.
- **`InteractiveCLIAgent`**: A concrete agent for interactive use via the command line.

### Reasoners
The reasoning layer implements the core logic for planning, acting, and self-healing.
- **`BaseReWOOReasoner`**: The abstract contract for a ReWOO (Reason-without-Observation) reasoner, defined in `rewoo_reasoner_contract.py`.
- **`ReWOOReasoner`**: The concrete implementation that generates plans and uses a **reflection loop** to recover from tool failures.

### Tools
The tool layer provides a generic abstraction for any action an agent can take.
- **`ToolInterface`**: An abstract class defining the contract for any tool provider (e.g., `search`, `execute`).

### Platform
The platform layer contains concrete implementations of the `ToolInterface`.
- **`JenticToolInterface`**: The adapter that allows the reasoner to use the Jentic platform for its tools.
- **`JenticClient`**: A low-level wrapper around the Jentic SDK.

### Memory
Pluggable memory backends for storing information across reasoning steps.
- **`BaseMemory`**: A simple key-value storage interface.
- **`ScratchPadMemory`**: An in-memory dictionary-based implementation.

### Inbox
Goal delivery systems that feed tasks to the agent.
- **`BaseInbox`**: An abstract interface for receiving goals.
- **`CLIInbox`**: An implementation that gets goals from interactive command-line input.

## 🚀 Quick Start

### Installation

```bash
# Clone and set up the project
git clone <repository-url>
cd actbots

# Install dependencies using PDM
pdm install

# Run tests
pdm run test

# Check code quality
pdm run lint
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

### Basic Usage

The following example demonstrates how to set up and run the `JenticReWOOReasoner`.

```python
import os
from jentic_agents.utils.llm import LiteLLMChatLLM
from jentic_agents.memory.scratch_pad import ScratchPadMemory
from jentic_agents.platform.jentic_tool_iface import JenticToolInterface
from jentic_agents.reasoners.rewoo_reasoner.core import ReWOOReasoner
from jentic_agents.platform.jentic_client import JenticClient
from jentic_agents.agents.interactive_cli_agent import InteractiveCLIAgent
from jentic_agents.inbox.cli_inbox import CLIInbox
# 1. Set up the components
llm_wrapper = LiteLLMChatLLM(model='claude-sonnet-4-20250514')
memory = ScratchPadMemory()
jentic_client = JenticClient(api_key='Jentic API KEY')
jentic_tools = JenticToolInterface(client=jentic_client)

inbox = CLIInbox()

# 2. Instantiate the Reasoner
reasoner = ReWOOReasoner(
    llm=llm_wrapper,
    tool=jentic_tools,
    memory=memory,
)

agent = InteractiveCLIAgent(
    reasoner=reasoner,
    memory=memory,
    inbox=inbox,
    jentic_client=jentic_client
)
agent.spin()  # Start the interactive loop

```

## 🔮 Future Enhancements

- **Advanced Planning**: Enhance the plan parser to understand conditional logic (`if fails...`).
- **Smarter Reflection**: Improve the reflection prompts with more sophisticated failure analysis.
- **More Tool Interfaces**: Implement interfaces for other tool providers (e.g., local shell commands, other APIs).
- **Vector Memory**: Add a vector database for more complex memory retrieval.

---

**Built for a new generation of resilient, modular, and truly autonomous AI agents.**
