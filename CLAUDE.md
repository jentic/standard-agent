# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Environment Setup:**
```bash
make install          # Create venv and install dependencies
source .venv/bin/activate  # Activate virtual environment
```

**Testing & Quality:**
```bash
make test             # Run unit tests with pytest
make lint             # Run ruff linting
make lint-strict      # Run ruff + mypy type checking
```

**Running the Agent:**
```bash
python main.py                    # Quick start with pre-built ReWOO agent
python main_build_your_own_agent.py  # Example of custom agent composition
```

## Architecture Overview

Standard Agent is a **modular AI agent framework** built on composition principles. The core design follows a layered architecture where each component can be swapped independently.

### Core Components

**StandardAgent** (`agents/standard_agent.py`)
- Top-level orchestrator that owns and injects LLM, Memory, and Tools into reasoner

**Reasoners** (`reasoner/`)
- `SequentialReasoner`: Default Plan → Execute → Reflect loop implementation
- Composed of 4 swappable parts: Planner, ExecuteStep, Reflector, AnswerBuilder
- Each component follows interface contracts defined in `reasoner/sequential/interface.py`

**Tools** (`tools/`)
- `JustInTimeToolingBase`: Abstract contract for external actions
- `JenticClient`: Default implementation using Jentic platform
- Tools are injected globally and accessible to all reasoner components

**Memory** (`memory/`)
- `BaseMemory`: Key-value store interface
- `ScratchPadMemory`: Simple in-memory implementation
- Shared across all agent components for state persistence

**LLM Integration** (`llm/`)
- `BaseLLM`: Uniform interface for language models
- `LiteLLM`: Implementation supporting OpenAI, Anthropic, Google models

### Configuration

**Environment Variables** (create `.env` from `.env.example`):
- `JENTIC_API_KEY`: Required for tool access
- `LLM_MODEL`: Model selection (default: claude-sonnet-4-20250514)
- Provider API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`

**Logging**: Configured via `config.json` with file rotation and console output

### Key Design Patterns

**Dependency Injection**: LLM, Memory, and Tools are injected once by StandardAgent

**Interface Segregation**: Each layer (Planner, ExecuteStep, etc.) implements focused contracts, enabling mix-and-match composition

**Error Recovery**: ReWOOReflector automatically analyzes failures and modifies plans for autonomous recovery

### Extension Points

- **Custom Reasoners**: Implement `BaseReasoner` for different reasoning strategies
- **New Tools**: Implement `JustInTimeToolingBase` for additional external integrations  
- **Memory Backends**: Implement `BaseMemory` for Redis, vector databases, etc.
- **I/O Channels**: Implement `BaseInbox`/`BaseOutbox` for Slack, Discord, web interfaces