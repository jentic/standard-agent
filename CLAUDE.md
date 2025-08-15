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

**Running Examples:**
```bash
python examples/cli_api_agent.py  # Example CLI using a prebuilt agent
```

## Architecture Overview

Standard Agent is a **modular AI agent framework** built on composition principles. The core design follows a layered architecture where each component can be swapped independently.

### Core Components

**StandardAgent** (`agents/standard_agent.py`)
- Top-level orchestrator that owns and injects LLM, Memory, and Tools into reasoner

**Reasoners** (`agents/reasoner/`)
- `rewoo.py`: Explicit Plan → Execute → Reflect loop in a single file
- `react.py`: Implicit Think → Act → Stop loop in a single file
- Both implement `BaseReasoner` and return a `ReasoningResult` with `transcript` and successful `tool_calls`

**Tools** (`agents/tools/`)
- `JustInTimeToolingBase`: Abstract contract for external actions
- `JenticClient`: Default implementation using Jentic platform
- Tools are injected globally and accessible to all reasoner components

**Memory** (`agents/memory/`)
- `DictMemory`: Simple in-memory key–value store shared by components

**LLM Integration** (`agents/llm/`)
- `BaseLLM`: Uniform interface for language models
- `LiteLLM`: Provider-agnostic wrapper with JSON-mode helpers

### Configuration

**Environment Variables** (create `.env`):
- LLM provider keys as needed: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`
- Tool backends require their own credentials; missing creds are surfaced in transcripts

**Logging**: Configured via `config.json` with file rotation and console output

### Key Design Patterns

**Dependency Injection**: LLM, Memory, and Tools are injected once by StandardAgent

**Interface Segregation**: Each layer (Planner, ExecuteStep, etc.) implements focused contracts, enabling mix-and-match composition

**Error Recovery**: ReWOO reflection suggests `change_tool`, `retry_params`, `rephrase_step`, or `give_up`. ReACT avoids reselecting failed tools and logs unauthorized errors.

### Prompts

Prompts are externalized to YAML under `agents/prompts/` and loaded via `load_prompts` with strict validation:
- `agents/prompts/agent.yaml` (summarization)
- `agents/prompts/reasoners/rewoo.yaml`
- `agents/prompts/reasoners/react.yaml`

### Extension Points

- Custom Reasoners: implement `BaseReasoner` in a single file (profile) and return a consistent `ReasoningResult`
- Tools: implement `JustInTimeToolingBase` for new providers
- Memory backends: swap `DictMemory` with any mapping-compatible store