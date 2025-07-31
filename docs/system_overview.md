# Standard Agent System Overview

## 1. Introduction

The `standard-agent` is a sophisticated and well-architected Python framework for building autonomous agents. It is designed with modularity, extensibility, and robustness in mind, enabling developers to create powerful agents that can reason, solve complex problems, and interact with external tools.

This document provides a high-level overview of the system's architecture, its core components, and the key design patterns that make it effective.

## 2. Core Architecture

The agent's architecture is centered around the `StandardAgent` class, which acts as a lightweight façade to orchestrate the core components of the system. It follows a dependency injection pattern, where the main agent class is initialized with four key services:

-   **LLM (Language Model):** The underlying language model that provides reasoning capabilities.
-   **Memory:** A storage backend for persisting conversation history, intermediate results, and other state.
-   **Tools:** A system for interacting with external APIs and functionalities.
-   **Reasoner:** The "brain" of the agent, responsible for executing the main thinking loop to achieve a user-defined goal.

This decoupled architecture allows each component to be developed, tested, and replaced independently, making the entire system highly flexible and maintainable.

## 3. Key Components

### 3.1. StandardAgent

-   **File:** `agents/standard_agent.py`
-   **Role:** The central coordinator that wires all other components together.
-   **Functionality:**
    -   Initializes and holds instances of the LLM, memory, tools, and reasoner.
    -   Provides the main entry point for solving a goal via the `solve()` method.
    -   Manages the agent's lifecycle state (`READY`, `BUSY`, `NEEDS_ATTENTION`).

### 3.2. LLM

-   **Files:** `agents/llm/base_llm.py`, `agents/llm/litellm.py`
-   **Role:** To provide natural language understanding and generation capabilities.
-   **Key Features:**
    -   **`BaseLLM` Interface:** Defines a common contract for all language models, ensuring that different LLMs can be swapped out with minimal code changes.
    -   **`LiteLLM` Implementation:** A concrete implementation that uses the `litellm` library to connect to a wide range of LLM providers (e.g., GPT-4o, Claude).
    -   **Fault-Tolerant JSON Parsing:** The `prompt_to_json` method includes robust retry logic to ensure that the LLM's output is always valid JSON, which is critical for structured data exchange.

### 3.3. Memory

-   **File:** `agents/memory/dict_memory.py`
-   **Role:** To store and retrieve information during the agent's execution.
-   **Key Features:**
    -   **`MutableMapping` Interface:** The system is designed to work with any storage backend that implements Python's `MutableMapping` interface.
    -   **`DictMemory`:** A simple, in-memory dictionary implementation is provided for development and testing. For production use cases, this can be easily replaced with a persistent storage solution like Redis.

### 3.4. Tools

-   **Files:** `agents/tools/base.py`, `agents/tools/jentic.py`
-   **Role:** To provide the agent with the ability to interact with the outside world.
-   **Key Features:**
    -   **Just-in-Time (JIT) Tooling:** The `JustInTimeToolingBase` interface defines a powerful and efficient design pattern. Instead of pre-loading all available tools, the agent can dynamically search for and load tools on-demand using natural language queries.
    -   **`JenticClient`:** A concrete implementation that integrates with the `jentic-sdk` to provide access to a library of external tools (both "operations" and "workflows"). It manages API authentication, execution, and error handling.

### 3.5. Reasoner

-   **Files:** `agents/reasoner/base.py`, `agents/reasoner/sequential/reasoner.py`, `agents/reasoner/prebuilt.py`
-   **Role:** The core cognitive engine that drives the agent's problem-solving process.
-   **Key Features:**
    -   **`BaseReasoner` Interface:** Defines the contract for all reasoning engines.
    -   **`SequentialReasoner`:** A sophisticated implementation that executes a multi-step plan to achieve a goal. It is composed of four distinct phases:
        1.  **Plan:** Generates a sequence of steps to solve the problem.
        2.  **Execute:** Runs each step, which may involve using a tool.
        3.  **Reflect:** (Optional) If an error occurs, this phase analyzes the failure and can modify the plan to recover.
        4.  **Summarize:** Generates the final answer once the plan is complete.
    -   **`ReWOOReasoner`:** A pre-configured `SequentialReasoner` that implements the "Reflect, Work, Observe" (ReWOO) methodology, a well-established pattern for building capable agents.

## 4. Design Patterns and Strengths

-   **Modularity:** The clear separation of concerns makes the system easy to understand and extend.
-   **Extensibility:** The use of abstract base classes throughout the codebase provides clear extension points for customizing functionality.
-   **Just-in-Time Tooling:** This design is highly scalable and efficient, allowing the agent to adapt to new tools without requiring code changes.
-   **Robustness:** The reflection mechanism in the reasoner and the fault-tolerant JSON handling in the LLM make the agent resilient to errors.

## 5. Areas for Contribution

The `standard-agent` project is a solid foundation. Here are some areas where the community could contribute:

-   **Configuration:** A detailed guide on how to configure the agent, particularly for setting up API keys and other secrets.
-   **Custom Reasoners:** Tutorials on how to create new reasoners by composing different planners, executors, and reflectors.
-   **Tool Integrations:** Examples of how to connect to other tool backends by implementing the `JustInTimeToolingBase`.
-   **Memory Persistence:** Guides on how to integrate persistent memory solutions like Redis or a database.
-   **Getting Started Guide:** A comprehensive tutorial on how to run the agent and use it to solve a sample problem.
