# Contributing to Standard Agent

First off, thank you for considering contributing to the Standard Agent! This document outlines the process for contributing to the Standard Agent Repository.

## Project Philosophy

`standard-agent` is a **modular AI agent framework** built on composition principles. It is intended as a minimal set of reference implementations of different types of agent reasoning and tool-use strategies. The goal is to show that building an agent can be as simple as writing a few lines of code.

The core design follows a **layered architecture** where each component can be swapped independently. This enables developers to mix and match different reasoning strategies, memory backends, and tool integrations without rewriting the entire system.

The core library should be something any reasonably experienced coder can read and understand end-to-end real quick.


## How to Contribute

There are many ways to contribute to the Standard Agent project. We welcome contributions in the following areas:

-   **Reasoning Strategies:** The primary goal of this library is to serve as a collection of reference implementations for different agent reasoning strategies implementing the `BaseReasoner` interface. We welcome contributions of well-documented, easy-to-understand implementations of patterns like ReAct, LATS, Plan-Act, etc.
-   **Examples:** We need good examples that show how `standard-agent` can be used to solve high-level goals. These examples should be clear, concise, and demonstrate the trade-offs between different reasoning strategies.
-   **Tool Integrations:** While the library comes with a Jentic implementation, you can integrate any tool backend by implementing the `ToolBase` interface. We welcome contributions of new tool integrations.
-   **Memory Implementations:** The library currently uses simple in-memory dictionaries for storage. We welcome contributions of persistent memory backends (Redis, SQLite, file-based storage) or specialized memory implementations (vector stores, semantic search, conversation summarizers) that implement the `MutableMapping` interface.
-   **Bug Fixes & Documentation:** We always appreciate well-documented bug reports and improvements to our documentation.

For example, the `SequentialReasoner` is composed of a `Plan`, `ExecuteStep`, `Reflect`, and `SummarizeResult` component. You can create new reasoning behaviors by implementing new "flavors" of these components—such as a ReAct-style executor or a critique-based reflector—and then mixing and matching them to create novel agent architectures. 
We also encourage the contribution of entirely new `BaseReasoner` implementations to explore different approaches, such as a Tree of Thoughts, Graph Of Thoughts, ReAct, or LATS reasoner.

If you have an idea for a new reasoner, a new component or a new way to combine existing ones, we'd love to see it!

## Code of Conduct
``
This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.


## Pull Request Process

1.  **Fork and Branch:** Create a fork of the repository and create a new branch for your feature or bug fix.
2.  **Code and Test:** Write your code and add tests to cover your changes. Make sure the existing test suite passes.
3.  **Update Documentation:** If you've added a new feature or changed an existing one, be sure to update the relevant documentation.
4.  **Submit a Pull Request:** Open a pull request to the `main` branch of the original repository. Provide a clear description of your changes and reference any relevant issues.


## Styleguides

### Git Commit Messages

-   Use the present tense ("Add feature" not "Added feature").
-   Use the imperative mood ("Move file to..." not "Moves file to...").
-   Limit the first line to 72 characters or less.
-   Reference issues and pull requests liberally after the first line.

### Python Styleguide

All Python code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) and should be formatted with [Black](https://github.com/psf/black).
