# Contributing to Standard Agent

First off, thank you for considering contributing to the Standard Agent! This document outlines the process for contributing to the Standard Agent Repository.

## Project Philosophy

`Standard-Agent` is a **modular AI agent framework** built on composition principles. It is intended as a minimal set of reference implementations of different types of agent reasoning and tool-use strategies which can then be used to compose agents. The goal is to show that building an agent can be as simple as writing a few lines of code.

The core design follows a **layered architecture** where each component can be swapped independently. This enables developers to mix and match different reasoning strategies, memory backends, and tool integrations without rewriting the entire system.

The core library should be something any reasonably experienced coder can read and understand end-to-end real quick.


## How to Contribute

There are many ways to contribute to the Standard Agent project. We welcome contributions in the following areas:

-   **Reasoning Strategies:** The primary goal of this library is to serve as a collection of reference implementations for different agent reasoning strategies implementing the `BaseReasoner` interface. We welcome contributions of well-documented, easy-to-understand implementations of patterns like ReAct, LATS, Plan-Act, etc.
-   **Examples:** We need good examples that show how `standard-agent` can be used to solve high-level goals. These examples should be clear, concise, and demonstrate the trade-offs between different reasoning strategies.
-   **Tool Integrations:** While the library comes with a Jentic implementation, you can integrate any tool backend by implementing the `ToolBase` interface. We welcome contributions of new tool integrations.
-   **Memory Implementations:** The library currently uses simple in-memory dictionaries for storage. We welcome contributions of persistent memory backends (Redis, SQLite, file-based storage) or specialized memory implementations (vector stores, semantic search, conversation summarizers) that implement the `MutableMapping` interface.
-   **Bug Fixes & Documentation:** We always appreciate well-documented bug reports and improvements to our documentation.

For example, the `SequentialReasoner` is composed of a `Plan`, `ExecuteStep`, `Reflect`, and `SummarizeResult` component. You can create new reasoning behaviors by implementing new "flavors" of these components—such as a ReAct-style executor or a critique-based reflector—and then mixing and matching them to create novel agents. 
We also encourage the contribution of entirely new `BaseReasoner` implementations to explore different approaches, such as a Tree of Thoughts, Graph of Thoughts, ReAct, or LATS reasoner.

If you have an idea to improve the Standard Agent, we'd love to see it!

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.


## Pull Request Process

To ensure a smooth and transparent process, we ask that all pull requests be linked to a GitHub issue.

1.  **Find or Create an Issue:** Before starting, check if an issue for your proposed change already exists. If not, please create a new one.
    *   For **new features or significant changes**, please detail what you aim to accomplish and your planned technical approach.
    *   For **bug fixes**, describe the bug and how you plan to fix it.

2.  **Fork & Branch:** Create a fork of the repository and make your changes in a descriptively named branch.

3.  **Code & Test:** Write your code and add tests to cover your changes. Make sure the existing test suite passes by running `make test`.

4.  **Update Documentation:** If you've added a new feature or changed an existing one, be sure to update the relevant documentation.

5.  **Submit a Pull Request:** When you're ready, submit a pull request.
    *   **Crucially, link the issue you created or are addressing in your PR description.**
    *   Provide a clear summary of the changes you've made.

6.  **Review and Merge:** The PR will be reviewed by maintainers, who may request changes if necessary. Once approved, your PR will be merged.

To make the review process as efficient as possible, please try to keep your pull requests **small and focused**. We also typically **squash commits** when merging a PR to maintain a clean and readable git history.

## Development Workflow

- **Branch Naming**: Use descriptive names: `feature/description`, `fix/issue-description`
- **Commit Messages**: Write clear, concise commit messages describing your changes
- **Documentation**: Update relevant documentation when making changes

## Community

- **Discussions**: [Insert forum/discussion board]
- **Questions**: Open an issue with a "question" label for any questions
- **Meetings**: [Insert information about community meetings if applicable]


## Recognition

Contributors will be recognized in our documentation and through appropriate credit mechanisms. We believe in acknowledging all forms of contribution.
Thank you for helping build and improve the Standard Agent!

---

*This document may be updated as the project evolves.*