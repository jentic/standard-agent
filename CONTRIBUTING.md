# Contributing to Standard Agent

Thank you for considering contributing to the Standard Agent! This document outlines the process for contributing to the Standard Agent Repository.

## Guiding Principles

The Standard-Agent library is guided by a simple mission: to demonstrate that building a powerful, modular AI agent can be straightforward and intuitive. We provide a minimal set of reference implementations that can be easily understood, extended, and composed.

Our core principles are:

-   **Composition:** We believe that new capabilities should emerge from the interplay of simple, independent components. Our goal is to provide a library of swappable parts that can be combined in novel ways.

-   **Clarity and Simplicity:** The entire codebase should be something any reasonably experienced coder can read and understand quickly. We prioritize clear, readable code over complex, "magic" abstractions.

-   **Extensibility by Default:** Every core component is built around a clear interface (`BaseReasoner`, `ToolBase`, `MutableMapping`), making it easy to add new reasoning strategies, tool integrations, and memory backends without needing to change the core library.

These principles guide our development and we welcome contributions that share this vision.


## How to Contribute

There are many ways to contribute to the Standard Agent project. We welcome contributions in the following areas:

-   **Reasoning Strategies (Profiles):** Clear, reference implementations of different reasoning strategies that implement the `BaseReasoner` interface. Current profiles include `ReWOOReasoner` (Plan → Execute → Reflect) and `ReACTReasoner` (Think → Act). We welcome well-documented, easy-to-understand profiles (e.g., ReAct variants, LATS, ToT, GoT).
-   **Examples:** We need good examples that show how `standard-agent` can be used to solve high-level goals. These examples should be clear and concise.
-   **Tool Integrations:** While the library comes with a Jentic implementation, you can integrate any tool backend by implementing the `ToolBase` interface. We welcome contributions of new tool integrations.
-   **Memory Implementations:** The library currently uses simple in-memory dictionaries for storage. We welcome contributions of persistent memory backends (Redis, SQLite, file-based storage) or specialized memory implementations (vector stores, semantic search, conversation summarizers) that implement the `MutableMapping` interface.
-   **Bug Fixes & Documentation:** We always appreciate well-documented bug reports and improvements to our documentation.

Reasoner prompts are externalized in YAML under `agents/prompts/` (e.g., `agents/prompts/reasoners/{rewoo,react}.yaml`, and `agents/prompts/agent.yaml` for the agent-level summarizer). Each profile declares the exact required keys and fails fast if a prompt is missing. When contributing new profiles, please:

- Keep the implementation as a single file implementing `BaseReasoner` if possible
- Add a YAML file under `agents/prompts/reasoners/your_profile.yaml` documenting required keys
- Use the strict loader (`agents/prompts/load_prompts`) to load your prompts

We also encourage contributions of entirely new `BaseReasoner` implementations to explore approaches like Tree-of-Thoughts, Graph-of-Thoughts, ReAct variants, or LATS.

If you have an idea to improve the Standard Agent, we'd love to see it!

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.


## Submitting Issues

We use GitHub Issues for all bug reports and feature requests. Before opening a new issue, please search the existing issues to see if your problem or idea has already been discussed.

When you're ready, please use our issue templates to create a report:

-   **[🐛 Bug Report](https://github.com/jentic/standard-agent/issues/new?assignees=&labels=bug&template=bug_report.md&title=%5BBug%5D+)**
-   **[✨ Feature Request](https://github.com/jentic/standard-agent/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFeature%5D+)**

This helps us stay organized and ensures we have all the information we need to address your submission efficiently. We look forward to your feedback!

## Pull Request Process

To ensure a smooth and transparent process, we ask that all pull requests be linked to a GitHub issue.

1.  **Find or Create an Issue:** Before starting, check if an issue for your proposed change already exists. If not, please create a new one.
    *   For **new features or significant changes**, please detail what you aim to accomplish and your planned technical approach.
    *   For **bug fixes**, describe the bug and how you plan to fix it.

2.  **Fork & Branch:** Create a fork of the repository and make your changes in a descriptively named branch.

3.  **Code & Test:** Write your code and add tests to cover your changes. Make sure the existing test suite passes.

4.  **Update Documentation:** If you've added a new feature or changed an existing one, be sure to update the relevant documentation.

5.  **Submit a Pull Request:** When you're ready, submit a pull request.
    *   **Crucially, link the issue you created or are addressing in your PR description.**
    *   Provide a clear summary of the changes you've made.

6.  **Review and Merge:** The PR will be reviewed by maintainers, who may request changes if necessary. Once approved, your PR will be merged.

To make the review process as efficient as possible, please try to keep your pull requests **small and focused**. We also typically **squash commits** when merging a PR to maintain a clean and readable git history.

## Development Workflow

- **Branch Naming**: Use descriptive names: `feature/description` or `fix/issue-description`.
- **Commit Messages**: Write clear, concise commit messages describing your changes.
- **Testing**: Before submitting your pull request, please ensure all tests pass by running `make test`.
- **Linting**: Please ensure your code adheres to the project's style guidelines by running `make lint` and `make lint-strict`.
- **Documentation**: Update relevant documentation when making changes.

## Recognition

Contributors will be recognized in our documentation and through appropriate credit mechanisms. We believe in acknowledging all forms of contribution.
Thank you for helping build and improve the Standard Agent!

---

*This document may be updated as the project evolves.*