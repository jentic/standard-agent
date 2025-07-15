# Contributing to Standard Agent

Thank you for your interest in contributing to Standard Agent! This document outlines the process for contributing to the Standard Agent framework.

## Our Mission

Standard Agent helps developers build reliable, modular agents that run Jentic workflows and operations end-to-end autonomously, but with human intervention only when neccessary. Designed for clarity, extensibility, and real-world integration.

> **Note:** We are actively developing tools and integrations to make the contribution process easier. Stay tuned for updates that will streamline the contribution workflow and help you effortlessly participate in building the Standard Agent framework.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [community@jentic.com](mailto:community@jentic.com).

## Getting Started

1.  **Fork the Repository**: Start by forking the repository to your own GitHub account.
2.  **Clone the Repository**: Clone your forked repository to your local machine.
3.  **If preparing a contribution** review [Contribution Process](#contribution-process).

## What We're Looking For

We welcome contributions in the following areas:

1.  **Core Framework Enhancements**: Improvements to reasoning strategies, memory backends, agent orchestration, performance/reliability or inbox/outbox implementations.
2.  **Self-Healing & Reflection**: Enhancements to failure detection, reflection strategies, and error recovery patterns.
3.  **Human in the Loop**: Features to improve our human-in-the-loop system, such as updating prompts or enhancing LLM reasoning in this area.
4.  **Documentation & Examples**: Better usage examples, tutorials, and architecture documentation.
5.  **Testing & Quality**: Increased test coverage, new integration tests, and performance benchmarks.

## Quality Standards

-   **Code Style**: We follow **[PEP 8](https://peps.python.org/pep-0008/)** for code standards and use **ruff** for formatting and linting.
-   **Docstrings**: All public modules, classes, and functions should have docstrings following **[PEP 257](https://peps.python.org/pep-0257/)**.
-   **Testing**: All new code requires comprehensive unit and integration tests using **pytest**.

## Contribution Process

1.  **Create a Branch**: Create a branch in your forked repository for your contribution.
2.  **Make Your Changes**: Implement your changes, following our style and quality guidelines.
3.  **Test Your Changes**: Ensure your contributions are thoroughly tested.
4.  **Submit a Pull Request**: Open a pull request from your branch to our main repository.

## Pull Request Process

1.  Update any relevant documentation with details of changes if appropriate.
2.  Include a clear description of the changes and their benefits in your PR.
3.  The PR will be reviewed by maintainers, who may request changes.
4.  Once approved, your PR will be merged by a maintainer.

## Development Workflow

-   **Branch Naming**: Use descriptive names: `feature/description`, `fix/issue-description`
-   **Commit Messages**: We follow **[Conventional Commits](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716)**. Write clear, concise commit messages describing your changes.
-   **Documentation**: Update relevant documentation when making changes.

## Directory Structure

When contributing please follow our established directory structure as documented in STRUCTURE.md. 

## Recognition

Contributors will be recognized in our documentation and through appropriate credit mechanisms. We believe in acknowledging all forms of contribution.

Thank you for helping build the open knowledge foundation for AI agents!

---

*This document may be updated as the project evolves.*