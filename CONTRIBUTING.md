# Contributing to Standard Agent

First off, thank you for considering contributing to Standard Agent! It's people like you that make this project such a great tool.

This document provides some guidelines for contributing to the project. Please feel free to propose changes to this document in a pull request.

## How to Contribute

There are many ways to contribute to the Standard Agent project. We welcome contributions in the following areas:

-   **Core Components:**
    -   **Reasoners:** Implement a new `BaseReasoner` to explore different problem-solving strategies.
    -   **LLM Integrations:** Add support for new language models by creating a new class that inherits from `BaseLLM`.
    -   **Tool Backends:** Connect to new tool providers by implementing the `JustInTimeToolingBase` interface.
    -   **Memory Stores:** Integrate a persistent memory backend by creating a class that implements the `MutableMapping` interface.
-   **Reporting Bugs:** If you find a bug, please open an issue on our GitHub repository. Be sure to include a clear description of the bug, steps to reproduce it, and any relevant logs or error messages.
-   **Suggesting Enhancements:** If you have an idea for a new feature or an improvement to an existing one, please open an issue to discuss it.
-   **Improving Documentation:** If you see an area where the documentation could be improved, please feel free to submit a pull request. This includes the `docs/system_overview.md` file.

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Setting up the Development Environment

To get started with development, you'll need to set up a local development environment. Here's how to do it:

1.  **Fork the repository:** Start by forking the repository on GitHub.
2.  **Clone your fork:** Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/YOUR_USERNAME/standard_agent.git
    ```
3.  **Create a virtual environment:** It's a good practice to use a virtual environment to manage your dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4.  **Install dependencies:** Install the required dependencies using pip.
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the tests:** To make sure everything is set up correctly, run the test suite.
    ```bash
    pytest
    ```

## Pull Request Process

1.  **Fork and Branch:** Create a fork of the repository and create a new branch for your feature or bug fix.
2.  **Code and Test:** Write your code and add tests to cover your changes. Make sure the existing test suite passes.
3.  **Update Documentation:** If you've added a new feature or changed an existing one, be sure to update the relevant documentation, including the `docs/system_overview.md` file.
4.  **Submit a Pull Request:** Open a pull request to the `main` branch of the original repository. Provide a clear description of your changes and reference any relevant issues.

## Configuration

The `config.json` file is used to configure the agent's services. To get started, you will need to add your `JENTIC_API_KEY` to this file.

```json
{
  "JENTIC_API_KEY": "your-api-key-here"
}
```

**Important:** Do not commit your `config.json` file to the repository. It is included in the `.gitignore` file to prevent accidental exposure of your API keys.


## Styleguides

### Git Commit Messages

-   Use the present tense ("Add feature" not "Added feature").
-   Use the imperative mood ("Move file to..." not "Moves file to...").
-   Limit the first line to 72 characters or less.
-   Reference issues and pull requests liberally after the first line.

### Python Styleguide

All Python code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) and should be formatted with [Black](https://github.com/psf/black).
