# Good First Issues Guide

Welcome to Standard Agent! We're excited to have you contribute during Hacktoberfest and beyond.

## What are Good First Issues?

Good first issues are beginner-friendly tasks that help you:
- Get familiar with our codebase
- Make a meaningful contribution
- Learn our development workflow
- Join our community

## How to Get Started

### 1. Set Up Your Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/standard-agent.git
cd standard-agent

# Set up the development environment
make install
source .venv/bin/activate

# Run tests to ensure everything works
make test
```

### 2. Find an Issue

Look for issues labeled with:
- `good first issue` - Perfect for newcomers
- `hacktoberfest` - Hacktoberfest-eligible
- `documentation` - Documentation improvements
- `enhancement` - Feature additions
- `bug` - Bug fixes

### 3. Work on Your Issue

1. **Comment on the issue** to let others know you're working on it
2. **Create a branch** for your work:
   ```bash
   git checkout -b fix/issue-number-short-description
   ```
3. **Make your changes** following our coding standards
4. **Test your changes**:
   ```bash
   make test
   make lint
   ```
5. **Commit and push**:
   ```bash
   git add .
   git commit -m "fix: descriptive commit message"
   git push origin your-branch-name
   ```

### 4. Submit Your Pull Request

- Use our PR template
- Link to the issue you're solving
- Provide clear description of changes
- Include tests if applicable

## Types of Good First Issues

### Documentation
- Fix typos or grammar
- Improve code examples
- Add missing docstrings
- Enhance README sections

### Code Quality
- Add type hints
- Improve test coverage
- Fix linting issues
- Add error handling

### Examples
- Create new example scripts
- Improve existing examples
- Add platform integrations
- Document use cases

### Tests
- Add unit tests
- Improve test coverage
- Add integration tests
- Fix flaky tests

## Development Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Write descriptive commit messages
- Add tests for new functionality

### Testing
```bash
# Run all tests
make test

# Run specific test file
pytest tests/path/to/test_file.py

# Run with coverage
pytest --cov=agents tests/
```

### Linting
```bash
# Check code style
make lint

# Auto-fix issues
ruff check . --fix
```

## Getting Help

- **Discord**: Join our [Discord community](https://discord.gg/TdbWXZsUSm)
- **Issues**: Ask questions in the issue comments
- **Discussions**: Use GitHub Discussions for broader questions

## Recognition

All contributors are recognized in our documentation. Your contributions help make AI agent development more accessible to everyone!

## Common Questions

**Q: I'm new to Python/AI/open source. Can I still contribute?**
A: Absolutely! Many good first issues require no AI expertise. Documentation, examples, and code quality improvements are valuable contributions.

**Q: How long should I wait before starting work on an issue?**
A: Comment on the issue first. If no one responds within 2-3 days, feel free to start working.

**Q: What if I get stuck?**
A: Ask for help! Comment on the issue or join our Discord. The community is here to support you.

**Q: Can I work on multiple issues at once?**
A: Start with one issue to get familiar with the workflow, then feel free to take on more.

Ready to contribute? Check out our [open good first issues](https://github.com/jentic/standard-agent/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and dive in!