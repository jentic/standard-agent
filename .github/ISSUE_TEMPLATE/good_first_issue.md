---
name: "🌟 Good First Issue"
about: "Create a beginner-friendly issue for new contributors"
title: "[Good First Issue] "
labels: ["good first issue", "hacktoberfest"]
assignees: ''
---

## 🎯 Description

<!-- Brief description of what needs to be done -->

## 🔍 Problem/Motivation

<!-- Why this issue exists and why it needs to be solved -->

## ✅ Acceptance Criteria

- [ ] <!-- Specific requirement 1 -->
- [ ] <!-- Specific requirement 2 -->
- [ ] <!-- Specific requirement 3 -->

## 📋 Step-by-Step Instructions

### Setup
1. Fork this repository by clicking the "Fork" button at the top right
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/standard-agent.git
   cd standard-agent
   ```
3. Set up the development environment:
   ```bash
   make install
   source .venv/bin/activate
   ```
4. Verify everything works:
   ```bash
   make test
   ```

### Making Changes
5. Create a new branch for your work:
   ```bash
   git checkout -b fix/issue-NUMBER-short-description
   ```
6. <!-- Add specific instructions for the task here -->
7. Test your changes:
   ```bash
   make test
   make lint
   ```

### Submitting
8. Commit your changes with a descriptive message:
   ```bash
   git add .
   git commit -m "type: brief description of changes"
   ```
9. Push to your fork:
   ```bash
   git push origin your-branch-name
   ```
10. Open a Pull Request:
    - Go to your fork on GitHub
    - Click "Compare & pull request"
    - Fill out the PR template
    - Link this issue in the PR description with "Fixes #ISSUE_NUMBER"

## 📚 Resources

- [Contributing Guide](./CONTRIBUTING.md)
- [Good First Issues Guide](./good_first_issue.md)
- [Development Setup](./README.md#quick-start)
- [Discord Community](https://discord.gg/TdbWXZsUSm)

## 🆘 Need Help?

- Comment on this issue with questions
- Join our [Discord](https://discord.gg/TdbWXZsUSm) for real-time help
- Check out similar completed PRs for examples

## 📊 Estimated Effort

**Time:** <!-- 10-30 minutes / 30-60 minutes / 1-2 hours -->
**Difficulty:** Beginner
**Skills needed:** <!-- Python basics / Git basics / Documentation / etc. -->

---

**Note:** Please comment on this issue before starting work to let others know you're working on it. This helps avoid duplicate efforts.

Welcome to the Standard Agent community! 🎉