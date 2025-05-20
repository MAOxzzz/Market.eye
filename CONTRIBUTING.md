# Contributing to Market Eye AI

Thank you for considering contributing to Market Eye AI! This document outlines the guidelines for contributing to this project. By following these guidelines, we can maintain code quality and streamline the contribution process.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Style Guidelines](#style-guidelines)
5. [Commit Guidelines](#commit-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Project Structure](#project-structure)

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/Market-Eye-AI-Powered-Stock-Analysis-System-.git
   cd Market-Eye-AI-Powered-Stock-Analysis-System-
   ```
3. **Set up the development environment**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a branch for your feature or fix**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

There are many ways to contribute to Market Eye AI:

1. **Report bugs** - Create an issue describing the bug and how to reproduce it
2. **Suggest enhancements** - Create an issue describing your proposed enhancement
3. **Implement new features** - Check open issues or suggest new features
4. **Improve documentation** - Help clarify or expand upon existing documentation
5. **Write tests** - Increase test coverage for existing features

## Style Guidelines

This project follows PEP 8 style guidelines with some modifications:

- Use 4 spaces for indentation
- Maximum line length is 100 characters
- Use docstrings for all functions, classes, and modules
- Use type hints where appropriate
- Include comprehensive comments for complex code sections

We recommend using tools like flake8 and black to ensure compliance:

```bash
# Install tools
pip install flake8 black

# Check code style
flake8 .

# Auto-format code
black .
```

## Commit Guidelines

- **Use descriptive commit messages** that clearly explain what was changed
- **Keep commits atomic** - One change per commit
- **Follow the conventional commits format**:

  ```
  <type>(<scope>): <description>

  [optional body]

  [optional footer]
  ```

  Types include:

  - feat: A new feature
  - fix: A bug fix
  - docs: Documentation changes
  - style: Code style changes (formatting, etc.)
  - refactor: Code changes that neither fix bugs nor add features
  - test: Adding or modifying tests
  - chore: Changes to build process or auxiliary tools

  Example: `feat(forecasting): implement LSTM model for stock prediction`

## Pull Request Process

1. **Update your fork** to include the latest changes from the main repository
2. **Run tests** to ensure your changes don't break existing functionality
3. **Update documentation** if necessary
4. **Create a pull request** with a clear description of the changes
5. **Address review comments** if any are provided

Pull requests require review from at least one maintainer before being merged.

## Project Structure

The project follows this structure:

```
market-eye-ai/
├── backend/                 # Backend components
│   ├── agents/              # AI agents and intelligent components
│   ├── models/              # ML models and prediction algorithms
│   └── database/            # Database schemas and management
├── frontend/                # Frontend components
│   └── pages/               # Individual UI pages and components
├── docs/                    # Documentation
│   └── report_templates/    # Templates for report generation
├── data/                    # Data storage and management
├── tests/                   # Tests for all components
├── utils/                   # Utility functions and helpers
├── scripts/                 # Utility scripts
└── requirements.txt         # Project dependencies
```

Please ensure that your contributions maintain this structure.

---

Thank you for your contributions to Market Eye AI! Your efforts help make this project better for everyone.
