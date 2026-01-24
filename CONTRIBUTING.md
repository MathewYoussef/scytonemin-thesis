# Contributing to Scytonemin Thesis

Thank you for your interest in contributing to this research repository! This document provides guidelines and workflows for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Questions and Support](#questions-and-support)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to `mathewyoussef@gmail.com`.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git for version control
- A GitHub account

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/scytonemin-thesis.git
   cd scytonemin-thesis
   ```

3. **Set up the development environment**:
   ```bash
   make setup        # Create virtual environment and install dependencies
   ```

4. **Verify your setup**:
   ```bash
   pytest tests -m "quick"
   make docs
   ```

## Development Workflow

### Creating a Branch

Create a feature branch for your work:

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-analysis` for new features
- `fix/correct-calibration` for bug fixes
- `docs/update-readme` for documentation changes
- `test/add-unit-tests` for test additions

### Making Changes

1. **Make your changes** in small, logical commits
2. **Write clear commit messages**:
   ```
   Short (50 chars or less) summary

   More detailed explanatory text, if necessary. Wrap it to about 72
   characters. The blank line separating the summary from the body is
   critical.

   - Bullet points are okay
   - Use imperative mood ("Add feature" not "Added feature")
   ```

3. **Keep your fork updated**:
   ```bash
   git remote add upstream https://github.com/MathewYoussef/scytonemin-thesis.git
   git fetch upstream
   git rebase upstream/main
   ```

## Testing

### Running Tests

Before submitting changes, ensure all tests pass:

```bash
# Run quick tests
pytest tests -m "quick"

# Run all tests
pytest tests

# Run specific test file
pytest tests/test_sanity.py

# Run with coverage
pytest tests --cov=src --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory, mirroring the `src/` structure
- Name test files with a `test_` prefix (e.g., `test_chromatography.py`)
- Use `pytest` fixtures and markers appropriately
- Mark quick tests with `@pytest.mark.quick`
- Ensure tests are reproducible and independent

Example test structure:

```python
import pytest
from src.module import function_to_test

@pytest.mark.quick
def test_function_behavior():
    """Test that function handles expected input correctly."""
    result = function_to_test(input_data)
    assert result == expected_output
```

## Submitting Changes

### Pull Request Process

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Provide a clear title describing the change
   - Fill out the PR template with relevant details
   - Reference any related issues (e.g., "Fixes #123")
   - Ensure CI checks pass

3. **Respond to feedback**:
   - Address review comments promptly
   - Push updates to the same branch
   - Request re-review when ready

4. **After approval**:
   - A maintainer will merge your PR
   - Delete your feature branch

### Pull Request Guidelines

- **Keep PRs focused**: One feature or fix per PR
- **Update documentation**: If you change behavior, update relevant docs
- **Add tests**: New features should include test coverage
- **Follow style guidelines**: Ensure code passes linting
- **Keep commits clean**: Squash or rebase if needed

## Style Guidelines

### Python Code Style

This project follows standard Python conventions:

- **PEP 8** for code style
- **Type hints** where appropriate (Python 3.11+)
- **Docstrings** for all public functions, classes, and modules

Example docstring format:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid
    """
    pass
```

### Code Organization

- Keep functions focused and single-purpose
- Use meaningful variable and function names
- Add comments for complex logic
- Group related functionality into modules
- Maintain the existing project structure

### Documentation

- Update README.md if adding new features
- Add docstrings to new modules and functions
- Keep documentation concise and accurate
- Use Markdown for documentation files

## Questions and Support

- **General questions**: Open a [Discussion](https://github.com/MathewYoussef/scytonemin-thesis/discussions)
- **Bug reports**: Use the bug report template in Issues
- **Feature requests**: Use the feature request template in Issues
- **Security issues**: Email `mathewyoussef@gmail.com` directly

## Recognition

Contributors will be acknowledged in the project. Significant contributions may be recognized in publications derived from this work, subject to standard academic practices.

---

Thank you for contributing to the Scytonemin Thesis project! Your efforts help advance open and reproducible scientific research.
