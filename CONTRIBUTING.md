# Contributing to Miles

Thank you for your interest in contributing to Miles! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Areas of Interest](#areas-of-interest)

## Getting Started

Miles is an enterprise-facing reinforcement learning framework for large-scale MoE post-training and production workloads. Before contributing, we recommend:

1. Reading the [README.md](README.md) for project overview
2. Reviewing the [Quick Start Guide](./docs/en/get_started/quick_start.md)
3. Understanding the [Architecture Overview](#architecture-overview)

### Architecture Overview

Miles follows a decoupled, modular design with three main subsystems:

```
Training (Megatron/FSDP) <---> Data Buffer <---> Rollout (SGLang + Router)
```

- **Training**: Main training loop using Megatron-LM or FSDP backends
- **Data Buffer**: Manages prompts, data sources, and rollout strategies
- **Rollout**: Generates samples and rewards via SGLang

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (H100/H200/B200 recommended)
- Git

### Setting Up Your Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/radixark/miles.git
   cd miles
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   apt install pre-commit -y  # Or: pip install pre-commit
   pre-commit install
   ```

## Code Style

We use automated tools to maintain consistent code style. All code must pass pre-commit checks before merging.

### Pre-commit Hooks

Our pre-commit configuration includes:

- **Black**: Code formatter (line length: 119)
- **isort**: Import sorter (black-compatible)
- **Ruff**: Linter (E, F, B, UP rules)
- **autoflake**: Removes unused imports

### Running Pre-commit

```bash
# Run on all files
pre-commit run --all-files --show-diff-on-failure --color=always

# Run on staged files only
pre-commit run
```

### Style Guidelines

- **Line length**: 119 characters (enforced by Black)
- **Imports**: Sorted by isort in sections (stdlib, third-party, first-party)
- **Type hints**: Encouraged for public APIs and complex functions
- **Docstrings**: Use Google-style docstrings for public functions

Example docstring:
```python
def compute_advantage(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Compute advantage estimates using GAE.

    Args:
        rewards: Tensor of shape (batch_size, seq_len) containing rewards.
        values: Tensor of shape (batch_size, seq_len) containing value estimates.

    Returns:
        Tensor of shape (batch_size, seq_len) containing advantage estimates.

    Raises:
        ValueError: If rewards and values have mismatched shapes.
    """
```

## Testing

We use pytest for testing. Tests are organized into categories using markers.

### Test Categories

- `unit`: Fast, isolated tests for individual functions
- `integration`: Tests for subsystem interactions
- `system`: End-to-end training tests
- `acceptance`: User acceptance criteria tests

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run tests with coverage
pytest --cov=miles --cov-report=html

# Run a specific test file
pytest tests/test_ppo_utils.py

# Run tests verbosely
pytest -v
```

### Writing Tests

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use appropriate markers:

```python
import pytest

@pytest.mark.unit
def test_compute_kl_k1():
    """Test k1 KL divergence estimator."""
    # Test implementation
    pass

@pytest.mark.integration
def test_training_loop():
    """Test full training loop integration."""
    pass
```

### Test Guidelines

- Each test should be independent and not rely on other tests
- Use fixtures for common setup
- Test edge cases (empty inputs, single elements, large batches)
- Mock external dependencies (SGLang, Megatron) for unit tests

## Pull Request Process

### Before Submitting

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following code style guidelines

3. **Run pre-commit**:
   ```bash
   pre-commit run --all-files
   ```

4. **Run tests**:
   ```bash
   pytest -m unit  # At minimum, run unit tests
   ```

5. **Write/update tests** for your changes

### Submitting Your PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Link to any related issues
   - Test results or evidence of testing

3. **PR Title Format**:
   - `feat: Add new feature X`
   - `fix: Fix bug in Y`
   - `docs: Update documentation for Z`
   - `refactor: Refactor module W`
   - `test: Add tests for V`

### Review Process

- All PRs require at least one review
- Address review comments promptly
- Keep PRs focused and reasonably sized
- Large changes should be discussed in an issue first

## Areas of Interest

We're especially interested in contributions for:

### High Priority

- **Unit tests**: Core algorithms (ppo_utils, loss computation) lack unit tests
- **Type hints**: Adding type annotations to improve code clarity
- **Documentation**: API reference, troubleshooting guides

### Features

- **New hardware backends**: Support for new GPU architectures
- **MoE RL recipes**: Training configurations for MoE models
- **Stability improvements**: Determinism and reproducibility
- **Multimodal training**: Vision-language model support
- **Speculative training**: Advanced speculative decoding

### Code Quality

- **Refactoring**: Breaking down large files (arguments.py, loss.py)
- **Error handling**: Better error messages and recovery
- **Performance**: Memory and compute optimizations

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check `docs/` for detailed guides

## License

By contributing to Miles, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to Miles! Your efforts help make enterprise RL training more accessible and reliable.
