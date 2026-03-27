# Contributing to GT-ADF

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/your-username/GT-ADF.git
cd GT-ADF
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Code Style

We use `black` for formatting and `flake8` for linting:

```bash
black src/ scripts/ tests/
flake8 src/ scripts/ tests/ --max-line-length=100
```

## Reporting Issues

Please open a GitHub issue with:
- Python / PyTorch / PyG version
- Full traceback
- Minimal reproducible example

## Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Add tests for new functionality
4. Ensure all tests pass
5. Open a pull request with a clear description
