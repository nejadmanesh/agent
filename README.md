# Agent Project Scaffold

This repository provides a pre-configured Python project scaffold featuring Poetry for
dependency management alongside essential development tooling such as pre-commit,
Black, isort, mypy, and MkDocs for documentation.

## Project Structure

```
.
├── docs/                # MkDocs documentation sources
├── src/app/             # Application package
├── tests/               # Automated tests
├── .pre-commit-config.yaml
├── mkdocs.yml           # Documentation configuration
└── pyproject.toml       # Poetry project definition and tool configs
```

## Getting Started

1. Install [Poetry](https://python-poetry.org/docs/#installation).
2. Install project dependencies:

   ```bash
   poetry install
   ```

3. Install the git hooks:

   ```bash
   poetry run pre-commit install
   ```

4. Copy `.env.example` to `.env` and provide real values.

## Development Commands

- Format code with Black: `poetry run black .`
- Sort imports with isort: `poetry run isort .`
- Run static checks with mypy: `poetry run mypy .`
- Serve documentation locally: `poetry run mkdocs serve`
