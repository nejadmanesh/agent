# Agent Project Scaffold

This repository provides a pre-configured Python project scaffold featuring Poetry for
dependency management alongside essential development tooling such as pre-commit,
Black, isort, mypy, and MkDocs for documentation.

## Project Structure

```
.
├── docs/                # MkDocs documentation sources
├── src/app/             # Application package
├── src/services/        # Inference engine and HTTP API service
├── src/workers/         # Celery worker for background inference tasks
├── src/data_pipeline/   # Utilities for preparing labelled Persian datasets
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

## Machine Learning Inference Stack

- **Inference engine**: `services.inference.TextInferenceEngine` lazily loads a
  persisted model artifact (via Joblib) and exposes a predictable API for
  generating ranked label predictions.
- **HTTP API**: `services.api.create_app` builds a FastAPI application that
  surfaces health and prediction endpoints backed by the shared inference
  engine instance.
- **Background workers**: `workers.configure_celery` wires the inference engine
  into a Celery application so asynchronous jobs can be queued via Redis or any
  supported broker.

## Development Commands

- Format code with Black: `poetry run black .`
- Sort imports with isort: `poetry run isort .`
- Run static checks with mypy: `poetry run mypy .`
- Serve documentation locally: `poetry run mkdocs serve`
