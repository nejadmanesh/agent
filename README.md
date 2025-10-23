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

## Containerization and Monitoring

- Build the application image:

  ```bash
  docker build -f infra/Dockerfile -t agent-app:latest .
  ```

- Start the full stack (application + monitoring) with Docker Compose:

  ```bash
  cd infra
  docker compose up --build
  ```

  This launches the Python worker, Prometheus, Alertmanager, Grafana, and cAdvisor on the
  default ports (`8000`, `9090`, `9093`, `3000`, and `8080`). Metrics are exported from the
  application at `http://localhost:8000/metrics`.

- Access Grafana at [http://localhost:3000](http://localhost:3000) (default credentials `admin` / `admin`).

- Alerts are defined under `infra/monitoring/prometheus/alert_rules.yml` and delivered to
  Alertmanager. Extend `infra/monitoring/alertmanager/alertmanager.yml` to route alerts to
  external systems (Slack, PagerDuty, etc.).

## Continuous Integration

GitHub Actions workflows in `.github/workflows/ci.yml` lint, test, build, and publish the Docker
image to GitHub Container Registry whenever commits land on `main`.
