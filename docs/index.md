# Agent Project

Welcome to the Agent project documentation. This site is generated with MkDocs and
provides guidance on how to work with the project scaffold.

## Getting Started

1. Install Poetry if it is not already installed.
2. Install the project dependencies:

   ```bash
   poetry install
   ```

3. Set up the pre-commit hooks:

   ```bash
   poetry run pre-commit install
   ```

4. Copy `.env.example` to `.env` and adjust the configuration values as needed.

## Development Tools

- **Black** automatically formats the source code.
- **isort** sorts the imports.
- **mypy** performs static type checking.
- **pre-commit** executes the configured checks before each commit.
- **MkDocs** builds the project documentation. Serve the docs locally with:

  ```bash
  poetry run mkdocs serve
  ```
