"""Celery task definitions for executing inference asynchronously."""

from __future__ import annotations

from typing import Any

from services.inference import TextInferenceEngine


def configure_celery(
    engine: TextInferenceEngine,
    *,
    broker_url: str,
    result_backend: str | None = None,
):
    """Return a configured Celery application instance.

    The Celery dependency is imported lazily to avoid coupling the rest of the
    codebase to the optional worker stack when it is not required (for example
    during unit testing or local experimentation).
    """

    from celery import Celery

    app = Celery("agent_inference", broker=broker_url, backend=result_backend)

    @app.task(name="workers.perform_inference")
    def perform_inference(text: str, top_k: int | None = None, threshold: float | None = None) -> list[dict[str, Any]]:
        results = engine.predict(text, top_k=top_k, threshold=threshold)
        return [
            {
                "label": result.label,
                "score": result.score,
            }
            for result in results
        ]

    return app


__all__ = ["configure_celery"]
