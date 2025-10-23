"""Celery worker glue for asynchronous inference jobs."""

from .tasks import configure_celery

__all__ = ["configure_celery"]
