"""Celery tasks responsible for running inference workloads."""

from __future__ import annotations

import os
from typing import Any, Dict
from uuid import uuid4

from celery import Celery

from api.dependencies import get_database, get_engine

celery_app = Celery(
    "inference_worker",
    broker=os.environ.get("CELERY_BROKER_URL", "memory://"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "rpc://"),
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_ignore_result=False,
)


@celery_app.task(name="worker.run_inference_task", bind=True)
def run_inference_task(self, input_text: str, model_name: str | None = None) -> Dict[str, Any]:
    engine = get_engine()
    database = get_database()
    task_id = self.request.id or str(uuid4())
    target_model = model_name or engine.default_model_name
    database.upsert_result(
        task_id,
        model_name=target_model,
        input_text=input_text,
        status="running",
    )
    try:
        result = engine.predict(input_text, model_name=model_name)
    except Exception as exc:
        database.upsert_result(
            task_id,
            model_name=target_model,
            input_text=input_text,
            status="failed",
            error=str(exc),
        )
        raise
    database.upsert_result(
        task_id,
        model_name=result.model_name,
        input_text=result.input_text,
        status="succeeded",
        output_text=result.output_text,
    )
    return {
        "task_id": task_id,
        "model_name": result.model_name,
        "output_text": result.output_text,
        "status": "succeeded",
    }
