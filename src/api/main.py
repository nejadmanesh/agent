"""Definition of the FastAPI application."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, status

from inference import MemoryStatus

from .dependencies import get_database, get_engine, get_memory_manager
from .schemas import (
    InferenceRequest,
    InferenceResponse,
    InferenceResultResponse,
    SynchronousInferenceResponse,
)


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def create_app() -> FastAPI:
    app = FastAPI(title="Inference Service", version="0.1.0")

    @app.post(
        "/inference/tasks",
        response_model=InferenceResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def enqueue_inference(
        request: InferenceRequest,
        engine=Depends(get_engine),
        database=Depends(get_database),
    ) -> InferenceResponse:
        from worker.tasks import run_inference_task

        async_result = run_inference_task.delay(request.input_text, request.model_name)
        model_name = request.model_name or engine.default_model_name
        database.upsert_result(
            async_result.id,
            model_name=model_name,
            input_text=request.input_text,
            status="pending",
        )
        return InferenceResponse(task_id=async_result.id, status="pending")

    @app.post(
        "/inference/sync",
        response_model=SynchronousInferenceResponse,
    )
    def synchronous_inference(
        request: InferenceRequest,
        engine=Depends(get_engine),
    ) -> SynchronousInferenceResponse:
        result = engine.predict(request.input_text, model_name=request.model_name)
        return SynchronousInferenceResponse(
            model_name=result.model_name,
            input_text=result.input_text,
            output_text=result.output_text,
            metadata=dict(result.metadata),
        )

    @app.get(
        "/inference/tasks/{task_id}",
        response_model=InferenceResultResponse,
    )
    def get_inference_result(task_id: str, database=Depends(get_database)) -> InferenceResultResponse:
        record = database.get_result(task_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
        return InferenceResultResponse(
            task_id=record.task_id,
            model_name=record.model_name,
            input_text=record.input_text,
            output_text=record.output_text,
            status=record.status,
            error=record.error,
            created_at=_parse_timestamp(record.created_at),
        )

    @app.get("/memory")
    def memory_status(memory_manager=Depends(get_memory_manager)) -> Dict[str, Any]:
        status_obj: MemoryStatus = memory_manager.status()
        return {
            "limit_bytes": status_obj.limit_bytes,
            "used_bytes": status_obj.used_bytes,
            "available_bytes": status_obj.available_bytes,
            "usage_ratio": status_obj.usage_ratio,
        }

    @app.get("/models")
    def list_models(engine=Depends(get_engine)) -> Dict[str, Any]:
        return {
            "default_model": engine.default_model_name,
            "available_models": list(engine.list_models().keys()),
        }

    @app.get("/health")
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
