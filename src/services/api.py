"""HTTP API surface for serving predictions from the inference engine."""

from __future__ import annotations

from typing import Iterable

from .inference import InferenceResult, TextInferenceEngine


def create_app(engine: TextInferenceEngine):
    """Create and configure a FastAPI application bound to *engine*.

    FastAPI is imported lazily so that the module remains importable even when
    the optional API dependencies are not installed.  This keeps the service
    layer usable in lightweight environments such as unit tests while still
    providing a fully fledged ASGI application for production deployments.
    """

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Agent Inference Service", version="0.1.0")

    class HealthResponse(BaseModel):
        status: str

    class PredictionRequest(BaseModel):
        text: str
        top_k: int | None = None
        threshold: float | None = None

    class PredictionResponse(BaseModel):
        label: str
        score: float

    def _serialise(results: Iterable[InferenceResult]) -> list[PredictionResponse]:
        return [PredictionResponse(label=result.label, score=result.score) for result in results]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:  # pragma: no cover - trivial proxy
        return HealthResponse(status="ok")

    @app.post("/predict", response_model=list[PredictionResponse])
    def predict(request: PredictionRequest) -> list[PredictionResponse]:
        results = engine.predict(
            request.text,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        if not results:
            raise HTTPException(status_code=400, detail="No predictions available for the provided text.")
        return _serialise(results)

    return app


__all__ = ["create_app"]
