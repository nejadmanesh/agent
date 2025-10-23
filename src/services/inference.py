"""Inference utilities shared between the HTTP API and worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class ModelConfig:
    """Configuration describing how to load and interpret a model artifact."""

    artifact_path: Path
    labels: Sequence[str]
    probability_threshold: float = 0.0
    default_top_k: int = 3


@dataclass(frozen=True)
class InferenceResult:
    """A single label prediction accompanied by its confidence score."""

    label: str
    score: float


class TextInferenceEngine:
    """Load a persisted model once and serve predictions on demand."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        model_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self._config = config
        self._model: Any | None = None
        self._lock = Lock()
        self._model_loader = model_loader or _default_joblib_loader

    @property
    def config(self) -> ModelConfig:
        """Return the configuration that governs the inference engine."""

        return self._config

    def predict(
        self,
        text: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[InferenceResult]:
        """Return the top predictions for *text*.

        The model is loaded lazily and shared between calls to minimise the cost
        of repeated predictions.  If the underlying model exposes a
        ``predict_proba`` method it will be used; otherwise the engine falls back
        to ``predict`` and generates a best-effort confidence distribution.
        """

        normalised = text.strip()
        if not normalised:
            return []

        model = self._load_model()
        scores = _resolve_probabilities(model, normalised, self._config.labels)

        active_top_k = top_k if top_k is not None else self._config.default_top_k
        active_threshold = (
            threshold if threshold is not None else self._config.probability_threshold
        )

        candidates = sorted(
            (
                InferenceResult(label=label, score=float(score))
                for label, score in zip(self._config.labels, scores)
            ),
            key=lambda item: item.score,
            reverse=True,
        )

        return [
            candidate
            for candidate in candidates[:active_top_k]
            if candidate.score >= active_threshold
        ]

    def _load_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._model_loader(self._config.artifact_path)
        return self._model


def _resolve_probabilities(model: Any, text: str, labels: Sequence[str]) -> Sequence[float]:
    """Return a probability distribution for *text* provided a model instance."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])
        first_row = probabilities[0]
        return _as_floats(first_row)

    predictions = model.predict([text])
    first_prediction = predictions[0]
    distribution = [0.0] * len(labels)

    if isinstance(first_prediction, str):
        try:
            index = labels.index(first_prediction)
        except ValueError:
            return distribution
        distribution[index] = 1.0
        return distribution

    if isinstance(first_prediction, int):
        if 0 <= first_prediction < len(labels):
            distribution[first_prediction] = 1.0
        return distribution

    if isinstance(first_prediction, Iterable):
        for label in first_prediction:
            try:
                index = labels.index(label)  # type: ignore[arg-type]
            except ValueError:
                continue
            distribution[index] = 1.0
        return distribution

    raise TypeError(
        "Model predict method returned an unsupported type: "
        f"{type(first_prediction)!r}"
    )


def _as_floats(probabilities: Sequence[Any]) -> list[float]:
    return [float(probability) for probability in probabilities]


def _default_joblib_loader(path: Path) -> Any:
    from joblib import load

    return load(path)
