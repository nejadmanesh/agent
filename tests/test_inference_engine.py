from __future__ import annotations

from pathlib import Path

import pytest

from services.inference import InferenceResult, ModelConfig, TextInferenceEngine


class _ProbabilisticModel:
    def predict_proba(self, inputs: list[str]) -> list[list[float]]:
        assert len(inputs) == 1
        return [[0.1, 0.6, 0.3]]


class _DeterministicModel:
    def __init__(self, label: str) -> None:
        self._label = label

    def predict(self, inputs: list[str]) -> list[str]:
        assert len(inputs) == 1
        return [self._label]


def _loader_factory(model: object):
    def _loader(_: Path) -> object:
        return model

    return _loader


def test_engine_uses_predict_proba_when_available(tmp_path: Path) -> None:
    config = ModelConfig(artifact_path=tmp_path / "model.joblib", labels=["neg", "neu", "pos"], default_top_k=2)
    engine = TextInferenceEngine(config, model_loader=_loader_factory(_ProbabilisticModel()))

    results = engine.predict("نمونه")

    assert results == [InferenceResult(label="neu", score=0.6), InferenceResult(label="pos", score=0.3)]


def test_engine_handles_predict_fallback(tmp_path: Path) -> None:
    config = ModelConfig(artifact_path=tmp_path / "model.joblib", labels=["neg", "pos"], probability_threshold=0.0)
    engine = TextInferenceEngine(config, model_loader=_loader_factory(_DeterministicModel("pos")))

    results = engine.predict("نمونه")

    assert results == [InferenceResult(label="pos", score=1.0)]


def test_predict_returns_empty_list_for_blank_input(tmp_path: Path) -> None:
    config = ModelConfig(artifact_path=tmp_path / "model.joblib", labels=["neg", "pos"], probability_threshold=0.5)
    engine = TextInferenceEngine(config, model_loader=_loader_factory(_DeterministicModel("pos")))

    assert engine.predict("    ") == []


def test_predict_applies_threshold(tmp_path: Path) -> None:
    config = ModelConfig(artifact_path=tmp_path / "model.joblib", labels=["neg", "pos"], probability_threshold=0.7)
    engine = TextInferenceEngine(config, model_loader=_loader_factory(_ProbabilisticModel()))

    results = engine.predict("نمونه", threshold=0.8)

    assert results == []


@pytest.mark.parametrize(
    "prediction, expected",
    [
        ([], []),
        (["missing"], []),
        (
            [["pos", "neg"]],
            [
                InferenceResult(label="neg", score=1.0),
                InferenceResult(label="pos", score=1.0),
            ],
        ),
    ],
)
def test_predict_handles_iterable_predictions(
    tmp_path: Path, prediction: list[list[str]], expected: list[InferenceResult]
) -> None:
    class _IterableModel:
        def predict(self, inputs: list[str]) -> list[list[str]]:
            assert len(inputs) == 1
            return prediction

    config = ModelConfig(artifact_path=tmp_path / "model.joblib", labels=["neg", "pos"], probability_threshold=0.0)
    engine = TextInferenceEngine(config, model_loader=_loader_factory(_IterableModel()))

    results = engine.predict("نمونه", top_k=2, threshold=0.0)

    assert results == expected
