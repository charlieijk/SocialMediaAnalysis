"""Train and evaluate simple sentiment models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .preprocess import normalize_text


def _normalize_batch(texts):
    return [normalize_text(t) for t in texts]


@dataclass
class ModelConfig:
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    test_size: float = 0.25
    random_state: int = 42
    max_iter: int = 200


def _build_pipeline(config: ModelConfig) -> Pipeline:
    normalizer = FunctionTransformer(_normalize_batch, validate=False)
    vectorizer = TfidfVectorizer(max_features=config.max_features, ngram_range=config.ngram_range)
    classifier = LogisticRegression(max_iter=config.max_iter, solver="lbfgs", random_state=config.random_state)
    return Pipeline([("normalize", normalizer), ("tfidf", vectorizer), ("clf", classifier)])


class SentimentExperiment:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.pipeline: Pipeline = _build_pipeline(self.config)
        self.metrics_: Dict[str, Any] | None = None

    def train(
        self,
        df: pd.DataFrame,
        *,
        text_column: str = "text",
        label_column: str = "label",
    ) -> Dict[str, Any]:
        texts = df[text_column].astype(str)
        labels = df[label_column].astype(int)
        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels,
        )
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_val)
        accuracy = float(accuracy_score(y_val, predictions))
        report = classification_report(y_val, predictions, output_dict=True, zero_division=0)
        self.metrics_ = {"accuracy": accuracy, "report": report}
        return self.metrics_

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict(list(texts))

    def evaluate(self, texts: Sequence[str], labels: Sequence[int]) -> Dict[str, Any]:
        preds = self.pipeline.predict(list(texts))
        accuracy = float(accuracy_score(labels, preds))
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        return {"accuracy": accuracy, "report": report}

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(self.config), "pipeline": self.pipeline}
        joblib.dump(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SentimentExperiment":
        payload = joblib.load(Path(path))
        config = ModelConfig(**payload["config"])
        experiment = cls(config=config)
        experiment.pipeline = payload["pipeline"]
        return experiment


__all__ = ["ModelConfig", "SentimentExperiment"]
