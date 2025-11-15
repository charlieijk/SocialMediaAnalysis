"""Command line helpers for training/evaluating the sentiment demo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from . import data
from .model import ModelConfig, SentimentExperiment


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _handle_train(args: argparse.Namespace) -> Dict[str, Any]:
    df = data.load_labeled_samples(
        args.data,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    config = ModelConfig(
        max_features=args.max_features,
        ngram_range=(args.min_ngram, args.max_ngram),
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    experiment = SentimentExperiment(config=config)
    metrics = experiment.train(df)
    model_path = experiment.save(args.model_path)
    return {"model_path": str(model_path), **metrics}


def _handle_predict(args: argparse.Namespace) -> Dict[str, Any]:
    experiment = SentimentExperiment.load(args.model_path)
    texts: List[str]
    if args.text:
        texts = args.text
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    predictions = experiment.predict(texts)
    return {"predictions": predictions.tolist()}


def _handle_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    experiment = SentimentExperiment.load(args.model_path)
    df = data.load_labeled_samples(
        args.data,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    metrics = experiment.evaluate(df["text"], df["label"])
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="social-media-analysis", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a logistic regression model")
    train.add_argument("--data", default=None, help="Path to a CSV file with text/sentiment columns")
    train.add_argument("--text-column", default="text", help="Column containing text")
    train.add_argument("--label-column", default="sentiment", help="Column containing the label")
    train.add_argument("--model-path", default="backend/models/saved/cli_sentiment_model.joblib")
    train.add_argument("--max-features", type=int, default=5000)
    train.add_argument("--min-ngram", type=int, default=1)
    train.add_argument("--max-ngram", type=int, default=2)
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--max-iter", type=int, default=200)
    train.set_defaults(func=_handle_train)

    predict = subparsers.add_parser("predict", help="Run inference with a saved model")
    predict.add_argument("--model-path", required=True)
    predict.add_argument("--text", action="append", help="Text to classify; can be repeated")
    predict.set_defaults(func=_handle_predict)

    evaluate = subparsers.add_parser("evaluate", help="Score a model on a labeled CSV")
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--data", default=None)
    evaluate.add_argument("--text-column", default="text")
    evaluate.add_argument("--label-column", default="sentiment")
    evaluate.set_defaults(func=_handle_evaluate)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = args.func(args)
    print(json.dumps(_to_serializable(result), indent=2))


__all__ = ["build_parser", "main"]
