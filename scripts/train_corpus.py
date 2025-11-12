#!/usr/bin/env python3
"""
CLI helper to train the sentiment stack on a labeled corpus.
Example:
    python scripts/train_corpus.py \
        --dataset data/processed/labeled_tweets.csv \
        --text-column text \
        --label-column sentiment \
        --label-map negative:0,neutral:1,positive:2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from backend.models.sentiment_models import SentimentAnalyzer  # noqa: E402
from backend.utils.feature_extractor import FeatureExtractor  # noqa: E402
from backend.utils.text_preprocessor import TextPreprocessor  # noqa: E402


def parse_label_map(raw: str | None) -> Dict[str, int] | None:
    if not raw:
        return None

    # Allow comma-separated pairs (negative:0,neutral:1) or JSON object.
    if raw.strip().startswith("{"):
        return json.loads(raw)

    mapping: Dict[str, int] = {}
    for pair in raw.split(","):
        if ":" not in pair:
            raise ValueError(f"Invalid label mapping entry '{pair}'. Use name:index pairs.")
        name, value = pair.split(":", 1)
        mapping[name.strip()] = int(value.strip())
    return mapping


def load_corpus(
    dataset_path: Path,
    text_column: str,
    label_column: str,
    limit: int | None,
    label_map: Dict[str, int] | None,
) -> Tuple[List[str], List[int]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(dataset_path)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(dataset_path, lines=suffix == ".jsonl")
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Use CSV/JSON/Parquet.")

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"Columns '{text_column}'/'{label_column}' not found. Available: {list(df.columns)}"
        )

    df = df[[text_column, label_column]].dropna()

    if label_map:
        df[label_column] = df[label_column].map(lambda v: label_map.get(str(v).strip(), v))

    if not pd.api.types.is_integer_dtype(df[label_column]):
        raise ValueError(
            f"Labels must be integers 0/1/2. Provide --label-map if your labels are strings. "
            f"Found dtype {df[label_column].dtype}"
        )

    if limit:
        df = df.head(limit)

    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(int).tolist()
    return texts, labels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the sentiment models on a labeled corpus."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/labeled_tweets.csv",
        help="Path to CSV/JSON/Parquet corpus with text+label columns.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the column containing the raw text.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of the column containing numeric sentiment labels (0,1,2).",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="Optional mapping for non-numeric labels. Example: 'negative:0,neutral:1,positive:2' or JSON string.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows for quick experiments.",
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Skip the Hugging Face fine-tuning step.",
    )
    parser.add_argument(
        "--transformer-only",
        action="store_true",
        help="Train only the transformer (no classical/NN/LSTM models).",
    )
    parser.add_argument(
        "--transformer-base",
        type=str,
        default="distilbert-base-uncased",
        help="Checkpoint name or path to fine-tune.",
    )
    parser.add_argument(
        "--transformer-epochs",
        type=int,
        default=3,
        help="Epochs for transformer fine-tuning.",
    )
    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=16,
        help="Batch size for transformer fine-tuning.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="backend/models/saved",
        help="Directory to store trained artifacts.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    dataset_path = Path(args.dataset)
    label_map = parse_label_map(args.label_map)
    texts, labels = load_corpus(
        dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
        limit=args.limit,
        label_map=label_map,
    )

    print(f"Loaded {len(texts)} samples from {dataset_path}")

    text_preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    analyzer = SentimentAnalyzer()
    analyzer.load_dependencies(feature_extractor, text_preprocessor)

    if args.transformer_only:
        print("Training transformer model only...")
        analyzer.train_transformer(
            texts,
            labels,
            base_model=args.transformer_base,
            epochs=args.transformer_epochs,
            batch_size=args.transformer_batch_size,
        )
    else:
        analyzer.train_all_models(
            texts,
            labels,
            include_transformer=not args.skip_transformer,
        )

    analyzer.save_models(args.save_dir)
    print(f"Models saved under {args.save_dir}")

    if analyzer.model_selection_results:
        print("\nBest parameters discovered during model selection:")
        for name, info in analyzer.model_selection_results.items():
            print(f"- {name}: {info}")


if __name__ == "__main__":
    main()
