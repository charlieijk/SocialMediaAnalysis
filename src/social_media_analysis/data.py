"""Data loading helpers for the packaged ML demos."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

PROCESSED_SUBDIR = Path("data") / "processed"
DEFAULT_LABELED_FILE = "sample_labeled_tweets.csv"
DEFAULT_UNLABELED_FILE = "sample_unlabeled_tweets.csv"


def _candidate_data_dirs() -> Iterable[Path]:
    env_dir = os.getenv("SMA_DATA_DIR")
    if env_dir:
        yield Path(env_dir).expanduser().resolve()

    # Walk upward from the module location to find the repo root and fallback to CWD.
    for parent in Path(__file__).resolve().parents:
        candidate = parent / PROCESSED_SUBDIR
        if candidate.exists():
            yield candidate
    cwd_candidate = Path.cwd() / PROCESSED_SUBDIR
    if cwd_candidate.exists():
        yield cwd_candidate


def default_processed_dir() -> Path:
    for candidate in _candidate_data_dirs():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate the processed data directory. "
        "Set the SMA_DATA_DIR environment variable or place CSVs under data/processed/."
    )


def load_csv(name: str | Path) -> pd.DataFrame:
    csv_path = Path(name).expanduser()
    if not csv_path.is_absolute() and not csv_path.exists():
        csv_path = default_processed_dir() / csv_path
    resolved = csv_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find dataset at {resolved}")
    return pd.read_csv(resolved)


def load_labeled_samples(
    csv_path: str | Path | None = None,
    *,
    text_column: str = "text",
    label_column: str = "sentiment",
) -> pd.DataFrame:
    """Return a clean DataFrame with the requested text/label columns."""
    df = load_csv(csv_path or DEFAULT_LABELED_FILE)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))} in {csv_path or DEFAULT_LABELED_FILE}"
        )
    cleaned = df[[text_column, label_column]].dropna().reset_index(drop=True)
    return cleaned.rename(columns={text_column: "text", label_column: "label"})


def load_unlabeled_samples(csv_path: str | Path | None = None, *, text_column: str = "text") -> pd.Series:
    df = load_csv(csv_path or DEFAULT_UNLABELED_FILE)
    if text_column not in df.columns:
        raise ValueError(
            f"Missing required column '{text_column}' in {csv_path or DEFAULT_UNLABELED_FILE}"
        )
    return df[text_column].dropna().reset_index(drop=True)


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    unique_labels: int


def summarize_dataframe(df: pd.DataFrame, label_column: str = "label") -> DatasetSummary:
    return DatasetSummary(rows=len(df), unique_labels=df[label_column].nunique())


__all__ = [
    "DatasetSummary",
    "DEFAULT_LABELED_FILE",
    "DEFAULT_UNLABELED_FILE",
    "default_processed_dir",
    "load_labeled_samples",
    "load_unlabeled_samples",
    "summarize_dataframe",
]
