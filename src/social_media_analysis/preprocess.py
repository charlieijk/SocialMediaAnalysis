"""Lightweight text normalization utilities used across the package."""
from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
MULTISPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize noisy social text into a tokenizable string."""
    lowered = text.lower()
    no_urls = URL_RE.sub(" ", lowered)
    no_handles = MENTION_RE.sub(" ", no_urls)
    no_emoji = EMOJI_RE.sub(" ", no_handles)
    hashtags_flat = re.sub(r"#(\w+)", r"\1", no_emoji)
    alnum = re.sub(r"[^0-9a-z!'?\s]", " ", hashtags_flat)
    squashed = MULTISPACE_RE.sub(" ", alnum)
    return squashed.strip()


def normalize_series(texts: Iterable[str]) -> pd.Series:
    return pd.Series([normalize_text(t) for t in texts])


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def generate_ngrams(tokens: List[str], ngram_range: tuple[int, int] = (1, 2)) -> List[str]:
    tokens = [t for t in tokens if t]
    min_n, max_n = ngram_range
    ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.append("_".join(tokens[i : i + n]))
    return ngrams


__all__ = ["normalize_text", "normalize_series", "tokenize", "generate_ngrams"]
