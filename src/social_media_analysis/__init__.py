"""Utilities for training and evaluating lightweight social media sentiment models."""
from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - convenience metadata lookup only
    __version__ = version("social-media-analysis")
except PackageNotFoundError:  # pragma: no cover - during editable installs
    __version__ = "0.0.0"

from . import data, model, preprocess  # re-export modules for convenience

__all__ = ["data", "model", "preprocess", "__version__"]
