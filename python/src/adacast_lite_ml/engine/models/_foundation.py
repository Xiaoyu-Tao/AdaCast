"""Foundation model metadata for zero-shot loader execution."""

from __future__ import annotations


FOUNDATION_MODEL_REPOS = {
    "Chronos": "amazon/chronos-bolt-base",
    "Chronos2": "amazon/chronos-2",
    "TimesFM": "google/timesfm-2.5-200m-pytorch",
    "Moirai": "Salesforce/moirai-2.0-R-small",
    "Sundial": "thuml/sundial-base-128m",
    "TiRex": "NX-AI/TiRex",
    "TimeMoE": "Maple728/TimeMoE-50M",
}

FOUNDATION_MODEL_NAMES = frozenset(FOUNDATION_MODEL_REPOS)

__all__ = ["FOUNDATION_MODEL_NAMES", "FOUNDATION_MODEL_REPOS"]
