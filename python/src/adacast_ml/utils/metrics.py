"""Fixed evaluation metrics for forecast scoring."""

from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((true - pred) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(true - pred)))


def wape(pred: np.ndarray, true: np.ndarray) -> float:
    """Weighted absolute percentage error."""
    denominator = np.sum(np.abs(true))
    if denominator == 0:
        return float("inf")
    return float(np.sum(np.abs(true - pred)) / denominator)


def mase(pred: np.ndarray, true: np.ndarray, train: np.ndarray) -> float:
    """Mean absolute scaled error using the train split as the naive baseline."""
    scale = np.mean(np.abs(np.diff(train)))
    if scale == 0:
        return float("inf")
    return float(np.mean(np.abs(true - pred)) / scale)
