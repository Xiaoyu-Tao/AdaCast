"""Prediction-row coordinates shared by adjustment diagnostics and application."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_coordinates(
    predictions: pd.DataFrame,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated explicit coordinates, with a legacy sliding-window fallback."""
    has_window = "window_index" in predictions.columns
    has_horizon = "horizon_index" in predictions.columns
    if has_window != has_horizon:
        raise ValueError(
            "predictions.csv must provide both window_index and horizon_index or neither"
        )

    if has_window:
        window_values = pd.to_numeric(predictions["window_index"], errors="coerce")
        horizon_values = pd.to_numeric(predictions["horizon_index"], errors="coerce")
        for name, values in (
            ("window_index", window_values),
            ("horizon_index", horizon_values),
        ):
            numeric = values.to_numpy(dtype=np.float64)
            if (
                not np.isfinite(numeric).all()
                or (numeric < 0).any()
                or not np.equal(numeric, np.floor(numeric)).all()
            ):
                raise ValueError(
                    f"predictions.csv {name} must contain non-negative integers"
                )
        return (
            window_values.to_numpy(dtype=np.int64),
            horizon_values.to_numpy(dtype=np.int64),
        )

    safe_pred_len = max(int(pred_len), 1)
    row_index = np.arange(len(predictions), dtype=np.int64)
    return row_index // safe_pred_len, row_index % safe_pred_len
