"""Tests for adjustment scaling under the fixed validation-front contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adacast_ml.adjustment.apply import effective_max_delta_ratio, target_scale


def test_target_scale_uses_full_training_split(tmp_path: Path) -> None:
    dataset_path = tmp_path / "series.csv"
    train_values = np.arange(70, dtype=np.float64)
    validation_front = np.full(5, 10000.0)
    remaining = np.arange(25, dtype=np.float64) + 100.0
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=100, freq="h"),
            "target": np.concatenate([train_values, validation_front, remaining]),
        }
    ).to_csv(dataset_path, index=False)

    task = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
    }

    mean, scale = target_scale(task, tmp_path / ".forecast" / "task.json")

    assert mean == pytest.approx(float(np.mean(train_values)))
    assert scale == pytest.approx(float(np.std(train_values)))


def test_target_scale_uses_timestamp_training_boundary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "timestamp-series.csv"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=20, freq="h"),
            "target": np.concatenate(
                [
                    np.arange(8, dtype=np.float64),
                    np.full(6, 10000.0),
                    np.full(6, 20000.0),
                ]
            ),
        }
    )
    frame.to_csv(dataset_path, index=False)
    task = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "split_mode": "timestamp",
        "train_end_timestamp": str(frame["date"].iloc[7]),
        "validation_end_timestamp": str(frame["date"].iloc[13]),
        "train_ratio": 0.4,
        "val_ratio": 0.3,
        "test_ratio": 0.3,
    }

    mean, scale = target_scale(task, tmp_path / ".forecast" / "task.json")

    assert mean == pytest.approx(float(np.mean(np.arange(8, dtype=np.float64))))
    assert scale == pytest.approx(float(np.std(np.arange(8, dtype=np.float64))))


def test_rule_bound_cannot_exceed_global_policy_bound() -> None:
    assert effective_max_delta_ratio(
        {"bounds": {"max_delta_ratio": 0.25}},
        0.1,
    ) == pytest.approx(0.1)
    assert effective_max_delta_ratio(
        {"bounds": {"max_delta_ratio": 0.03}},
        0.1,
    ) == pytest.approx(0.03)
