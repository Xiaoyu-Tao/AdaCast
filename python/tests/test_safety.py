"""Tests for Phase 1 safety invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adacast_ml.data_loader import (
    ADJUSTMENT_FIT_HOLDOUT_RATIO,
    compute_split_indices,
    compute_training_boundaries,
    load_dataset,
    reject_if_test_in_forecast,
    resolve_split_indices,
    resolve_training_boundaries,
)
from adacast_ml.evaluator import evaluate
from adacast_ml.utils.hash import verify_evaluator


class TestSAFE01TestSetInaccessible:
    def test_test_split_rejected_in_forecast(self) -> None:
        spec = {"phase": "forecast", "eval_split": "test"}
        with pytest.raises(SystemExit, match="SAFETY VIOLATION"):
            reject_if_test_in_forecast(spec)

    def test_val_split_allowed_in_forecast(self) -> None:
        reject_if_test_in_forecast({"phase": "forecast", "eval_split": "val"})

    def test_test_split_allowed_post_forecast(self) -> None:
        reject_if_test_in_forecast({"phase": "post-forecast", "eval_split": "test"})

    def test_default_phase_is_restrictive(self) -> None:
        with pytest.raises(SystemExit, match="SAFETY VIOLATION"):
            reject_if_test_in_forecast({"eval_split": "test"})

    def test_load_dataset_respects_allowed_splits(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="h"),
                "OT": np.linspace(0, 99, 100),
            }
        ).to_csv(csv_path, index=False)

        result = load_dataset(
            {
                "dataset_path": str(csv_path),
                "target_col": "OT",
                "time_col": "date",
                "train_ratio": 0.7,
                "val_ratio": 0.1,
            },
            allowed_splits=["train", "val"],
        )

        assert set(result) == {"train", "val"}
        assert len(result["train"]["data"]) == 70
        assert len(result["val"]["data"]) == 5
        assert result["val"]["absolute_start_index"] == 75
        assert result["val"]["target_start_index"] == 0
        assert np.asarray(result["val"]["raw"])[0, 0] == 75.0

    def test_load_dataset_ignores_text_and_incomplete_calendar_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="h"),
                "OT": np.linspace(0, 99, 100),
                "exog": np.linspace(100, 199, 100),
                "holiday_text": ["workday"] * 50 + ["holiday"] * 50,
                "partial_calendar_numeric": [np.nan] * 50 + [1.0] * 50,
            }
        ).to_csv(csv_path, index=False)

        result = load_dataset(
            {
                "dataset_path": str(csv_path),
                "target_col": "OT",
                "time_col": "date",
                "train_ratio": 0.7,
                "val_ratio": 0.1,
            },
            allowed_splits=["train", "val"],
        )

        assert result["train"]["feature_columns"] == ["OT", "exog"]
        assert result["train"]["target_index"] == 0
        assert np.asarray(result["train"]["raw"]).shape[1] == 2


class TestSAFE02FrozenSplits:
    def test_splits_computed_from_spec_ratios(self) -> None:
        assert compute_split_indices(1000, 0.7, 0.1) == (700, 800)
        assert compute_split_indices(1000, 0.7, 0.1, 0.2) == (700, 800)
        assert compute_training_boundaries(1000, 0.7, 0.1) == (700, 750, 800)
        assert ADJUSTMENT_FIT_HOLDOUT_RATIO == pytest.approx(1 / 2)

    def test_split_ratios_must_be_positive_and_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            compute_split_indices(1000, 0.7, 0.3, 0.0)
        with pytest.raises(ValueError, match="must equal 1"):
            compute_split_indices(1000, 0.7, 0.1, 0.1)

    def test_timestamp_boundaries_are_inclusive_origin_boundaries_with_cross_split_context(
        self,
        tmp_path: Path,
    ) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=12, freq="h"),
                "OT": np.arange(12, dtype=np.float64),
            }
        )
        spec = {
            "time_col": "date",
            "split_mode": "timestamp",
            "train_end_timestamp": str(frame["date"].iloc[4]),
            "validation_end_timestamp": str(frame["date"].iloc[8]),
            "train_ratio": 5 / 12,
            "val_ratio": 4 / 12,
            "test_ratio": 3 / 12,
        }

        assert resolve_split_indices(frame, spec) == (5, 9)
        assert resolve_training_boundaries(frame, spec) == (5, 7, 9)

        csv_path = tmp_path / "timestamp-split.csv"
        frame.to_csv(csv_path, index=False)
        splits = load_dataset(
            {
                **spec,
                "dataset_path": str(csv_path),
                "target_col": "OT",
                "seq_len": 2,
                "pred_len": 2,
            },
            allowed_splits=["train", "adjustment_fit", "val", "test"],
        )
        assert np.asarray(splits["train"]["raw"])[-1, 0] == 4.0
        assert splits["adjustment_fit"]["absolute_start_index"] == 3
        assert splits["adjustment_fit"]["forecast_origin_index"] == 2
        assert splits["adjustment_fit"]["target_start_index"] == 2
        assert np.asarray(splits["adjustment_fit"]["raw"])[2, 0] == 5.0
        assert splits["val"]["absolute_start_index"] == 5
        assert np.asarray(splits["val"]["raw"])[2, 0] == 7.0
        assert splits["test"]["absolute_start_index"] == 7
        assert splits["test"]["target_start_index"] == 2
        assert np.asarray(splits["test"]["raw"])[2, 0] == 9.0

    def test_timestamp_split_rejects_non_chronological_or_tampered_contract(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01 00:00:00",
                        "2026-01-01 02:00:00",
                        "2026-01-01 01:00:00",
                        "2026-01-01 03:00:00",
                    ]
                ),
                "OT": np.arange(4, dtype=np.float64),
            }
        )
        spec = {
            "time_col": "date",
            "split_mode": "timestamp",
            "train_end_timestamp": "2026-01-01 00:00:00",
            "validation_end_timestamp": "2026-01-01 02:00:00",
            "train_ratio": 0.25,
            "val_ratio": 0.25,
            "test_ratio": 0.5,
        }
        with pytest.raises(ValueError, match="strictly chronological"):
            resolve_split_indices(frame, spec)

        sorted_frame = frame.sort_values("date").reset_index(drop=True)
        with pytest.raises(ValueError, match="do not match"):
            resolve_split_indices(
                sorted_frame,
                {
                    **spec,
                    "validation_end_timestamp": "2026-01-01 02:00:00",
                    "train_ratio": 0.5,
                },
            )

    def test_load_dataset_raises_on_empty_allowed_splits(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=20, freq="h"),
                "OT": np.linspace(0, 19, 20),
            }
        ).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="allowed_splits must not be empty"):
            load_dataset(
                {
                    "dataset_path": str(csv_path),
                    "target_col": "OT",
                    "time_col": "date",
                    "train_ratio": 0.7,
                    "val_ratio": 0.1,
                },
                allowed_splits=[],
            )

    def test_scaler_fit_on_train_only(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        train_values = np.linspace(0.0, 69.0, 70)
        adjustment_fit_values = np.full(5, 10000.0)
        val_values = np.full(5, 100.0)
        test_values = np.full(20, 1000.0)

        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="h"),
                "OT": np.concatenate(
                    [train_values, adjustment_fit_values, val_values, test_values]
                ),
            }
        ).to_csv(csv_path, index=False)

        result = load_dataset(
            {
                "dataset_path": str(csv_path),
                "target_col": "OT",
                "time_col": "date",
                "train_ratio": 0.7,
                "val_ratio": 0.1,
            },
            allowed_splits=["train", "val"],
        )

        assert np.mean(result["train"]["data"]) == pytest.approx(0.0, abs=1e-6)
        assert np.mean(result["val"]["data"]) > 1.0

    def test_adjustment_fit_uses_validation_front_with_training_context(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="h"),
                "OT": np.arange(100, dtype=np.float64),
            }
        ).to_csv(csv_path, index=False)

        result = load_dataset(
            {
                "dataset_path": str(csv_path),
                "target_col": "OT",
                "time_col": "date",
                "train_ratio": 0.7,
                "val_ratio": 0.1,
                "seq_len": 12,
            },
            allowed_splits=["train", "adjustment_fit"],
        )

        fit = result["adjustment_fit"]
        assert len(result["train"]["data"]) == 70
        assert fit["absolute_start_index"] == 58
        assert fit["target_start_index"] == 12
        assert len(fit["data"]) == 17
        assert np.asarray(fit["raw"])[12, 0] == 70.0
        assert np.asarray(fit["raw"])[-1, 0] == 74.0

    def test_forecast_gap_starts_targets_after_each_validation_boundary(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="h"),
                "OT": np.arange(100, dtype=np.float64),
            }
        ).to_csv(csv_path, index=False)

        result = load_dataset(
            {
                "dataset_path": str(csv_path),
                "target_col": "OT",
                "time_col": "date",
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "seq_len": 12,
                "pred_len": 3,
                "forecast_gap": 4,
            },
            allowed_splits=["train", "adjustment_fit", "val"],
        )

        fit = result["adjustment_fit"]
        validation = result["val"]
        assert fit["absolute_start_index"] == 48
        assert fit["forecast_origin_index"] == 12
        assert fit["target_start_index"] == 16
        assert fit["forecast_offset"] == 4
        assert np.asarray(fit["raw"])[16, 0] == 64.0
        assert validation["absolute_start_index"] == 58
        assert validation["forecast_origin_index"] == 12
        assert validation["target_start_index"] == 16
        assert validation["forecast_offset"] == 14
        assert np.asarray(validation["raw"])[16, 0] == 74.0

    def test_timestamp_origin_boundary_can_target_the_next_full_day(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "quarter-hour-data.csv"
        frame = pd.DataFrame(
            {
                "date": pd.date_range(
                    "2025-06-20 00:00:00",
                    "2025-10-02 23:45:00",
                    freq="15min",
                ),
                "OT": np.arange(
                    len(
                        pd.date_range(
                            "2025-06-20 00:00:00",
                            "2025-10-02 23:45:00",
                            freq="15min",
                        )
                    ),
                    dtype=np.float64,
                ),
            }
        )
        frame.to_csv(csv_path, index=False)

        result = load_dataset(
            {
                "dataset_path": str(csv_path),
                "target_col": "OT",
                "time_col": "date",
                "split_mode": "timestamp",
                "train_end_timestamp": "2025-06-30 11:45:00",
                "validation_end_timestamp": "2025-09-30 11:45:00",
                "seq_len": 96,
                "pred_len": 96,
                "forecast_gap": 48,
            },
            allowed_splits=["adjustment_fit", "test"],
        )

        fit = result["adjustment_fit"]
        fit_timestamps = np.asarray(fit["timestamps"]).astype(str)
        fit_origin = int(fit["forecast_origin_index"])
        fit_target = int(fit["target_start_index"])
        assert fit_timestamps[fit_origin - 1] == "2025-06-30 11:45:00"
        assert fit_timestamps[fit_target] == "2025-07-01 00:00:00"
        assert fit_timestamps[fit_target + 95] == "2025-07-01 23:45:00"

        test = result["test"]
        test_timestamps = np.asarray(test["timestamps"]).astype(str)
        test_origin = int(test["forecast_origin_index"])
        test_target = int(test["target_start_index"])
        assert test_timestamps[test_origin - 1] == "2025-09-30 11:45:00"
        assert test_timestamps[test_target] == "2025-10-01 00:00:00"
        assert test_timestamps[test_target + 95] == "2025-10-01 23:45:00"

    def test_forecast_gap_rejects_invalid_or_insufficient_context(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=40, freq="h"),
                "OT": np.arange(40, dtype=np.float64),
            }
        ).to_csv(csv_path, index=False)
        base_spec = {
            "dataset_path": str(csv_path),
            "target_col": "OT",
            "time_col": "date",
            "train_ratio": 0.5,
            "val_ratio": 0.3,
            "seq_len": 12,
            "pred_len": 2,
        }

        with pytest.raises(ValueError, match="forecast_gap must be a nonnegative integer"):
            load_dataset(
                {**base_spec, "forecast_gap": -1},
                allowed_splits=["train", "val"],
            )
        with pytest.raises(ValueError, match=r"shorter than forecast_gap \+ pred_len"):
            load_dataset(
                {**base_spec, "forecast_gap": 9},
                allowed_splits=["adjustment_fit"],
            )

        tiny_csv_path = tmp_path / "tiny.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="h"),
                "OT": np.arange(12, dtype=np.float64),
            }
        ).to_csv(tiny_csv_path, index=False)
        with pytest.raises(ValueError, match="shorter than seq_len"):
            load_dataset(
                {
                    **base_spec,
                    "dataset_path": str(tiny_csv_path),
                    "train_ratio": 0.5,
                    "val_ratio": 1 / 3,
                    "seq_len": 9,
                    "pred_len": 1,
                },
                allowed_splits=["train", "val"],
            )

    def test_non_adjustment_run_does_not_return_reserved_tail(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="h"),
                "OT": np.arange(100, dtype=np.float64),
            }
        ).to_csv(csv_path, index=False)

        result = load_dataset(
            {
                "dataset_path": str(csv_path),
                "target_col": "OT",
                "time_col": "date",
                "train_ratio": 0.7,
                "val_ratio": 0.1,
                "seq_len": 12,
            },
            allowed_splits=["train", "val"],
        )

        assert set(result) == {"train", "val"}
        assert "adjustment_fit" not in result

    def test_adjustment_fit_rejects_validation_front_shorter_than_horizon(
        self,
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=70, freq="h"),
                "OT": np.arange(70, dtype=np.float64),
            }
        ).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="shorter than pred_len"):
            load_dataset(
                {
                    "dataset_path": str(csv_path),
                    "target_col": "OT",
                    "time_col": "date",
                    "train_ratio": 0.7,
                    "val_ratio": 0.1,
                    "seq_len": 12,
                    "pred_len": 8,
                },
                allowed_splits=["train", "adjustment_fit"],
            )


class TestSAFE03EvalProtocolFixed:
    def test_evaluator_ignores_spec_metric_preferences(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "test"
        run_dir.mkdir(parents=True)
        hash_store = tmp_path / ".forecast" / "evaluator.sha256"

        pd.DataFrame(
            {
                "timestamp": ["t1", "t2"],
                "actual": [1.0, 2.0],
                "predicted": [1.5, 2.5],
            }
        ).to_csv(run_dir / "predictions.csv", index=False)

        result = evaluate(
            run_dir,
            {
                "model": "test",
                "eval_split": "val",
                "train_ratio": 0.7,
                "val_ratio": 0.1,
                "metrics": ["rmse"],
            },
            hash_store,
        )

        assert set(result["metrics"]) == {"mse", "mae", "wape", "mase"}


class TestSAFE04DriftDetection:
    def test_eval_json_contains_hashes(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "test"
        run_dir.mkdir(parents=True)
        hash_store = tmp_path / ".forecast" / "evaluator.sha256"

        pd.DataFrame(
            {
                "timestamp": ["t1"],
                "actual": [1.0],
                "predicted": [1.0],
            }
        ).to_csv(run_dir / "predictions.csv", index=False)

        result = evaluate(
            run_dir,
            {"model": "test", "eval_split": "val", "train_ratio": 0.7, "val_ratio": 0.1},
            hash_store,
        )

        assert len(result["evaluator_hash"]) == 64
        assert result["spec_hash"] == result["run_id"]

    def test_modified_evaluator_detected(self, tmp_path: Path) -> None:
        evaluator_py = tmp_path / "evaluator.py"
        evaluator_py.write_text("original content\n", encoding="utf-8")
        store = tmp_path / ".forecast" / "evaluator.sha256"

        verify_evaluator(evaluator_py, store)
        evaluator_py.write_text("modified content\n", encoding="utf-8")

        with pytest.raises(SystemExit, match="INTEGRITY VIOLATION"):
            verify_evaluator(evaluator_py, store)
