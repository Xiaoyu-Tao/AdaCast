"""Tests for importing user-submitted prediction priors."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from adacast_ml import prediction_prior


def _write_real_and_prior(tmp_path: Path) -> tuple[Path, Path, Path]:
    forecast_root = tmp_path / ".forecast"
    forecast_root.mkdir()
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    real = pd.DataFrame(
        {
            "date": dates.astype(str),
            "y": [float(i) for i in range(100)],
            "exog": [float(i % 7) for i in range(100)],
        }
    )
    prior = real.copy()
    prior.loc[50:, "y"] = prior.loc[50:, "y"] + 10.0
    real_path = tmp_path / "real.csv"
    prior_path = tmp_path / "prior.csv"
    real.to_csv(real_path, index=False)
    prior.to_csv(prior_path, index=False)
    task = {
        "dataset_path": str(real_path),
        "target_col": "y",
        "time_col": "date",
        "freq": "h",
        "seq_len": 4,
        "label_len": 4,
        "pred_len": 3,
        "window_stride": 2,
        "split_mode": "ratio",
        "train_ratio": 0.5,
        "val_ratio": 0.3,
        "test_ratio": 0.2,
        "eval_metrics": ["mse", "mae"],
        "model_families": ["UserPrior"],
        "frozen_at": "2026-06-25T00:00:00.000Z",
    }
    task_path = forecast_root / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return task_path, prior_path, forecast_root


def _args(
    *,
    task_path: Path,
    prior_path: Path,
    forecast_root: Path,
    action: str = "import",
    source_validation_run_id: str | None = None,
    expected_prior_sha256: str | None = None,
) -> Namespace:
    return Namespace(
        action=action,
        task_json=str(task_path),
        prior_dataset_path=str(prior_path),
        forecast_root=str(forecast_root),
        evaluator_hash_store=str(forecast_root / "evaluator.sha256"),
        source_validation_run_id=source_validation_run_id,
        expected_prior_sha256=expected_prior_sha256,
    )


def test_prediction_prior_import_and_materialize_test_are_split_safe(tmp_path: Path) -> None:
    task_path, prior_path, forecast_root = _write_real_and_prior(tmp_path)

    imported = prediction_prior.run_import(
        _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
    )

    assert imported["source_model_set"]["created_from"] == "user_prediction_prior"
    assert imported["source_model_set"]["members"][0]["model"] == "UserPrior"
    assert imported["fit_run_manifest"]["status"] == "complete"
    assert imported["validation_run"]["run_id"] != imported["fit_run"]["run_id"]
    assert not any(
        (forecast_root / "runs" / run_id / "spec.json").exists()
        for run_id in [imported["fit_run"]["run_id"]]
    )

    validation_predictions = pd.read_csv(
        tmp_path / imported["validation_run"]["predictions_path"]
    )
    fit_predictions = pd.read_csv(tmp_path / imported["fit_run"]["predictions_path"])
    assert validation_predictions.iloc[0]["timestamp"] == "2024-01-03 18:00:00"
    assert fit_predictions.iloc[0]["timestamp"] == "2024-01-03 02:00:00"
    assert validation_predictions["window_index"].tolist()[:6] == [0, 0, 0, 1, 1, 1]

    test_result = prediction_prior.run_materialize_test(
        _args(
            task_path=task_path,
            prior_path=prior_path,
            forecast_root=forecast_root,
            action="materialize-test",
            source_validation_run_id=imported["validation_run"]["run_id"],
            expected_prior_sha256=imported["prior_dataset_sha256"],
        )
    )

    test_predictions = pd.read_csv(tmp_path / test_result["test_run"]["predictions_path"])
    assert test_predictions.iloc[0]["timestamp"] == "2024-01-04 08:00:00"
    assert test_predictions["prediction_layout"].unique().tolist() == ["sliding_window"]
    assert test_result["test_run"]["run_id"] != imported["validation_run"]["run_id"]
    assert (forecast_root / "runs" / test_result["test_run"]["run_id"] / "model" / "reuse.json").exists()
    manifest = prediction_prior.load_json(forecast_root / "user-prior" / "manifest.json")
    assert manifest["test_run_id"] == test_result["test_run"]["run_id"]


def test_short_horizon_prior_keeps_one_grid_across_validation_halves(
    tmp_path: Path,
) -> None:
    forecast_root = tmp_path / ".forecast"
    forecast_root.mkdir()
    row_count = 2_000
    train_end = 1_400
    adjustment_fit_end = 1_500
    val_end = 1_600
    dates = pd.date_range("2024-01-01", periods=row_count, freq="h")
    real = pd.DataFrame(
        {
            "date": dates.astype(str),
            "y": [float(i) for i in range(row_count)],
            "exog": [float(i % 24) for i in range(row_count)],
        }
    )
    prior = real.copy()
    # Mimic a 24-step forecast written every 48 rows. Rows between forecast
    # blocks intentionally remain equal to the ground truth.
    prediction_target_anchor = 1_440
    for target_start in range(prediction_target_anchor, row_count - 24 + 1, 48):
        prior.loc[target_start : target_start + 23, "y"] += 10.0

    real_path = tmp_path / "real-short.csv"
    prior_path = tmp_path / "prior-short.csv"
    real.to_csv(real_path, index=False)
    prior.to_csv(prior_path, index=False)
    task = {
        "dataset_path": str(real_path),
        "target_col": "y",
        "time_col": "date",
        "freq": "h",
        "seq_len": 168,
        "label_len": 168,
        "pred_len": 24,
        "window_stride": 48,
        "split_mode": "ratio",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "eval_metrics": ["mse", "mae"],
        "model_families": ["UserPrior"],
        "frozen_at": "2026-08-19T00:00:00.000Z",
    }
    task_path = forecast_root / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")

    fit_starts = prediction_prior.prediction_window_starts(
        "adjustment_fit",
        train_end=train_end,
        adjustment_fit_end=adjustment_fit_end,
        val_end=val_end,
        row_count=row_count,
        seq_len=168,
        forecast_gap=0,
        pred_len=24,
        window_stride=48,
        prediction_target_anchor=prediction_target_anchor,
    )
    validation_starts = prediction_prior.prediction_window_starts(
        "val",
        train_end=train_end,
        adjustment_fit_end=adjustment_fit_end,
        val_end=val_end,
        row_count=row_count,
        seq_len=168,
        forecast_gap=0,
        pred_len=24,
        window_stride=48,
        prediction_target_anchor=prediction_target_anchor,
    )
    assert [start + 168 for start in fit_starts] == [1_440]
    assert [start + 168 for start in validation_starts] == [1_536]
    assert all((start - fit_starts[0]) % 48 == 0 for start in validation_starts)

    imported = prediction_prior.run_import(
        _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
    )
    source = prediction_prior.load_json(
        forecast_root
        / "runs"
        / imported["validation_run"]["run_id"]
        / "model"
        / "prior-source.json"
    )
    fit_predictions = pd.read_csv(tmp_path / imported["fit_run"]["predictions_path"])
    validation_predictions = pd.read_csv(
        tmp_path / imported["validation_run"]["predictions_path"]
    )

    assert fit_predictions.groupby("window_index").size().tolist() == [24]
    assert validation_predictions.groupby("window_index").size().tolist() == [24]
    assert validation_predictions.iloc[0]["timestamp"] == str(dates[1_536])
    assert validation_predictions["horizon_index"].tolist() == list(range(24))
    assert (validation_predictions["predicted"] != validation_predictions["actual"]).all()
    assert source["prediction_grid_target_anchor_index"] == prediction_target_anchor


def test_prediction_prior_resets_grid_at_test_split_boundary(tmp_path: Path) -> None:
    forecast_root = tmp_path / ".forecast"
    forecast_root.mkdir()
    row_count = 2_000
    train_end = 1_400
    val_end = 1_600
    dates = pd.date_range("2024-01-01", periods=row_count, freq="h")
    real = pd.DataFrame(
        {
            "date": dates.astype(str),
            "y": [float(i) for i in range(row_count)],
            "exog": [float(i % 24) for i in range(row_count)],
        }
    )
    prior = real.copy()
    validation_target_anchor = 1_440
    for target_start in range(validation_target_anchor, val_end - 24 + 1, 48):
        prior.loc[target_start : target_start + 23, "y"] += 10.0
    test_target_anchor = val_end
    for target_start in range(test_target_anchor, row_count - 24 + 1, 48):
        prior.loc[target_start : target_start + 23, "y"] += 20.0

    real_path = tmp_path / "real-split-grids.csv"
    prior_path = tmp_path / "prior-split-grids.csv"
    real.to_csv(real_path, index=False)
    prior.to_csv(prior_path, index=False)
    task = {
        "dataset_path": str(real_path),
        "target_col": "y",
        "time_col": "date",
        "freq": "h",
        "seq_len": 168,
        "label_len": 168,
        "pred_len": 24,
        "window_stride": 48,
        "split_mode": "ratio",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "eval_metrics": ["mse", "mae"],
        "model_families": ["UserPrior"],
        "frozen_at": "2026-08-21T00:00:00.000Z",
    }
    task_path = forecast_root / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")

    imported = prediction_prior.run_import(
        _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
    )
    test_result = prediction_prior.run_materialize_test(
        _args(
            task_path=task_path,
            prior_path=prior_path,
            forecast_root=forecast_root,
            action="materialize-test",
            source_validation_run_id=imported["validation_run"]["run_id"],
            expected_prior_sha256=imported["prior_dataset_sha256"],
        )
    )

    test_predictions = pd.read_csv(tmp_path / test_result["test_run"]["predictions_path"])
    expected_starts = list(range(test_target_anchor, row_count - 24 + 1, 48))
    assert test_predictions.groupby("window_index").size().tolist() == [24] * len(expected_starts)
    assert test_predictions.iloc[0]["timestamp"] == str(dates[test_target_anchor])
    assert test_predictions.iloc[-1]["timestamp"] == str(dates[expected_starts[-1] + 23])
    assert (test_predictions["predicted"] != test_predictions["actual"]).all()
    test_spec = prediction_prior.load_json(
        tmp_path / test_result["test_run"]["spec_path"]
    )
    assert test_spec["prediction_grid_target_anchor_index"] == test_target_anchor


def test_prediction_prior_rejects_test_materialization_after_frozen_task_changes(
    tmp_path: Path,
) -> None:
    task_path, prior_path, forecast_root = _write_real_and_prior(tmp_path)
    imported = prediction_prior.run_import(
        _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
    )
    task = prediction_prior.load_json(task_path)
    task["reflection_enabled"] = False
    task_path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match="Frozen task contract changed"):
        prediction_prior.run_materialize_test(
            _args(
                task_path=task_path,
                prior_path=prior_path,
                forecast_root=forecast_root,
                action="materialize-test",
                source_validation_run_id=imported["validation_run"]["run_id"],
                expected_prior_sha256=imported["prior_dataset_sha256"],
            )
        )


def test_prediction_prior_rejects_non_target_changes(tmp_path: Path) -> None:
    task_path, prior_path, forecast_root = _write_real_and_prior(tmp_path)
    prior = pd.read_csv(prior_path)
    prior.loc[55, "exog"] = 999.0
    prior.to_csv(prior_path, index=False)

    with pytest.raises(ValueError, match="may only replace validation/test target values"):
        prediction_prior.run_import(
            _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
        )


def test_prediction_prior_ignores_text_and_incomplete_calendar_columns_in_matrices(tmp_path: Path) -> None:
    task_path, prior_path, forecast_root = _write_real_and_prior(tmp_path)
    task = prediction_prior.load_json(task_path)
    real_path = Path(task["dataset_path"])
    real = pd.read_csv(real_path)
    prior = pd.read_csv(prior_path)
    real["holiday_text"] = ["workday"] * 50 + ["holiday"] * 50
    prior["holiday_text"] = real["holiday_text"]
    real["partial_calendar_numeric"] = [None] * 50 + [1.0] * 50
    prior["partial_calendar_numeric"] = real["partial_calendar_numeric"]
    real.to_csv(real_path, index=False)
    prior.to_csv(prior_path, index=False)

    imported = prediction_prior.run_import(
        _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
    )

    original = pd.read_csv(real_path)
    prior_frame = pd.read_csv(prior_path)
    columns = prediction_prior.scaled_matrices(original, prior_frame, task, train_end=50)[2]
    assert columns == ["y", "exog"]
    assert imported["source_model_set"]["members"][0]["model"] == "UserPrior"


def test_prediction_prior_applies_gap_without_materializing_gap_rows(tmp_path: Path) -> None:
    task_path, prior_path, forecast_root = _write_real_and_prior(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["forecast_gap"] = 2
    task_path.write_text(json.dumps(task), encoding="utf-8")

    imported = prediction_prior.run_import(
        _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
    )
    fit_predictions = pd.read_csv(tmp_path / imported["fit_run"]["predictions_path"])
    validation_predictions = pd.read_csv(
        tmp_path / imported["validation_run"]["predictions_path"]
    )

    assert fit_predictions.iloc[0]["timestamp"] == "2024-01-03 04:00:00"
    assert validation_predictions.iloc[0]["timestamp"] == "2024-01-03 20:00:00"
    assert fit_predictions.groupby("window_index").size().unique().tolist() == [3]
    assert validation_predictions.groupby("window_index").size().unique().tolist() == [3]

    test_result = prediction_prior.run_materialize_test(
        _args(
            task_path=task_path,
            prior_path=prior_path,
            forecast_root=forecast_root,
            action="materialize-test",
            source_validation_run_id=imported["validation_run"]["run_id"],
            expected_prior_sha256=imported["prior_dataset_sha256"],
        )
    )
    test_predictions = pd.read_csv(tmp_path / test_result["test_run"]["predictions_path"])
    assert test_predictions.iloc[0]["timestamp"] == "2024-01-04 10:00:00"
    assert test_predictions["timestamp"].tolist()[:3] == [
        "2024-01-04 10:00:00",
        "2024-01-04 11:00:00",
        "2024-01-04 12:00:00",
    ]
    assert test_predictions["prediction_layout"].unique().tolist() == ["sliding_window"]
    test_run_dir = forecast_root / "runs" / test_result["test_run"]["run_id"]
    gap_predictions = pd.read_csv(test_run_dir / "gap_predictions.csv")
    assert gap_predictions["timestamp"].tolist()[:2] == [
        "2024-01-04 08:00:00",
        "2024-01-04 09:00:00",
    ]
    assert gap_predictions["gap_index"].tolist()[:2] == [0, 1]
    assert "actual" not in gap_predictions.columns


def test_prediction_prior_contract_rejects_legacy_mean_horizon_layout() -> None:
    timestamps = pd.date_range("2024-01-01", periods=16, freq="h").astype(str).to_numpy()
    rows = [
        {
            "timestamp": str(timestamps[6 + horizon]),
            "actual": 0.0,
            "predicted": 0.0,
            "prediction_layout": "mean_prediction_by_horizon",
            "window_index": 0,
            "horizon_index": horizon,
        }
        for horizon in range(3)
    ]

    with pytest.raises(ValueError, match="prediction_layout='sliding_window'"):
        prediction_prior.validate_prediction_rows(
            rows=rows,
            timestamps=timestamps,
            split="test",
            window_start_indices=[0],
            seq_len=4,
            forecast_gap=2,
            pred_len=3,
            window_stride=2,
        )


def test_prediction_prior_rejects_invalid_forecast_gap(tmp_path: Path) -> None:
    task_path, prior_path, forecast_root = _write_real_and_prior(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["forecast_gap"] = -1
    task_path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match="forecast_gap must be a nonnegative integer"):
        prediction_prior.run_import(
            _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
        )


def test_prediction_prior_rejects_gap_larger_than_available_target_span(tmp_path: Path) -> None:
    task_path, prior_path, forecast_root = _write_real_and_prior(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["forecast_gap"] = 48
    task_path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(ValueError, match=r"shorter than forecast_gap=48 plus pred_len=3"):
        prediction_prior.run_import(
            _args(task_path=task_path, prior_path=prior_path, forecast_root=forecast_root)
        )
