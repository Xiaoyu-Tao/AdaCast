"""Regression tests for deterministic adjustment application artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import adacast_ml.adjustment.apply as adjustment_apply


def test_zero_trigger_preserves_raw_predictions_exactly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "series.csv"
    task_path = tmp_path / "task.json"
    predictions_path = tmp_path / "predictions.csv"
    candidate_path = tmp_path / "candidate.json"
    out_dir = tmp_path / "adjusted"

    timestamps = pd.date_range("2026-01-01", periods=24, freq="15min")
    pd.DataFrame(
        {
            "date": timestamps.astype(str),
            "OT": np.linspace(1.0, 24.0, 24),
        }
    ).to_csv(dataset_path, index=False)
    task_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "time_col": "date",
                "target_col": "OT",
                "pred_len": 4,
                "train_ratio": 0.5,
                "val_ratio": 0.3,
                "test_ratio": 0.2,
            }
        ),
        encoding="utf-8",
    )

    rng = np.random.default_rng(871)
    raw_predictions = rng.normal(size=12)
    pd.DataFrame(
        {
            "timestamp": timestamps[:12].astype(str),
            "predicted": raw_predictions,
            "actual": rng.normal(size=12),
        }
    ).to_csv(predictions_path, index=False)
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "dormant-skill",
                "rules": [
                    {
                        "id": "february-only",
                        "condition": {"kind": "timestamp_month_in", "values": [2]},
                        "action": {"type": "add", "value": 100.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # This scale makes a full original-scale round-trip measurably lossy for
    # many values. A zero-trigger application must bypass that round-trip.
    monkeypatch.setattr(
        adjustment_apply,
        "target_scale",
        lambda _task, _task_path: (719022922.1174326, 0.00015653732231190253),
    )

    adjustment_apply.main(
        [
            "--task-json",
            str(task_path),
            "--predictions-csv",
            str(predictions_path),
            "--candidate-json",
            str(candidate_path),
            "--out-dir",
            str(out_dir),
            "--source-run-id",
            "validation-run",
            "--split",
            "val",
        ]
    )

    adjusted = pd.read_csv(out_dir / "adjusted_predictions.csv")
    np.testing.assert_array_equal(adjusted["predicted"], adjusted["raw_predicted"])

    apply_result = json.loads((out_dir / "apply-results.json").read_text(encoding="utf-8"))
    assert apply_result["matched_rows"] == 0
    assert apply_result["changed_rows"] == 0
    assert apply_result["triggered_rule_count"] == 0
    assert apply_result["zero_trigger"] is True


def test_enforce_delta_bound_preserves_valid_downward_adjustments() -> None:
    raw = np.asarray([50_000.0, 50_000.0, 50_000.0], dtype=np.float64)
    proposed = np.asarray([49_500.0, 48_000.0, 50_500.0], dtype=np.float64)

    bounded, clipped = adjustment_apply.enforce_delta_bound(
        raw_original=raw,
        adjusted_original=proposed,
        max_delta_ratio=0.1,
        min_delta_original=-1_000.0,
        max_delta_original=0.0,
    )

    np.testing.assert_array_equal(bounded, np.asarray([49_500.0, 49_000.0, 50_000.0]))
    assert clipped == 2


def test_enforce_delta_bound_keeps_upward_and_symmetric_limits_unchanged() -> None:
    raw = np.asarray([50_000.0, 100.0], dtype=np.float64)

    upward, upward_clipped = adjustment_apply.enforce_delta_bound(
        raw_original=raw,
        adjusted_original=np.asarray([51_000.0, 80.0]),
        max_delta_ratio=0.1,
        min_delta_original=0.0,
        max_delta_original=3_000.0,
    )
    np.testing.assert_array_equal(upward, np.asarray([51_000.0, 100.0]))
    assert upward_clipped == 1

    symmetric, symmetric_clipped = adjustment_apply.enforce_delta_bound(
        raw_original=raw,
        adjusted_original=np.asarray([56_000.0, 80.0]),
        max_delta_ratio=0.1,
    )
    np.testing.assert_array_equal(symmetric, np.asarray([55_000.0, 90.0]))
    assert symmetric_clipped == 2


def test_candidate_application_is_first_match_wins(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "series.csv"
    task_path = tmp_path / "task.json"
    predictions_path = tmp_path / "predictions.csv"
    candidate_path = tmp_path / "candidate.json"
    out_dir = tmp_path / "adjusted"

    timestamps = pd.date_range("2026-01-01", periods=12, freq="h")
    pd.DataFrame({"date": timestamps.astype(str), "OT": np.arange(12.0)}).to_csv(
        dataset_path, index=False
    )
    task_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "time_col": "date",
                "target_col": "OT",
                "pred_len": 4,
                "train_ratio": 0.5,
                "val_ratio": 0.3,
                "test_ratio": 0.2,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"timestamp": timestamps.astype(str), "predicted": np.full(12, 10.0), "actual": np.full(12, 10.0)}
    ).to_csv(predictions_path, index=False)
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "priority-rules",
                "rules": [
                    {
                        "id": "first",
                        "condition": {"kind": "timestamp_month_in", "values": [1]},
                        "action": {"type": "add", "value": 1.0},
                    },
                    {
                        "id": "second-overlap",
                        "condition": {"kind": "timestamp_month_in", "values": [1]},
                        "action": {"type": "add", "value": 5.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adjustment_apply, "target_scale", lambda _task, _path: (0.0, 1.0))

    adjustment_apply.main(
        [
            "--task-json",
            str(task_path),
            "--predictions-csv",
            str(predictions_path),
            "--candidate-json",
            str(candidate_path),
            "--out-dir",
            str(out_dir),
            "--source-run-id",
            "validation-run",
            "--split",
            "val",
        ]
    )

    adjusted = pd.read_csv(out_dir / "adjusted_predictions.csv")
    np.testing.assert_array_equal(adjusted["predicted"], np.full(12, 11.0))
    result = json.loads((out_dir / "apply-results.json").read_text(encoding="utf-8"))
    assert result["applied_rules"][0]["matched_rows"] == 12
    assert result["applied_rules"][1]["condition_matched_rows"] == 12
    assert result["applied_rules"][1]["matched_rows"] == 0
    assert result["applied_rules"][1]["overlap_skipped_rows"] == 12
