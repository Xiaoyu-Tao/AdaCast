"""Tests for adjustment diagnostics evidence references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adacast_ml.adjustment import apply, diagnostics
from adacast_ml.adjustment.diagnostics import diagnostics_evidence_ref
from adacast_ml.adjustment.prediction_layout import prediction_coordinates


def test_rule_hint_evidence_points_to_canonical_fit_diagnostics(tmp_path: Path) -> None:
    task_path = tmp_path / ".forecast" / "task.json"
    out_path = (
        tmp_path
        / ".forecast"
        / "adjustments"
        / "diagnostics"
        / "fit"
        / "fit-run"
        / "diagnostics.json"
    )

    assert diagnostics_evidence_ref(task_path, out_path) == (
        ".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"
    )


def test_explicit_continuous_prediction_coordinates_are_not_wrapped() -> None:
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0, 0, 0],
            "horizon_index": [11, 12, 13, 14],
        }
    )

    windows, horizons = prediction_coordinates(predictions, pred_len=12)

    assert windows.tolist() == [0, 0, 0, 0]
    assert horizons.tolist() == [11, 12, 13, 14]


def test_partial_or_fractional_prediction_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="both window_index and horizon_index"):
        prediction_coordinates(pd.DataFrame({"window_index": [0]}), pred_len=12)
    with pytest.raises(ValueError, match="non-negative integers"):
        prediction_coordinates(
            pd.DataFrame({"window_index": [0], "horizon_index": [0.5]}),
            pred_len=12,
        )


def test_strided_prediction_coordinates_survive_diagnostics_and_apply(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "series.csv"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=60, freq="h"),
            "target": 10.0 + np.sin(np.arange(60) / 4.0),
            "weather": np.linspace(0.0, 1.0, 60),
        }
    )
    frame.to_csv(dataset_path, index=False)
    task = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 3,
        "forecast_gap": 2,
        "window_stride": 2,
        "train_ratio": 0.5,
        "val_ratio": 0.3,
        "test_ratio": 0.2,
    }
    task_path = tmp_path / ".forecast" / "task.json"
    task_path.parent.mkdir()
    task_path.write_text(json.dumps(task), encoding="utf-8")

    train_target = frame["target"].iloc[:30].to_numpy(dtype=np.float64)
    mean = float(np.mean(train_target))
    scale = float(np.std(train_target))
    window_starts = [30, 32, 34, 36]
    prediction_rows = []
    for window_index, start in enumerate(window_starts):
        for horizon_index in range(3):
            absolute_index = start + horizon_index
            actual = float((frame["target"].iloc[absolute_index] - mean) / scale)
            prediction_rows.append(
                {
                    "timestamp": str(frame["date"].iloc[absolute_index]),
                    "actual": actual,
                    "predicted": actual - 0.1,
                    "prediction_layout": "sliding_window",
                    "window_index": window_index,
                    "horizon_index": horizon_index,
                }
            )
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics.main(
        [
            "--task-json",
            str(task_path),
            "--predictions-csv",
            str(predictions_path),
            "--out",
            str(diagnostics_path),
        ]
    )
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics_payload["observed_condition_domains"]["horizon_indices"] == [
        0,
        1,
        2,
    ]
    assert diagnostics_payload["overall"]["row_count"] == 12
    assert diagnostics_payload["task_context"]["window_contract"] == {
        "seq_len": 8,
        "forecast_gap": 2,
        "pred_len": 3,
        "window_stride": 2,
        "horizon_index_zero_meaning": "first evaluated point after forecast_gap",
    }
    evidence_candidates = diagnostics_payload["evidence_candidates"]
    assert set(evidence_candidates) == {
        "classification_contract",
        "experience",
        "context",
        "knowledge",
    }
    assert evidence_candidates["experience"]["rule_hints"] == diagnostics_payload["rule_hints"]
    assert evidence_candidates["experience"]["evidence_candidate_id"] == "experience:validation-front"
    assert evidence_candidates["context"]["evidence_candidate_id"] == "context:validation-front"
    assert evidence_candidates["knowledge"]["evidence_candidate_id"] == "knowledge:frozen-task"
    assert evidence_candidates["context"]["observed_condition_domains"] == diagnostics_payload[
        "observed_condition_domains"
    ]
    assert "dataset_feature_ledger" not in evidence_candidates["context"]
    assert evidence_candidates["knowledge"]["dataset_feature_ledger"] == diagnostics_payload[
        "dataset_feature_ledger"
    ]
    assert evidence_candidates["knowledge"]["task_semantics_and_structure"]["window_contract"] == diagnostics_payload[
        "task_context"
    ]["window_contract"]
    for candidate_type in ("experience", "context", "knowledge"):
        artifact = diagnostics_payload["evidence_candidate_storage"][candidate_type]
        artifact_path = Path(artifact["path"])
        assert artifact_path.name == f"evidence-{candidate_type}.json"
        assert artifact_path.exists()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "stride-candidate",
                "rules": [
                    {
                        "id": "middle-horizon-only",
                        "condition": {"kind": "horizon_index_in", "values": [1]},
                        "action": {"type": "add", "value": 0.01},
                        "bounds": {"max_delta_ratio": 0.1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "adjusted"
    apply.main(
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
            "stride-run",
            "--split",
            "fit",
        ]
    )

    adjusted = pd.read_csv(out_dir / "adjusted_predictions.csv")
    assert adjusted["window_index"].tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]
    assert adjusted["horizon_index"].tolist() == [0, 1, 2] * 4
    changed = ~np.isclose(
        adjusted["predicted"].to_numpy(dtype=np.float64),
        adjusted["raw_predicted"].to_numpy(dtype=np.float64),
    )
    assert changed.tolist() == [False, True, False] * 4

    blind_input = json.loads((out_dir / "blind-input.json").read_text(encoding="utf-8"))
    assert [row["window_index"] for row in blind_input["rows"]] == adjusted[
        "window_index"
    ].tolist()
    assert [row["horizon_index"] for row in blind_input["rows"]] == adjusted[
        "horizon_index"
    ].tolist()
    assert all("actual" not in row for row in blind_input["rows"])


def test_user_prior_style_text_shifted_time_and_absolute_cap_rule(tmp_path: Path) -> None:
    dataset_path = tmp_path / "load_calendar.csv"
    timestamps = pd.date_range("2026-01-01 00:00", periods=72, freq="h")
    dataset = pd.DataFrame(
        {
            "date": timestamps,
            "target": 60000.0,
            "天气": ["阴" if item.date().isoformat() == "2026-01-01" else "晴" for item in timestamps],
            "是否极端天气": ["是" if item.date().isoformat() == "2026-01-01" else "否" for item in timestamps],
            "节假日": ["元旦复工" if item.date().isoformat() == "2026-01-02" else "" for item in timestamps],
        }
    )
    dataset.to_csv(dataset_path, index=False)
    task = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 4,
        "train_ratio": 0.5,
        "val_ratio": 0.3,
        "test_ratio": 0.2,
    }
    task_path = tmp_path / ".forecast" / "task.json"
    task_path.parent.mkdir()
    task_path.write_text(json.dumps(task), encoding="utf-8")

    predictions = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 06:00:00",
                "2026-01-01 07:00:00",
                "2026-01-01 08:00:00",
                "2026-01-01 09:00:00",
            ],
            "actual": [0.0, 0.0, 0.0, 0.0],
            "predicted": [0.0, 0.0, 0.0, 0.0],
            "window_index": [0, 0, 0, 0],
            "horizon_index": [0, 1, 2, 3],
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "user-prior-risk-day",
                "rules": [
                    {
                        "id": "risky-day-up-only-after-morning",
                        "condition": {
                            "kind": "all",
                            "conditions": [
                                {"kind": "timestamp_time_between", "start": "07:15", "end": "23:45"},
                                {"kind": "exogenous_text_in", "column": "天气", "values": ["阴"]},
                                {"kind": "exogenous_bool_is", "column": "是否极端天气", "value": True},
                                {
                                    "kind": "exogenous_text_contains",
                                    "column": "节假日",
                                    "values": ["元旦复工"],
                                    "day_offset": 1,
                                },
                            ],
                        },
                        "action": {"type": "add", "value": 5000.0},
                        "bounds": {"min_delta_original": 0.0, "max_delta_original": 3000.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"max_point_delta_ratio": 1.0}), encoding="utf-8")

    out_dir = tmp_path / "adjusted"
    apply.main(
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
            "user-prior-run",
            "--split",
            "fit",
            "--policy-json",
            str(policy_path),
        ]
    )

    adjusted = pd.read_csv(out_dir / "adjusted_predictions.csv")
    assert adjusted["adjusted_predicted_original"].tolist() == [
        60000.0,
        60000.0,
        63000.0,
        63000.0,
    ]
    blind_input = json.loads((out_dir / "blind-input.json").read_text(encoding="utf-8"))
    assert "exog_text__天气" in blind_input["rows"][0]
    assert "exog_text__day_p1__节假日" in blind_input["rows"][0]
    assert all("actual" not in row for row in blind_input["rows"])


def test_diagnostics_exposes_text_shifted_and_minute_domains(tmp_path: Path) -> None:
    dataset_path = tmp_path / "load_calendar.csv"
    timestamps = pd.date_range("2026-01-01 00:00", periods=72, freq="h")
    dataset = pd.DataFrame(
        {
            "date": timestamps,
            "target": 100.0,
            "天气": ["阴" if item.date().isoformat() == "2026-01-01" else "晴" for item in timestamps],
            "节假日": ["元旦复工" if item.date().isoformat() == "2026-01-02" else "" for item in timestamps],
        }
    )
    dataset.to_csv(dataset_path, index=False)
    task = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 2,
        "train_ratio": 0.5,
        "val_ratio": 0.3,
        "test_ratio": 0.2,
        "adjustment_skill_priors": "When next day is 元旦复工 and the current day is 阴, increase only after 07:00.",
    }
    task_path = tmp_path / ".forecast" / "task.json"
    task_path.parent.mkdir()
    task_path.write_text(json.dumps(task), encoding="utf-8")
    predictions = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 08:00:00", "2026-01-01 09:00:00"],
            "actual": [0.5, 0.6],
            "predicted": [0.4, 0.5],
            "window_index": [0, 0],
            "horizon_index": [0, 1],
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    diagnostics_path = tmp_path / "diagnostics.json"

    diagnostics.main(
        [
            "--task-json",
            str(task_path),
            "--predictions-csv",
            str(predictions_path),
            "--out",
            str(diagnostics_path),
        ]
    )

    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    domains = payload["observed_condition_domains"]
    assert domains["timestamp_minutes_of_day"] == [480, 540]
    assert "阴" in domains["exogenous_text"]["天气"]["values"]
    assert "元旦复工" in domains["exogenous_text"]["day_p1__节假日"]["values"]
    assert payload["task_context"]["adjustment_skill_priors"].startswith("When next day")
    ledger = {row["column"]: row for row in payload["dataset_feature_ledger"]}
    assert ledger["天气"]["usable_for_exogenous_text_conditions"] is True
