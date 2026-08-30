from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from adacast_ml.critic import pattern_search, trajectory_viz, window_viz
from adacast_ml.critic.persistence_baseline import compute_persistence_baseline


def write_dataset(path: Path, rows: int = 70) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="15min"),
            "target": np.sin(np.arange(rows) / 5.0),
            "feature": np.cos(np.arange(rows) / 7.0),
        }
    )
    frame.to_csv(path, index=False)


def test_persistence_baseline_uses_runner_sliding_windows_and_scale(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "data.csv"
    write_dataset(dataset_path, rows=100)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "model": "DLinear",
        "seq_len": 12,
        "label_len": 6,
        "pred_len": 4,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "phase": "post-forecast",
        "eval_split": "test",
    }
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

    result = compute_persistence_baseline(run_dir)

    assert result["eval_split"] == "test"
    assert result["scale"] == "runner_standardized_training_prefix"
    assert result["window_count"] == 17
    assert result["point_count"] == 68
    assert all(np.isfinite(value) for value in result["metrics"].values())


def test_persistence_baseline_uses_frozen_window_stride(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.csv"
    write_dataset(dataset_path, rows=100)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "model": "DLinear",
        "seq_len": 12,
        "label_len": 6,
        "pred_len": 4,
        "window_stride": 2,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "phase": "post-forecast",
        "eval_split": "test",
    }
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

    result = compute_persistence_baseline(run_dir)

    assert result["window_count"] == 9
    assert result["point_count"] == 36


def test_persistence_baseline_applies_forecast_gap(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.csv"
    write_dataset(dataset_path, rows=100)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "model": "DLinear",
        "seq_len": 12,
        "label_len": 6,
        "pred_len": 4,
        "forecast_gap": 3,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "phase": "post-forecast",
        "eval_split": "test",
    }
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

    result = compute_persistence_baseline(run_dir)

    assert result["window_count"] == 14
    assert result["point_count"] == 56


def test_trajectory_deduplicates_run_ids_and_excludes_test_runs(
    tmp_path: Path,
) -> None:
    history = tmp_path / "history.jsonl"
    rows = [
        {
            "run_id": "val-success",
            "model": "A",
            "status": "success",
            "metrics": {"mse": 1.0, "mae": 0.5},
            "spec": {"eval_split": "val"},
        },
        {
            "run_id": "val-success",
            "model": "A",
            "status": "success",
            "metrics": {"mse": 1.0, "mae": 0.5},
            "spec": {"model": "manual"},
        },
        {
            "run_id": "val-crash",
            "model": "B",
            "status": "crash",
            "metrics": None,
            "spec": {"eval_split": "val"},
        },
        {
            "run_id": "test-success",
            "model": "A",
            "status": "success",
            "metrics": {"mse": 0.8, "mae": 0.4},
            "spec": {"eval_split": "test"},
        },
    ]
    history.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    experiments, attempt_count = trajectory_viz.load_experiment_metrics(str(history))

    assert attempt_count == 2
    assert len(experiments) == 1
    assert experiments[0]["round"] == 1


def test_pattern_search_derives_leakage_safe_boundaries(tmp_path: Path, capsys) -> None:
    dataset_path = tmp_path / "data.csv"
    write_dataset(dataset_path)

    pattern_search.main(
        [
            "--dataset-path",
            str(dataset_path),
            "--target-col",
            "target",
            "--time-col",
            "date",
            "--train-ratio",
            "0.6",
            "--val-ratio",
            "0.2",
            "--pred-len",
            "4",
            "--top-k",
            "2",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert output["query_length"] == 4
    assert output["query_window_start"] == 52
    assert output["query_window_end"] == 56
    assert output["search_length"] == 42
    assert output["search_window_end"] == 42
    assert all(match["end_index"] <= 42 for match in output["matches"])


def test_pattern_search_uses_timestamp_boundaries_from_task_json(
    tmp_path: Path, capsys
) -> None:
    dataset_path = tmp_path / "data.csv"
    write_dataset(dataset_path)
    frame = pd.read_csv(dataset_path)
    task_path = tmp_path / ".forecast" / "task.json"
    task_path.parent.mkdir()
    task_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "target_col": "target",
                "time_col": "date",
                "pred_len": 4,
                "split_mode": "timestamp",
                "train_end_timestamp": frame["date"].iloc[41],
                "validation_end_timestamp": frame["date"].iloc[55],
                "train_ratio": 42 / 70,
                "val_ratio": 14 / 70,
                "test_ratio": 14 / 70,
            }
        ),
        encoding="utf-8",
    )

    pattern_search.main(["--task-json", str(task_path), "--top-k", "2"])
    output = json.loads(capsys.readouterr().out)

    assert output["query_window_start"] == 52
    assert output["query_window_end"] == 56
    assert output["search_window_end"] == 42


def test_pattern_search_uses_ratio_boundaries_from_task_json(
    tmp_path: Path, capsys
) -> None:
    dataset_path = tmp_path / "data.csv"
    write_dataset(dataset_path)
    task_path = tmp_path / ".forecast" / "task.json"
    task_path.parent.mkdir()
    task_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "target_col": "target",
                "time_col": "date",
                "pred_len": 4,
                "split_mode": "ratio",
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
            }
        ),
        encoding="utf-8",
    )

    pattern_search.main(["--task-json", str(task_path), "--top-k", "2"])
    output = json.loads(capsys.readouterr().out)

    assert output["query_window_start"] == 52
    assert output["query_window_end"] == 56
    assert output["search_window_end"] == 42


def test_pattern_search_rejects_partial_boundary_configuration(
    tmp_path: Path, capsys
) -> None:
    dataset_path = tmp_path / "data.csv"
    write_dataset(dataset_path)

    with np.testing.assert_raises(SystemExit):
        pattern_search.main(
            [
                "--dataset-path",
                str(dataset_path),
                "--target-col",
                "target",
                "--time-col",
                "date",
                "--train-ratio",
                "0.6",
                "--top-k",
                "2",
            ]
        )
    output = json.loads(capsys.readouterr().out)
    assert "Provide either" in output["error"]


def test_window_visualization_aggregates_overlapping_timestamps(
    tmp_path: Path, capsys
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:15"],
            "actual": [1.0, 2.0, 2.0],
            "predicted": [0.8, 1.8, 2.0],
        }
    ).to_csv(run_dir / "predictions.csv", index=False)
    np.save(run_dir / "actual.npy", np.asarray([1.0, 2.0, 2.0]))
    output_path = tmp_path / "window.png"

    window_viz.main(
        [
            "--best-run-dir",
            str(run_dir),
            "--actual-npy",
            str(run_dir / "actual.npy"),
            "--out",
            str(output_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert output["raw_points"] == 3
    assert output["time_steps"] == 2
    assert output["aggregation"] == "mean_prediction_by_timestamp"
    assert output_path.exists()
