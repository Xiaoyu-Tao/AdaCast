"""Import user-submitted prediction priors as AdaCast forecast artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from adacast_lite_ml.engine.data_loader import (
    numeric_feature_columns,
    parse_forecast_gap,
    resolve_training_boundaries,
)
from adacast_lite_ml.engine.evaluator import evaluate
from adacast_lite_ml.engine.utils.hash import (
    compute_file_hash,
    compute_spec_hash,
)

MODEL_NAME = "UserPrior"
MODEL_FAMILY = "user_prior"
PREDICTION_LAYOUT = "sliding_window"
PREDICTION_ARTIFACT_FILES = {
    "predictions_sha256": "predictions.csv",
    "actual_sha256": "actual.npy",
    "pred_sha256": "pred.npy",
    "train_scale_sha256": "train_scale.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import AdaCast user prediction prior")
    parser.add_argument("--action", choices=["import", "materialize-test"], required=True)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--prior-dataset-path", required=True)
    parser.add_argument("--forecast-root", required=True)
    parser.add_argument("--evaluator-hash-store", required=True)
    parser.add_argument("--source-validation-run-id")
    parser.add_argument("--expected-prior-sha256")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def project_relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def resolve_dataset_path(task: dict[str, Any], task_path: Path) -> Path:
    dataset_path = Path(str(task["dataset_path"]))
    if dataset_path.is_absolute():
        return dataset_path
    return (task_path.parent.parent / dataset_path).resolve()


def timestamp_key(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.notna().all():
        return parsed.astype("int64").astype(str)
    return values.astype(str).str.strip()


def assert_same_columns(original: pd.DataFrame, prior: pd.DataFrame) -> None:
    if list(original.columns) != list(prior.columns):
        raise ValueError(
            "Prediction prior CSV must keep exactly the same column names and order as the real dataset"
        )


def assert_same_timestamps(
    original: pd.DataFrame,
    prior: pd.DataFrame,
    time_col: str,
) -> None:
    if time_col not in prior.columns:
        raise ValueError(f"Prediction prior CSV is missing time column {time_col!r}")
    if not timestamp_key(original[time_col]).equals(timestamp_key(prior[time_col])):
        raise ValueError(
            "Prediction prior CSV timestamps must exactly align with the real dataset"
        )


def series_equal(left: pd.Series, right: pd.Series) -> bool:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    if left_numeric.notna().all() and right_numeric.notna().all():
        return bool(
            np.allclose(
                left_numeric.to_numpy(dtype=np.float64),
                right_numeric.to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-9,
                equal_nan=True,
            )
        )
    return left.astype(str).fillna("").equals(right.astype(str).fillna(""))


def validate_prior_frame(
    *,
    task: dict[str, Any],
    original: pd.DataFrame,
    prior: pd.DataFrame,
    train_end: int,
) -> None:
    time_col = str(task["time_col"])
    target_col = str(task["target_col"])
    if len(original) != len(prior):
        raise ValueError(
            "Prediction prior CSV must have the same row count as the real dataset"
        )
    assert_same_columns(original, prior)
    assert_same_timestamps(original, prior, time_col)
    if target_col not in prior.columns:
        raise ValueError(f"Prediction prior CSV is missing target column {target_col!r}")

    for column in original.columns:
        if column == target_col:
            if not series_equal(original[column].iloc[:train_end], prior[column].iloc[:train_end]):
                raise ValueError(
                    "Prediction prior CSV training target values must remain identical to the real dataset"
                )
            continue
        if not series_equal(original[column], prior[column]):
            raise ValueError(
                f"Prediction prior CSV may only replace validation/test target values; column {column!r} differs"
            )

    target_values = pd.to_numeric(prior[target_col], errors="coerce")
    if not np.isfinite(target_values.to_numpy(dtype=np.float64)).all():
        raise ValueError("Prediction prior target column must contain finite numeric values")


def target_index(task: dict[str, Any], columns: list[str]) -> int:
    target_col = str(task["target_col"])
    if target_col not in columns:
        raise ValueError(f"Target column {target_col!r} not found in dataset")
    return columns.index(target_col)


def scaled_matrices(
    original: pd.DataFrame,
    prior: pd.DataFrame,
    task: dict[str, Any],
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    time_col = str(task["time_col"])
    columns = numeric_feature_columns(original, time_col, str(task["target_col"]))
    target_pos = target_index(task, columns)
    original_features = original[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    prior_features = prior[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(original_features[:train_end])
    return (
        scaler.transform(original_features),
        scaler.transform(prior_features),
        columns,
        target_pos,
    )


def infer_prediction_target_anchor(
    *,
    original: pd.DataFrame,
    prior: pd.DataFrame,
    target_col: str,
    train_end: int,
    search_start: int | None = None,
    search_end: int | None = None,
    target_start_timestamp: str | None = None,
    time_col: str | None = None,
) -> int:
    """Recover the first target position represented by a submitted-prior split grid."""
    actual = pd.to_numeric(original[target_col], errors="raise").to_numpy(dtype=np.float64)
    predicted = pd.to_numeric(prior[target_col], errors="raise").to_numpy(dtype=np.float64)
    changed = ~np.isclose(actual, predicted, rtol=0.0, atol=1e-9, equal_nan=True)
    if target_start_timestamp is not None:
        if not time_col or time_col not in original.columns:
            raise ValueError("An explicit prior target timestamp requires time_col")
        timestamps = pd.to_datetime(original[time_col], errors="raise", utc=True, format="mixed")
        requested = pd.to_datetime(target_start_timestamp, errors="raise", utc=True)
        matches = np.flatnonzero(timestamps.eq(requested).to_numpy())
        if len(matches) != 1 or int(matches[0]) < train_end:
            raise ValueError("Explicit prior target anchor must identify one post-training timestamp")
        return int(matches[0])
    start = train_end if search_start is None else search_start
    end = len(changed) if search_end is None else search_end
    if start < train_end or end < start or end > len(changed):
        raise ValueError(
            "Prediction prior target-anchor search bounds must stay within the post-training dataset"
        )
    changed[:start] = False
    changed[end:] = False
    changed_indices = np.flatnonzero(changed)
    if len(changed_indices) == 0:
        raise ValueError(
            "Prediction prior target column must contain at least one value that differs "
            f"from the real dataset in rows [{start}, {end})"
        )
    return int(changed_indices[0])


def prediction_window_starts(
    split: str,
    *,
    train_end: int,
    adjustment_fit_end: int,
    val_end: int,
    row_count: int,
    seq_len: int,
    forecast_gap: int,
    pred_len: int,
    window_stride: int,
    prediction_target_anchor: int,
) -> list[int]:
    """Return full windows selected from one globally anchored prediction grid.

    The grid is recovered from the submitted prior's first changed target position.
    Validation-front, validation-back, and test only select complete windows from that
    grid; none of the split boundaries starts a new grid. A window crossing a split
    boundary is excluded so evidence cannot be shared between adjacent splits.
    """
    if window_stride <= 0:
        raise ValueError("Prediction prior requires a positive window_stride")
    if split not in {"adjustment_fit", "val", "test"}:
        raise ValueError(f"Unknown split {split!r}")
    anchor = prediction_target_anchor - forecast_gap - seq_len
    if anchor < 0:
        raise ValueError(
            "Prediction prior target anchor is too early for "
            f"seq_len={seq_len} and forecast_gap={forecast_gap}"
        )
    last_start = row_count - seq_len - forecast_gap - pred_len
    all_starts = range(anchor, last_start + 1, window_stride)

    def belongs_to_split(window_start: int) -> bool:
        forecast_origin = window_start + seq_len
        target_end = forecast_origin + forecast_gap + pred_len
        if split == "adjustment_fit":
            return forecast_origin >= train_end and target_end <= adjustment_fit_end
        if split == "val":
            return forecast_origin >= adjustment_fit_end and target_end <= val_end
        if split == "test":
            return forecast_origin >= val_end and target_end <= row_count
        return False

    starts = [window_start for window_start in all_starts if belongs_to_split(window_start)]
    if not starts:
        raise ValueError(
            f"Prediction prior split {split!r} has no complete window on the global "
            f"window_stride={window_stride} grid"
        )
    return starts


def build_spec(
    task: dict[str, Any],
    eval_split: str,
    prediction_target_anchor: int,
) -> dict[str, Any]:
    phase = "post-forecast" if eval_split == "test" else "forecast"
    return {
        **task,
        "model": MODEL_NAME,
        "phase": phase,
        "eval_split": eval_split,
        "input_contract_version": 5,
        "source_type": "user_prediction_prior",
        "prediction_grid_target_anchor_index": prediction_target_anchor,
        "metrics": task.get("eval_metrics", task.get("metrics", ["mse", "mae", "wape", "mase"])),
    }


def validate_prediction_rows(
    *,
    rows: list[dict[str, float | int | str]],
    timestamps: np.ndarray,
    split: str,
    window_start_indices: list[int],
    seq_len: int,
    forecast_gap: int,
    pred_len: int,
    window_stride: int,
) -> None:
    window_count = len(window_start_indices)
    expected_count = window_count * pred_len
    if len(rows) != expected_count:
        raise ValueError(
            f"Prediction prior output contract mismatch for split {split!r}: "
            f"expected {expected_count} rows, got {len(rows)}"
        )
    if window_stride <= 0:
        raise ValueError("Prediction prior output contract requires a positive window_stride")
    if window_count <= 0:
        raise ValueError("Prediction prior output contract requires at least one window")
    for index, window_start in enumerate(window_start_indices):
        if window_start < 0:
            raise ValueError(
                f"Prediction prior output contract mismatch for split {split!r}: "
                f"window {index} has negative start index {window_start}"
            )
        if index > 0:
            delta = window_start - window_start_indices[index - 1]
            if delta <= 0 or delta % window_stride != 0:
                raise ValueError(
                    f"Prediction prior output contract mismatch for split {split!r}: "
                    "window starts do not share one global stride grid"
                )

    last_target_end = window_start_indices[-1] + seq_len + forecast_gap + pred_len
    if last_target_end > len(timestamps):
        raise ValueError(
            f"Prediction prior output contract mismatch for split {split!r}: "
            "target window exceeds the available split timestamps"
        )

    def require_int(row: dict[str, float | int | str], key: str, row_offset: int) -> int:
        try:
            return int(row[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Prediction prior output contract mismatch for split {split!r}: "
                f"row {row_offset} has invalid {key}={row.get(key)!r}"
            ) from exc

    for window_index in range(window_count):
        target_begin = window_start_indices[window_index] + seq_len + forecast_gap
        for horizon_index in range(pred_len):
            row_offset = window_index * pred_len + horizon_index
            row = rows[row_offset]
            if row.get("prediction_layout") != PREDICTION_LAYOUT:
                raise ValueError(
                    f"Prediction prior output contract mismatch for split {split!r}: "
                    f"expected prediction_layout={PREDICTION_LAYOUT!r}, got {row.get('prediction_layout')!r}"
                )
            if require_int(row, "window_index", row_offset) != window_index:
                raise ValueError(
                    f"Prediction prior output contract mismatch for split {split!r}: "
                    f"row {row_offset} has window_index={row.get('window_index')!r}, expected {window_index}"
                )
            if require_int(row, "horizon_index", row_offset) != horizon_index:
                raise ValueError(
                    f"Prediction prior output contract mismatch for split {split!r}: "
                    f"row {row_offset} has horizon_index={row.get('horizon_index')!r}, expected {horizon_index}"
                )
            expected_timestamp = str(timestamps[target_begin + horizon_index])
            if str(row.get("timestamp")) != expected_timestamp:
                raise ValueError(
                    f"Prediction prior output contract mismatch for split {split!r}: "
                    f"row {row_offset} has timestamp={row.get('timestamp')!r}, "
                    f"expected {expected_timestamp!r}"
                )


def compute_prediction_artifact_hashes(run_dir: Path) -> dict[str, str]:
    return {
        key: compute_file_hash(run_dir / filename)
        for key, filename in PREDICTION_ARTIFACT_FILES.items()
    }


def build_eval_artifact_binding(
    *,
    model_artifact_sha256: str,
    model_source_run_id: str,
    prediction_artifacts: dict[str, str],
) -> dict[str, object]:
    return {
        "model_artifact": {
            "kind": "external_prediction_prior",
            "sha256": model_artifact_sha256,
            "source_run_id": model_source_run_id,
            "foundation_revision": None,
        },
        "prediction_artifacts": prediction_artifacts,
    }


def write_prior_source(
    *,
    run_dir: Path,
    prior_dataset_path: Path,
    prior_dataset_sha256: str,
    task_hash: str,
    prediction_target_anchor: int,
) -> tuple[Path, str]:
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    prior_source_path = model_dir / "prior-source.json"
    prior_source = {
        "kind": "external_prediction_prior",
        "model": MODEL_NAME,
        "format": "adacast_user_prediction_prior",
        "source_dataset_path": str(prior_dataset_path),
        "source_dataset_sha256": prior_dataset_sha256,
        "prediction_grid_target_anchor_index": prediction_target_anchor,
        "task_hash": task_hash,
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    prior_source_path.write_text(
        json.dumps(prior_source, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return prior_source_path, compute_file_hash(prior_source_path)


def write_reuse(
    *,
    run_dir: Path,
    source_run_id: str,
    source_artifact_sha256: str,
    evaluated_split: str,
) -> None:
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "reuse.json").write_text(
        json.dumps(
            {
                "kind": "model_reuse",
                "model": MODEL_NAME,
                "source_run_id": source_run_id,
                "source_artifact_kind": "external_prediction_prior",
                "source_artifact_sha256": source_artifact_sha256,
                "foundation_revision": None,
                "training_performed": False,
                "evaluated_split": evaluated_split,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_prediction_run(
    *,
    task: dict[str, Any],
    forecast_root: Path,
    original: pd.DataFrame,
    original_scaled: np.ndarray,
    prior_scaled: np.ndarray,
    target_pos: int,
    train_values: np.ndarray,
    eval_split: str,
    prior_dataset_path: Path,
    prior_dataset_sha256: str,
    source_validation_run_id: str | None,
    source_artifact_sha256: str | None,
    evaluator_hash_store: Path,
    prediction_target_anchor: int,
) -> dict[str, Any]:
    seq_len = int(task["seq_len"])
    pred_len = int(task["pred_len"])
    forecast_gap = parse_forecast_gap(task)
    window_stride = int(task.get("window_stride", 1))
    train_end, adjustment_fit_end, val_end = resolve_training_boundaries(original, task)
    if eval_split == "adjustment_fit":
        if adjustment_fit_end - train_end < forecast_gap + pred_len:
            required_window = (
                "pred_len"
                if forecast_gap == 0
                else f"forecast_gap={forecast_gap} plus pred_len={pred_len}"
            )
            raise ValueError(
                f"The validation-front evidence split is shorter than {required_window}"
            )
    if eval_split == "val":
        if val_end - adjustment_fit_end < forecast_gap + pred_len:
            required_window = (
                "pred_len"
                if forecast_gap == 0
                else f"forecast_gap={forecast_gap} plus pred_len={pred_len}"
            )
            raise ValueError(
                f"The held-out validation split is shorter than {required_window}"
            )
    window_start_indices = prediction_window_starts(
        eval_split,
        train_end=train_end,
        adjustment_fit_end=adjustment_fit_end,
        val_end=val_end,
        row_count=len(original),
        seq_len=seq_len,
        forecast_gap=forecast_gap,
        pred_len=pred_len,
        window_stride=window_stride,
        prediction_target_anchor=prediction_target_anchor,
    )
    timestamps = original[str(task["time_col"])].to_numpy()
    window_count = len(window_start_indices)

    spec = build_spec(task, eval_split, prediction_target_anchor)
    run_id = compute_spec_hash(spec)
    run_dir = (
        forecast_root / "adjustments" / "fit-runs" / "runs" / run_id
        if eval_split == "adjustment_fit"
        else forecast_root / "runs" / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    rows: list[dict[str, float | int | str]] = []
    gap_rows: list[dict[str, float | int | str]] = []
    actual_values: list[float] = []
    predicted_values: list[float] = []
    for window_index, s_begin in enumerate(window_start_indices):
        s_end = s_begin + seq_len
        target_begin = s_end + forecast_gap
        target_end = target_begin + pred_len
        if eval_split == "test" and forecast_gap > 0:
            for gap_index, row_index in enumerate(range(s_end, target_begin)):
                gap_rows.append(
                    {
                        "timestamp": str(timestamps[row_index]),
                        "predicted": float(prior_scaled[row_index, target_pos]),
                        "prediction_layout": "forecast_gap",
                        "window_index": window_index,
                        "gap_index": gap_index,
                    }
                )
        for horizon_index, row_index in enumerate(range(target_begin, target_end)):
            actual_value = float(original_scaled[row_index, target_pos])
            predicted_value = float(prior_scaled[row_index, target_pos])
            actual_values.append(actual_value)
            predicted_values.append(predicted_value)
            rows.append(
                {
                    "timestamp": str(timestamps[row_index]),
                    "actual": actual_value,
                    "predicted": predicted_value,
                    "prediction_layout": PREDICTION_LAYOUT,
                    "window_index": window_index,
                    "horizon_index": horizon_index,
                }
            )

    validate_prediction_rows(
        rows=rows,
        timestamps=timestamps,
        split=eval_split,
        window_start_indices=window_start_indices,
        seq_len=seq_len,
        forecast_gap=forecast_gap,
        pred_len=pred_len,
        window_stride=window_stride,
    )

    predictions_path = run_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    np.save(run_dir / "actual.npy", np.asarray(actual_values, dtype=np.float64), allow_pickle=False)
    if eval_split == "test" and forecast_gap > 0:
        pd.DataFrame(gap_rows).to_csv(run_dir / "gap_predictions.csv", index=False)
    np.save(run_dir / "pred.npy", np.asarray(predicted_values, dtype=np.float64), allow_pickle=False)
    train_scale = float(np.mean(np.abs(np.diff(train_values)))) if len(train_values) > 1 else 0.0
    (run_dir / "train_scale.json").write_text(
        json.dumps({"train_scale": train_scale}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text(
        "\n".join(
            [
                "user_prediction_prior=true",
                f"eval_split={eval_split}",
                f"prediction_grid_target_anchor_index={prediction_target_anchor}",
                "training_performed=false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    task_hash = compute_spec_hash(task)
    if source_validation_run_id is None:
        _, model_artifact_sha256 = write_prior_source(
            run_dir=run_dir,
            prior_dataset_path=prior_dataset_path,
            prior_dataset_sha256=prior_dataset_sha256,
            task_hash=task_hash,
            prediction_target_anchor=prediction_target_anchor,
        )
        model_source_run_id = run_id
    else:
        if source_artifact_sha256 is None:
            raise ValueError("source_artifact_sha256 is required when reusing a validation prior source")
        model_artifact_sha256 = source_artifact_sha256
        model_source_run_id = source_validation_run_id
        write_reuse(
            run_dir=run_dir,
            source_run_id=source_validation_run_id,
            source_artifact_sha256=source_artifact_sha256,
            evaluated_split=eval_split,
        )

    prediction_hashes = compute_prediction_artifact_hashes(run_dir)
    eval_result = evaluate(
        run_dir=run_dir,
        spec=spec,
        evaluator_hash_store=evaluator_hash_store,
        predictions_path=predictions_path,
        train_values=train_values,
    )
    if compute_prediction_artifact_hashes(run_dir) != prediction_hashes:
        raise ValueError("Prediction prior artifacts changed during evaluation")
    eval_result = {
        **eval_result,
        "run_id": run_id,
        "spec_hash": run_id,
        "artifact_binding": build_eval_artifact_binding(
            model_artifact_sha256=model_artifact_sha256,
            model_source_run_id=model_source_run_id,
            prediction_artifacts=prediction_hashes,
        ),
    }
    (run_dir / "eval.json").write_text(
        json.dumps(eval_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    project_root = forecast_root.parent
    return {
        "run_id": run_id,
        "spec_hash": run_id,
        "run_path": project_relative(run_dir, project_root),
        "spec_path": project_relative(run_dir / "spec.json", project_root),
        "predictions_path": project_relative(predictions_path, project_root),
        "eval_path": project_relative(run_dir / "eval.json", project_root),
        "metrics": eval_result["metrics"],
        "model_artifact_sha256": model_artifact_sha256,
    }


def write_source_manifests(
    *,
    forecast_root: Path,
    prior_dataset_path: Path,
    prior_dataset_sha256: str,
    validation: dict[str, Any],
    fit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    source_member = {
        "family": MODEL_FAMILY,
        "validation_run_id": validation["run_id"],
        "model": MODEL_NAME,
        "spec_hash": validation["spec_hash"],
        "validation_metrics": validation["metrics"],
        "artifacts": {
            "spec_path": validation["spec_path"],
            "eval_path": validation["eval_path"],
            "predictions_path": validation["predictions_path"],
        },
    }
    source_set_id = compute_spec_hash(
        {
            "created_from": "user_prediction_prior",
            "prior_dataset_sha256": prior_dataset_sha256,
            "members": [source_member],
        }
    )
    source_set = {
        "source_model_set_id": source_set_id,
        "created_at": now,
        "created_from": "user_prediction_prior",
        "members": [source_member],
    }
    fit_member = {
        "family": MODEL_FAMILY,
        "validation_run_id": validation["run_id"],
        "model": MODEL_NAME,
        "validation_spec_hash": validation["spec_hash"],
        "fit_spec_hash": fit["spec_hash"],
        "fit_run_id": fit["run_id"],
        "status": "success",
        "fit_method": "user_prediction_prior_validation_front_half",
        "spec_path": fit["spec_path"],
        "fit_run_path": fit["run_path"],
        "predictions_path": fit["predictions_path"],
        "eval_path": fit["eval_path"],
        "metrics": fit["metrics"],
        "error": None,
    }
    fit_manifest = {
        "source_model_set_id": source_set_id,
        "created_at": now,
        "updated_at": now,
        "status": "complete",
        "members": [fit_member],
    }
    user_prior_manifest = {
        "status": "active",
        "created_at": now,
        "prior_dataset_path": str(prior_dataset_path),
        "prior_dataset_sha256": prior_dataset_sha256,
        "model": MODEL_NAME,
        "family": MODEL_FAMILY,
        "source_model_set_id": source_set_id,
        "validation_run_id": validation["run_id"],
        "fit_run_id": fit["run_id"],
        "test_run_id": None,
        "validation_metrics": validation["metrics"],
        "fit_metrics": fit["metrics"],
    }

    (forecast_root / "adjustments").mkdir(parents=True, exist_ok=True)
    (forecast_root / "adjustments" / "source-model-set.json").write_text(
        json.dumps(source_set, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (forecast_root / "adjustments" / "fit-runs").mkdir(parents=True, exist_ok=True)
    (forecast_root / "adjustments" / "fit-runs" / "manifest.json").write_text(
        json.dumps(fit_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (forecast_root / "user-prior").mkdir(parents=True, exist_ok=True)
    (forecast_root / "user-prior" / "manifest.json").write_text(
        json.dumps(user_prior_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return source_set, fit_manifest, user_prior_manifest


def prepare_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path, pd.DataFrame, pd.DataFrame, str]:
    task_path = Path(args.task_json).resolve()
    task = load_json(task_path)
    real_dataset_path = resolve_dataset_path(task, task_path)
    prior_dataset_path = Path(args.prior_dataset_path).resolve()
    if not prior_dataset_path.exists():
        raise ValueError(f"Prediction prior CSV does not exist: {prior_dataset_path}")
    forecast_root = Path(args.forecast_root).resolve()
    original = pd.read_csv(real_dataset_path, low_memory=False)
    prior = pd.read_csv(prior_dataset_path, low_memory=False)
    train_end, _, _ = resolve_training_boundaries(original, task)
    validate_prior_frame(task=task, original=original, prior=prior, train_end=train_end)
    prior_sha256 = compute_file_hash(prior_dataset_path)
    expected_sha = args.expected_prior_sha256
    if expected_sha and expected_sha != prior_sha256:
        raise ValueError(
            "Prediction prior CSV has changed since it was imported; refuse to materialize test artifacts"
        )
    return task, task_path, forecast_root, prior_dataset_path, original, prior, prior_sha256


def run_import(args: argparse.Namespace) -> dict[str, Any]:
    task, _task_path, forecast_root, prior_dataset_path, original, prior, prior_sha256 = prepare_inputs(args)
    train_end, _, _ = resolve_training_boundaries(original, task)
    prediction_target_anchor = infer_prediction_target_anchor(
        original=original,
        prior=prior,
        target_col=str(task["target_col"]),
        train_end=train_end,
        target_start_timestamp=task.get("prediction_target_start_timestamp"),
        time_col=str(task["time_col"]),
    )
    original_scaled, prior_scaled, _, target_pos = scaled_matrices(original, prior, task, train_end)
    train_values = original_scaled[:train_end, target_pos].astype(np.float64)
    evaluator_hash_store = Path(args.evaluator_hash_store).resolve()

    validation = write_prediction_run(
        task=task,
        forecast_root=forecast_root,
        original=original,
        original_scaled=original_scaled,
        prior_scaled=prior_scaled,
        target_pos=target_pos,
        train_values=train_values,
        eval_split="val",
        prior_dataset_path=prior_dataset_path,
        prior_dataset_sha256=prior_sha256,
        source_validation_run_id=None,
        source_artifact_sha256=None,
        evaluator_hash_store=evaluator_hash_store,
        prediction_target_anchor=prediction_target_anchor,
    )
    fit = write_prediction_run(
        task=task,
        forecast_root=forecast_root,
        original=original,
        original_scaled=original_scaled,
        prior_scaled=prior_scaled,
        target_pos=target_pos,
        train_values=train_values,
        eval_split="adjustment_fit",
        prior_dataset_path=prior_dataset_path,
        prior_dataset_sha256=prior_sha256,
        source_validation_run_id=validation["run_id"],
        source_artifact_sha256=validation["model_artifact_sha256"],
        evaluator_hash_store=evaluator_hash_store,
        prediction_target_anchor=prediction_target_anchor,
    )
    source_set, fit_manifest, user_prior_manifest = write_source_manifests(
        forecast_root=forecast_root,
        prior_dataset_path=prior_dataset_path,
        prior_dataset_sha256=prior_sha256,
        validation=validation,
        fit=fit,
    )
    return {
        "status": "success",
        "action": "import",
        "prior_dataset_path": str(prior_dataset_path),
        "prior_dataset_sha256": prior_sha256,
        "source_model_set": source_set,
        "fit_run_manifest": fit_manifest,
        "user_prior_manifest": user_prior_manifest,
        "validation_run": validation,
        "fit_run": fit,
    }


def run_materialize_test(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source_validation_run_id:
        raise ValueError("--source-validation-run-id is required for materialize-test")
    task, _task_path, forecast_root, prior_dataset_path, original, prior, prior_sha256 = prepare_inputs(args)
    train_end, _, val_end = resolve_training_boundaries(original, task)
    test_prediction_target_anchor = infer_prediction_target_anchor(
        original=original,
        prior=prior,
        target_col=str(task["target_col"]),
        train_end=train_end,
        search_start=val_end,
        target_start_timestamp=task.get("prediction_target_start_timestamp"),
        time_col=str(task["time_col"]),
    )
    original_scaled, prior_scaled, _, target_pos = scaled_matrices(original, prior, task, train_end)
    train_values = original_scaled[:train_end, target_pos].astype(np.float64)
    validation_source_path = forecast_root / "runs" / args.source_validation_run_id / "model" / "prior-source.json"
    if not validation_source_path.exists():
        raise ValueError(
            f"Imported validation prior source is missing: {validation_source_path}"
        )
    validation_source = load_json(validation_source_path)
    validation_prediction_target_anchor = validation_source.get(
        "prediction_grid_target_anchor_index"
    )
    if not isinstance(validation_prediction_target_anchor, int):
        raise ValueError(
            "Imported validation prior source is missing its prediction-grid target anchor"
        )
    current_task_hash = compute_spec_hash(task)
    if validation_source.get("task_hash") != current_task_hash:
        raise ValueError(
            "Frozen task contract changed after prediction-prior import; "
            "refusing to materialize test predictions from a different task"
        )
    validation_spec_path = (
        forecast_root / "runs" / args.source_validation_run_id / "spec.json"
    )
    if not validation_spec_path.exists():
        raise ValueError(
            f"Imported validation execution contract is missing: {validation_spec_path}"
        )
    if compute_spec_hash(load_json(validation_spec_path)) != compute_spec_hash(
        build_spec(task, "val", validation_prediction_target_anchor)
    ):
        raise ValueError(
            "Imported validation execution contract does not match the frozen task; "
            "refusing to materialize test predictions"
        )
    source_artifact_sha256 = compute_file_hash(validation_source_path)
    test = write_prediction_run(
        task=task,
        forecast_root=forecast_root,
        original=original,
        original_scaled=original_scaled,
        prior_scaled=prior_scaled,
        target_pos=target_pos,
        train_values=train_values,
        eval_split="test",
        prior_dataset_path=prior_dataset_path,
        prior_dataset_sha256=prior_sha256,
        source_validation_run_id=args.source_validation_run_id,
        source_artifact_sha256=source_artifact_sha256,
        evaluator_hash_store=Path(args.evaluator_hash_store).resolve(),
        prediction_target_anchor=test_prediction_target_anchor,
    )

    manifest_path = forecast_root / "user-prior" / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    manifest = {
        **manifest,
        "status": "test_materialized",
        "test_run_id": test["run_id"],
        "test_metrics": test["metrics"],
        "test_materialized_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "action": "materialize-test",
        "prior_dataset_path": str(prior_dataset_path),
        "prior_dataset_sha256": prior_sha256,
        "test_run": test,
        "user_prior_manifest": manifest,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_import(args) if args.action == "import" else run_materialize_test(args)
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        json.dump(
            {
                "status": "error",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
