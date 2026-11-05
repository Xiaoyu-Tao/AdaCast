"""Apply deterministic adjustment rules without exposing horizon actuals."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adacast_lite_ml.engine.adjustment.prediction_layout import prediction_coordinates

from adacast_lite_ml.engine.data_loader import resolve_training_boundaries


FORBIDDEN_BLIND_KEYS = {"actual", "target", "residual", "error", "metric", "mse", "mae", "wape", "mase"}
FORBIDDEN_BLIND_PREFIXES = tuple(f"{key}_" for key in FORBIDDEN_BLIND_KEYS)
EXOG_PREFIX = "exog__"
EXOG_TEXT_PREFIX = "exog_text__"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a AdaCast adjustment candidate")
    parser.add_argument("--task-json", required=True, help="Path to .forecast/task.json")
    parser.add_argument("--predictions-csv", required=True, help="Raw predictions.csv path")
    parser.add_argument("--candidate-json", required=True, help="Adjustment candidate.json path")
    parser.add_argument("--out-dir", required=True, help="Directory for adjusted artifacts")
    parser.add_argument("--source-run-id", required=True, help="Raw source run id")
    parser.add_argument("--split", choices=["fit", "val", "test"], required=True)
    parser.add_argument("--policy-json", help="Optional adjustment policy.json path")
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_dataset_path(task: dict[str, Any], task_path: Path) -> Path:
    dataset_path = Path(str(task["dataset_path"]))
    if dataset_path.is_absolute():
        return dataset_path
    return (task_path.parent.parent / dataset_path).resolve()


def target_scale(task: dict[str, Any], task_path: Path) -> tuple[float, float]:
    dataset_path = resolve_dataset_path(task, task_path)
    df = pd.read_csv(dataset_path, low_memory=False)
    feature_cols = [column for column in df.columns if column != task["time_col"]]
    if task["target_col"] not in feature_cols:
        raise ValueError(f"Target column {task['target_col']!r} not found in dataset")
    target = pd.to_numeric(df[task["target_col"]], errors="coerce").to_numpy(dtype=np.float64)
    model_train_end, _, _ = resolve_training_boundaries(df, task)
    train_target = target[:model_train_end]
    mean = float(np.nanmean(train_target))
    scale = float(np.nanstd(train_target))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return mean, scale


def scaled_to_original(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return values * scale + mean


def original_to_scaled(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return (values - mean) / scale


def row_hour(timestamp: Any) -> int | None:
    parsed = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.hour)


def row_month(timestamp: Any) -> int | None:
    parsed = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.month)


def timestamp_key(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().any():
        return parsed.astype("int64").astype(str)
    return values.astype(str)


def safe_exogenous_columns(task: dict[str, Any], dataset: pd.DataFrame) -> list[str]:
    time_col = str(task["time_col"])
    target_col = str(task["target_col"])
    blocked = {time_col, target_col}
    result = []
    for column in dataset.columns:
        if task.get("exogenous_columns") is not None and column not in task["exogenous_columns"]:
            continue
        lower = str(column).lower()
        if column in blocked:
            continue
        if lower in FORBIDDEN_BLIND_KEYS or lower.startswith(FORBIDDEN_BLIND_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            result.append(str(column))
    return result


def safe_text_exogenous_columns(task: dict[str, Any], dataset: pd.DataFrame) -> list[str]:
    time_col = str(task["time_col"])
    target_col = str(task["target_col"])
    blocked = {time_col, target_col}
    result = []
    for column in dataset.columns:
        if task.get("exogenous_columns") is not None and column not in task["exogenous_columns"]:
            continue
        lower = str(column).lower()
        if column in blocked:
            continue
        if lower in FORBIDDEN_BLIND_KEYS or lower.startswith(FORBIDDEN_BLIND_PREFIXES):
            continue
        if pd.api.types.is_object_dtype(dataset[column]) or pd.api.types.is_bool_dtype(dataset[column]):
            result.append(str(column))
    return result


def day_offset_tag(day_offset: int) -> str:
    if day_offset == 0:
        return "current"
    return f"p{day_offset}" if day_offset > 0 else f"m{abs(day_offset)}"


def exog_numeric_column(column: str, day_offset: int = 0) -> str:
    if day_offset == 0:
        return f"{EXOG_PREFIX}{column}"
    return f"{EXOG_PREFIX}day_{day_offset_tag(day_offset)}__{column}"


def exog_text_column(column: str, day_offset: int = 0) -> str:
    if day_offset == 0:
        return f"{EXOG_TEXT_PREFIX}{column}"
    return f"{EXOG_TEXT_PREFIX}day_{day_offset_tag(day_offset)}__{column}"


def timestamp_date_key(values: pd.Series, day_offset: int = 0) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().any():
        shifted = parsed + timedelta(days=day_offset)
        return shifted.dt.normalize().astype("int64").astype(str)
    return values.astype(str)


def condition_day_offsets(condition: dict[str, Any]) -> set[int]:
    kind = condition.get("kind")
    if kind in {
        "exogenous_numeric_between",
        "exogenous_text_in",
        "exogenous_text_contains",
        "exogenous_bool_is",
    }:
        return {int(condition.get("day_offset", 0) or 0)}
    if kind in {"all", "any"}:
        offsets: set[int] = set()
        for child in condition.get("conditions", []):
            offsets.update(condition_day_offsets(child))
        return offsets
    return set()


def candidate_day_offsets(candidate: dict[str, Any]) -> set[int]:
    offsets = {0}
    for rule in candidate.get("rules", []):
        offsets.update(condition_day_offsets(rule.get("condition", {})))
    return offsets


def attach_exogenous_features(
    task: dict[str, Any],
    task_path: Path,
    rows: pd.DataFrame,
    *,
    day_offsets: Iterable[int] = (0,),
) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(task, task_path)
    dataset = pd.read_csv(dataset_path, low_memory=False)
    time_col = str(task["time_col"])
    if time_col not in dataset.columns:
        return rows
    exogenous_columns = safe_exogenous_columns(task, dataset)
    text_columns = safe_text_exogenous_columns(task, dataset)
    if not exogenous_columns and not text_columns:
        return rows

    result = rows.copy()
    if exogenous_columns:
        feature_df = dataset[[time_col, *exogenous_columns]].copy()
        feature_df["__timestamp_key"] = timestamp_key(feature_df[time_col])
        feature_df = feature_df.drop(columns=[time_col]).drop_duplicates("__timestamp_key", keep="last")
        feature_df = feature_df.rename(columns={column: exog_numeric_column(column) for column in exogenous_columns})
        result["__timestamp_key"] = timestamp_key(result["timestamp"])
        result = result.merge(feature_df, on="__timestamp_key", how="left")
        result = result.drop(columns=["__timestamp_key"])

    if text_columns:
        feature_df = dataset[[time_col, *text_columns]].copy()
        feature_df["__date_key"] = timestamp_date_key(feature_df[time_col])
        feature_df = feature_df.drop(columns=[time_col]).drop_duplicates("__date_key", keep="last")
        feature_df = feature_df.rename(columns={column: exog_text_column(column) for column in text_columns})
        result["__date_key"] = timestamp_date_key(result["timestamp"])
        result = result.merge(feature_df, on="__date_key", how="left")
        result = result.drop(columns=["__date_key"])

    for day_offset in sorted(set(int(offset) for offset in day_offsets if int(offset) != 0)):
        shifted_columns = list(dict.fromkeys([*exogenous_columns, *text_columns]))
        if not shifted_columns:
            continue
        day_df = dataset[[time_col, *shifted_columns]].copy()
        day_df["__date_key"] = timestamp_date_key(day_df[time_col])
        day_df = day_df.drop(columns=[time_col]).drop_duplicates("__date_key", keep="last")
        renamed: dict[str, str] = {}
        for column in exogenous_columns:
            renamed[column] = exog_numeric_column(column, day_offset)
        for column in text_columns:
            renamed[column] = exog_text_column(column, day_offset)
        day_df = day_df.rename(columns=renamed)
        result["__date_key"] = timestamp_date_key(result["timestamp"], day_offset=day_offset)
        result = result.merge(day_df, on="__date_key", how="left")
        result = result.drop(columns=["__date_key"])
    return result


def parse_hhmm(value: str) -> int:
    try:
        hour_text, minute_text = str(value).split(":", maxsplit=1)
        hour = int(hour_text)
        minute = int(minute_text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid HH:mm time value: {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid HH:mm time value: {value!r}")
    return hour * 60 + minute


def row_minute_of_day(timestamp: Any) -> int | None:
    parsed = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.hour) * 60 + int(parsed.minute)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    return text not in {"", "0", "0.0", "false", "no", "n", "否"}


def condition_mask(condition: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    kind = condition.get("kind")
    if kind == "always":
        return np.ones(len(rows), dtype=bool)
    if kind == "all":
        conditions = condition.get("conditions", [])
        if not conditions:
            return np.zeros(len(rows), dtype=bool)
        mask = np.ones(len(rows), dtype=bool)
        for child in conditions:
            mask &= condition_mask(child, rows)
        return mask
    if kind == "any":
        conditions = condition.get("conditions", [])
        mask = np.zeros(len(rows), dtype=bool)
        for child in conditions:
            mask |= condition_mask(child, rows)
        return mask
    if kind == "horizon_index_in":
        values = {int(value) for value in condition.get("values", [])}
        return rows["horizon_index"].map(lambda value: int(value) in values).to_numpy(dtype=bool)
    if kind == "timestamp_hour_in":
        values = {int(value) for value in condition.get("values", [])}
        return rows["timestamp"].map(lambda value: (row_hour(value) in values)).to_numpy(dtype=bool)
    if kind == "timestamp_month_in":
        values = {int(value) for value in condition.get("values", [])}
        return rows["timestamp"].map(lambda value: (row_month(value) in values)).to_numpy(dtype=bool)
    if kind == "timestamp_time_between":
        start = parse_hhmm(str(condition["start"]))
        end = parse_hhmm(str(condition["end"]))
        minutes = rows["timestamp"].map(row_minute_of_day)
        if start <= end:
            return minutes.map(lambda value: value is not None and start <= value <= end).to_numpy(dtype=bool)
        return minutes.map(lambda value: value is not None and (value >= start or value <= end)).to_numpy(dtype=bool)
    if kind == "raw_predicted_original_between":
        lower = float(condition["min"])
        upper = float(condition["max"])
        if lower > upper:
            raise ValueError("raw_predicted_original_between requires min <= max")
        return (
            (rows["raw_predicted_original"].to_numpy(dtype=np.float64) >= lower)
            & (rows["raw_predicted_original"].to_numpy(dtype=np.float64) <= upper)
        )
    if kind == "exogenous_numeric_between":
        column = str(condition["column"])
        day_offset = int(condition.get("day_offset", 0) or 0)
        lower_column = column.lower()
        if lower_column in FORBIDDEN_BLIND_KEYS or lower_column.startswith(FORBIDDEN_BLIND_PREFIXES):
            raise ValueError("exogenous_numeric_between cannot target actual/residual/metric columns")
        exog_column = exog_numeric_column(column, day_offset)
        if exog_column not in rows.columns:
            raise ValueError(f"exogenous_numeric_between column not available in blind context: {column!r}")
        lower = float(condition["min"])
        upper = float(condition["max"])
        if lower > upper:
            raise ValueError("exogenous_numeric_between requires min <= max")
        values = pd.to_numeric(rows[exog_column], errors="coerce").to_numpy(dtype=np.float64)
        return np.isfinite(values) & (values >= lower) & (values <= upper)
    if kind == "exogenous_text_in":
        column = str(condition["column"])
        day_offset = int(condition.get("day_offset", 0) or 0)
        lower_column = column.lower()
        if lower_column in FORBIDDEN_BLIND_KEYS or lower_column.startswith(FORBIDDEN_BLIND_PREFIXES):
            raise ValueError("exogenous_text_in cannot target actual/residual/metric columns")
        exog_column = exog_text_column(column, day_offset)
        if exog_column not in rows.columns:
            raise ValueError(f"exogenous_text_in column not available in blind context: {column!r}")
        values = {clean_text(value) for value in condition.get("values", [])}
        return rows[exog_column].map(lambda value: clean_text(value) in values).to_numpy(dtype=bool)
    if kind == "exogenous_text_contains":
        column = str(condition["column"])
        day_offset = int(condition.get("day_offset", 0) or 0)
        lower_column = column.lower()
        if lower_column in FORBIDDEN_BLIND_KEYS or lower_column.startswith(FORBIDDEN_BLIND_PREFIXES):
            raise ValueError("exogenous_text_contains cannot target actual/residual/metric columns")
        exog_column = exog_text_column(column, day_offset)
        if exog_column not in rows.columns:
            raise ValueError(f"exogenous_text_contains column not available in blind context: {column!r}")
        needles = [clean_text(value) for value in condition.get("values", []) if clean_text(value)]
        return rows[exog_column].map(lambda value: any(needle in clean_text(value) for needle in needles)).to_numpy(dtype=bool)
    if kind == "exogenous_bool_is":
        column = str(condition["column"])
        day_offset = int(condition.get("day_offset", 0) or 0)
        lower_column = column.lower()
        if lower_column in FORBIDDEN_BLIND_KEYS or lower_column.startswith(FORBIDDEN_BLIND_PREFIXES):
            raise ValueError("exogenous_bool_is cannot target actual/residual/metric columns")
        text_column = exog_text_column(column, day_offset)
        numeric_column = exog_numeric_column(column, day_offset)
        if text_column in rows.columns:
            return rows[text_column].map(lambda value: truthy(value) == bool(condition["value"])).to_numpy(dtype=bool)
        if numeric_column in rows.columns:
            numeric = pd.to_numeric(rows[numeric_column], errors="coerce")
            return numeric.map(
                lambda value: pd.notna(value) and ((float(value) != 0.0) == bool(condition["value"]))
            ).to_numpy(dtype=bool)
        raise ValueError(f"exogenous_bool_is column not available in blind context: {column!r}")
    raise ValueError(f"Unsupported adjustment condition kind: {kind!r}")


def apply_action(raw_original: np.ndarray, action: dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    action_type = action.get("type")
    if action_type == "add":
        return raw_original + float(action["value"])
    if action_type == "multiply":
        return raw_original * float(action["value"])
    if action_type == "clip":
        lower = action.get("min", None)
        upper = action.get("max", None)
        return np.clip(
            raw_original,
            None if lower is None else float(lower),
            None if upper is None else float(upper),
        )
    if action_type == "add_profile":
        values = np.asarray(action.get("values", []), dtype=np.float64)
        horizons = rows["horizon_index"].to_numpy(dtype=np.int64)
        if values.size == 0:
            raise ValueError("add_profile requires at least one value")
        if np.any(horizons < 0) or np.any(horizons >= values.size):
            raise ValueError("add_profile values must cover every matched horizon_index")
        return raw_original + values[horizons]
    raise ValueError(f"Unsupported adjustment action type: {action_type!r}")


def enforce_delta_bound(
    *,
    raw_original: np.ndarray,
    adjusted_original: np.ndarray,
    max_delta_ratio: float,
    max_delta_original: float | None = None,
    min_delta_original: float | None = None,
) -> tuple[np.ndarray, int]:
    delta = adjusted_original - raw_original
    denom = np.maximum(np.abs(raw_original), 1e-12)
    ratio_delta = denom * max_delta_ratio
    max_delta = ratio_delta.copy()
    min_delta = -ratio_delta
    if max_delta_original is not None:
        max_delta = np.minimum(max_delta, float(max_delta_original))
    if min_delta_original is not None:
        min_delta = np.maximum(min_delta, float(min_delta_original))
    clipped_delta = np.clip(delta, min_delta, max_delta)
    clipped = int(np.sum(np.abs(delta - clipped_delta) > 1e-12))
    return raw_original + clipped_delta, clipped


def effective_max_delta_ratio(rule: dict[str, Any], policy_max_delta_ratio: float) -> float:
    rule_max_delta_ratio = float(
        rule.get("bounds", {}).get("max_delta_ratio", policy_max_delta_ratio)
    )
    return min(rule_max_delta_ratio, policy_max_delta_ratio)


def rule_absolute_bound(rule: dict[str, Any], key: str) -> float | None:
    value = rule.get("bounds", {}).get(key, None)
    if value is None:
        return None
    return float(value)


def audit_blind_rows(blind_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    forbidden_seen = sorted(
        {
            key
            for row in blind_rows
            for key in row
            if key.lower() in FORBIDDEN_BLIND_KEYS or key.lower().startswith(FORBIDDEN_BLIND_PREFIXES)
        }
    )
    events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "check": "blind_input_forbidden_keys",
            "passed": len(forbidden_seen) == 0,
            "forbidden_keys": forbidden_seen,
        }
    )
    return events


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task_path = Path(args.task_json)
    predictions_path = Path(args.predictions_csv)
    candidate_path = Path(args.candidate_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task = load_json(task_path)
    candidate = load_json(candidate_path)
    policy = load_json(args.policy_json) if args.policy_json else {}
    max_policy_delta = float(policy.get("max_point_delta_ratio", 0.1))

    pred_df = pd.read_csv(predictions_path)
    required_columns = {"timestamp", "predicted"}
    missing = required_columns.difference(pred_df.columns)
    if missing:
        raise ValueError(f"predictions.csv missing required columns: {sorted(missing)}")

    mean, scale = target_scale(task, task_path)
    raw_scaled = pred_df["predicted"].to_numpy(dtype=np.float64)
    raw_original = scaled_to_original(raw_scaled, mean, scale)
    adjusted_original = raw_original.copy()

    pred_len = int(task["pred_len"])
    row_index = np.arange(len(pred_df), dtype=np.int64)
    window_index, horizon_index = prediction_coordinates(pred_df, pred_len)
    work_rows = pd.DataFrame(
        {
            "timestamp": pred_df["timestamp"].astype(str),
            "prediction_row_index": row_index,
            "window_index": window_index,
            "horizon_index": horizon_index,
            "raw_predicted": raw_scaled,
            "raw_predicted_original": raw_original,
        }
    )
    work_rows = attach_exogenous_features(task, task_path, work_rows, day_offsets=candidate_day_offsets(candidate))

    applied_rules: list[dict[str, Any]] = []
    total_clipped = 0
    matched_any = np.zeros(len(pred_df), dtype=bool)
    for rule in candidate.get("rules", []):
        condition_matches = condition_mask(rule["condition"], work_rows)
        overlap_skipped = condition_matches & matched_any
        mask = condition_matches & ~matched_any
        if not mask.any():
            applied_rules.append(
                {
                    "rule_id": rule["id"],
                    "condition_matched_rows": int(condition_matches.sum()),
                    "matched_rows": 0,
                    "overlap_skipped_rows": int(overlap_skipped.sum()),
                    "clipped_rows": 0,
                }
            )
            continue
        matched_any |= mask
        proposed = apply_action(adjusted_original[mask], rule["action"], work_rows.loc[mask])
        max_delta = effective_max_delta_ratio(rule, max_policy_delta)
        bounded, clipped = enforce_delta_bound(
            raw_original=raw_original[mask],
            adjusted_original=proposed,
            max_delta_ratio=max_delta,
            max_delta_original=rule_absolute_bound(rule, "max_delta_original"),
            min_delta_original=rule_absolute_bound(rule, "min_delta_original"),
        )
        adjusted_original[mask] = bounded
        total_clipped += clipped
        applied_rules.append(
            {
                "rule_id": rule["id"],
                "condition_matched_rows": int(condition_matches.sum()),
                "matched_rows": int(mask.sum()),
                "overlap_skipped_rows": int(overlap_skipped.sum()),
                "clipped_rows": clipped,
                "max_delta_ratio": max_delta,
                "max_delta_original": rule_absolute_bound(rule, "max_delta_original"),
                "min_delta_original": rule_absolute_bound(rule, "min_delta_original"),
            }
        )

    # Keep every unchanged prediction bit-for-bit identical to the raw artifact.
    # A full scale round-trip can introduce floating-point noise and falsely turn
    # a dormant candidate into a tiny negative validation result.
    proposed_change_mask = adjusted_original != raw_original
    adjusted_scaled = raw_scaled.copy()
    if proposed_change_mask.any():
        adjusted_scaled[proposed_change_mask] = original_to_scaled(
            adjusted_original[proposed_change_mask], mean, scale
        )
    changed_mask = adjusted_scaled != raw_scaled

    blind_rows = []
    for index, row in work_rows.iterrows():
        blind_row = {
            "timestamp": row["timestamp"],
            "prediction_row_index": int(row["prediction_row_index"]),
            "window_index": int(row["window_index"]),
            "horizon_index": int(row["horizon_index"]),
            "raw_predicted": float(row["raw_predicted"]),
            "raw_predicted_original": float(row["raw_predicted_original"]),
        }
        for column in work_rows.columns:
            if column.startswith(EXOG_PREFIX):
                value = row[column]
                blind_row[column] = None if pd.isna(value) else float(value)
            elif column.startswith(EXOG_TEXT_PREFIX):
                blind_row[column] = clean_text(row[column])
        blind_rows.append(blind_row)

    blind_input = {
        "source_run_id": args.source_run_id,
        "candidate_id": candidate["candidate_id"],
        "split": args.split,
        "target_values_visible": False,
        "residuals_visible": False,
        "metrics_visible": False,
        "rows": blind_rows,
    }
    audit_events = audit_blind_rows(blind_rows)

    blind_input_path = out_dir / "blind-input.json"
    adjusted_values_path = out_dir / "adjusted-values.json"
    adjusted_predictions_path = out_dir / "adjusted_predictions.csv"
    audit_path = out_dir / "leakage-audit.jsonl"
    apply_results_path = out_dir / "apply-results.json"

    blind_input_path.write_text(json.dumps(blind_input, indent=2, sort_keys=True), encoding="utf-8")

    adjusted_rows = []
    for index, row in work_rows.iterrows():
        adjusted_rows.append(
            {
                "timestamp": row["timestamp"],
                "prediction_row_index": int(row["prediction_row_index"]),
                "window_index": int(row["window_index"]),
                "horizon_index": int(row["horizon_index"]),
                "raw_predicted": float(raw_scaled[index]),
                "adjusted_predicted": float(adjusted_scaled[index]),
                "raw_predicted_original": float(raw_original[index]),
                "adjusted_predicted_original": float(adjusted_original[index]),
            }
        )

    adjusted_values_path.write_text(
        json.dumps(
            {
                "source_run_id": args.source_run_id,
                "candidate_id": candidate["candidate_id"],
                "split": args.split,
                "rows": adjusted_rows,
                "locked_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_df = pred_df.copy()
    output_df["raw_predicted"] = raw_scaled
    output_df["predicted"] = adjusted_scaled
    output_df["raw_predicted_original"] = raw_original
    output_df["adjusted_predicted_original"] = adjusted_original
    output_df["prediction_row_index"] = row_index
    output_df["window_index"] = window_index
    output_df["horizon_index"] = horizon_index
    if "actual" in output_df.columns:
        output_df["actual_original"] = scaled_to_original(output_df["actual"].to_numpy(dtype=np.float64), mean, scale)
    output_df.to_csv(adjusted_predictions_path, index=False)

    with audit_path.open("w", encoding="utf-8") as handle:
        for event in audit_events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    apply_result = {
        "status": "success",
        "source_run_id": args.source_run_id,
        "candidate_id": candidate["candidate_id"],
        "split": args.split,
        "blind_input_path": str(blind_input_path),
        "adjusted_values_path": str(adjusted_values_path),
        "adjusted_predictions_path": str(adjusted_predictions_path),
        "leakage_audit_path": str(audit_path),
        "applied_rules": applied_rules,
        "matched_rows": int(matched_any.sum()),
        "changed_rows": int(changed_mask.sum()),
        "triggered_rule_count": int(
            sum(int(rule.get("matched_rows", 0)) > 0 for rule in applied_rules)
        ),
        "zero_trigger": bool(not matched_any.any()),
        "total_clipped_rows": total_clipped,
        "row_count": int(len(pred_df)),
    }
    apply_results_path.write_text(json.dumps(apply_result, indent=2, sort_keys=True), encoding="utf-8")
    json.dump(apply_result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
