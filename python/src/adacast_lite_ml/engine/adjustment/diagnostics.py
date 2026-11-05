"""Summarize validation-front adjustment fit residuals for candidate design."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adacast_lite_ml.engine.adjustment.prediction_layout import prediction_coordinates
from adacast_lite_ml.engine.evidence import EVIDENCE_CANDIDATE_BOUNDARIES, load_data_analysis_knowledge

from adacast_lite_ml.engine.adjustment.apply import (
    EXOG_PREFIX,
    EXOG_TEXT_PREFIX,
    FORBIDDEN_BLIND_KEYS,
    FORBIDDEN_BLIND_PREFIXES,
    attach_exogenous_features,
    resolve_dataset_path,
    safe_exogenous_columns,
    safe_text_exogenous_columns,
    scaled_to_original,
    target_scale,
)


DIAGNOSTIC_SHIFT_DAY_OFFSETS = (-1, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create AdaCast adjustment fit diagnostics")
    parser.add_argument("--task-json", required=True, help="Path to .forecast/task.json")
    parser.add_argument("--predictions-csv", required=True, help="Fit-run predictions.csv path")
    parser.add_argument("--out", required=True, help="Path to diagnostics JSON")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum high-signal rows per bucket family")
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def summarize_rows(frame: pd.DataFrame) -> dict[str, Any]:
    residual = frame["residual_original"].to_numpy(dtype=np.float64)
    actual = frame["actual_original"].to_numpy(dtype=np.float64)
    predicted = frame["raw_predicted_original"].to_numpy(dtype=np.float64)
    errors = predicted - actual
    row_count = int(len(frame))
    under = residual > 0
    over = residual < 0
    under_rate = float(np.mean(under)) if row_count else 0.0
    over_rate = float(np.mean(over)) if row_count else 0.0
    return {
        "row_count": row_count,
        "residual_mean_original": float(np.mean(residual)) if row_count else 0.0,
        "residual_median_original": float(np.median(residual)) if row_count else 0.0,
        "mae_original": float(np.mean(np.abs(errors))) if row_count else 0.0,
        "mse_original": float(np.mean(np.square(errors))) if row_count else 0.0,
        "underprediction_rate": under_rate,
        "overprediction_rate": over_rate,
        "sign_consistency": float(max(under_rate, over_rate)),
        "raw_predicted_original_min": float(np.min(predicted)) if row_count else None,
        "raw_predicted_original_max": float(np.max(predicted)) if row_count else None,
    }


def grouped_summary(
    frame: pd.DataFrame,
    group_col: str,
    *,
    top_k: int,
    min_rows: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_value, group in frame.groupby(group_col, dropna=False):
        if len(group) < min_rows:
            continue
        summary = summarize_rows(group)
        score = abs(float(summary["residual_mean_original"])) * np.sqrt(float(summary["row_count"]))
        rows.append(
            {
                "group": None if pd.isna(group_value) else str(group_value),
                "signal_score": float(score),
                **summary,
            }
        )
    rows.sort(key=lambda item: (item["signal_score"], item["row_count"]), reverse=True)
    return rows[:top_k]


def grouped_summary_multi(
    frame: pd.DataFrame,
    group_cols: list[str],
    *,
    top_k: int,
    min_rows: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_value, group in frame.groupby(group_cols, dropna=False):
        if len(group) < min_rows:
            continue
        values = group_value if isinstance(group_value, tuple) else (group_value,)
        group_values = {
            column: None if pd.isna(value) else int(value) if isinstance(value, (np.integer, int)) else str(value)
            for column, value in zip(group_cols, values, strict=True)
        }
        summary = summarize_rows(group)
        score = abs(float(summary["residual_mean_original"])) * np.sqrt(float(summary["row_count"]))
        rows.append(
            {
                "group_values": group_values,
                "signal_score": float(score),
                **summary,
            }
        )
    rows.sort(key=lambda item: (item["signal_score"], item["row_count"]), reverse=True)
    return rows[:top_k]


def rule_hint_from_bucket(
    *,
    bucket: dict[str, Any],
    conditions: list[dict[str, Any]],
    evidence_ref: str,
    hint_id: str,
    min_sign_consistency: float = 0.6,
) -> dict[str, Any] | None:
    row_count = int(bucket.get("row_count", 0))
    residual_mean = float(bucket.get("residual_mean_original", 0.0))
    sign_consistency = float(bucket.get("sign_consistency", 0.0))
    if row_count <= 0 or abs(residual_mean) <= 0 or sign_consistency < min_sign_consistency:
        return None
    condition = conditions[0] if len(conditions) == 1 else {"kind": "all", "conditions": conditions}
    return {
        "id": hint_id,
        "condition": condition,
        "action": {
            "type": "add",
            "value": residual_mean,
        },
        "bounds": {
            "max_delta_ratio": 0.05,
        },
        "evidence_ref": evidence_ref,
        "row_count": row_count,
        "residual_mean_original": residual_mean,
        "sign_consistency": sign_consistency,
        "note": "Validation-front fit hint only; write a candidate and validate on the held-out validation back half before accepting.",
    }


def parse_int_group(value: Any) -> int:
    return int(float(str(value)))


def build_rule_hints(output: dict[str, Any], evidence_ref: str, top_k: int) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []

    for bucket in output["top_horizon_biases"]:
        group = bucket.get("group")
        if group is None:
            continue
        hint = rule_hint_from_bucket(
            bucket=bucket,
            conditions=[{"kind": "horizon_index_in", "values": [parse_int_group(group)]}],
            evidence_ref=evidence_ref,
            hint_id=f"horizon-{group}",
        )
        if hint:
            hints.append(hint)

    for bucket in output["timestamp_hour_biases"]:
        group = bucket.get("group")
        if group is None:
            continue
        hint = rule_hint_from_bucket(
            bucket=bucket,
            conditions=[{"kind": "timestamp_hour_in", "values": [parse_int_group(group)]}],
            evidence_ref=evidence_ref,
            hint_id=f"hour-{group}",
        )
        if hint:
            hints.append(hint)

    for bucket in output["timestamp_month_biases"]:
        group = bucket.get("group")
        if group is None:
            continue
        hint = rule_hint_from_bucket(
            bucket=bucket,
            conditions=[{"kind": "timestamp_month_in", "values": [parse_int_group(group)]}],
            evidence_ref=evidence_ref,
            hint_id=f"month-{group}",
        )
        if hint:
            hints.append(hint)

    for bucket in output["raw_predicted_original_quantile_biases"]:
        lower = bucket.get("raw_predicted_original_min")
        upper = bucket.get("raw_predicted_original_max")
        if lower is None or upper is None:
            continue
        hint = rule_hint_from_bucket(
            bucket=bucket,
            conditions=[
                {
                    "kind": "raw_predicted_original_between",
                    "min": float(lower),
                    "max": float(upper),
                }
            ],
            evidence_ref=evidence_ref,
            hint_id=f"prediction-range-{len(hints) + 1}",
        )
        if hint:
            hints.append(hint)

    for family, condition_columns in (
        ("horizon-hour", ["horizon_index", "timestamp_hour"]),
        ("month-horizon", ["timestamp_month", "horizon_index"]),
    ):
        for bucket in output[f"top_{family.replace('-', '_')}_biases"]:
            values = bucket["group_values"]
            conditions = []
            if values.get("horizon_index") is not None:
                conditions.append({"kind": "horizon_index_in", "values": [parse_int_group(values["horizon_index"])]})
            if values.get("timestamp_hour") is not None:
                conditions.append({"kind": "timestamp_hour_in", "values": [parse_int_group(values["timestamp_hour"])]})
            if values.get("timestamp_month") is not None:
                conditions.append({"kind": "timestamp_month_in", "values": [parse_int_group(values["timestamp_month"])]})
            if len(conditions) < len(condition_columns):
                continue
            hint = rule_hint_from_bucket(
                bucket=bucket,
                conditions=conditions,
                evidence_ref=evidence_ref,
                hint_id=f"{family}-{len(hints) + 1}",
            )
            if hint:
                hints.append(hint)

    hints.sort(key=lambda item: (abs(float(item["residual_mean_original"])) * np.sqrt(float(item["row_count"]))), reverse=True)
    return hints[:top_k]


def timestamp_hour(value: Any) -> int | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.hour)


def timestamp_month(value: Any) -> int | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.month)


def predicted_quantile_labels(values: pd.Series) -> pd.Series:
    try:
        buckets = pd.qcut(values, q=5, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index)
    labels = []
    for item in buckets.astype(str):
        labels.append(item)
    return pd.Series(labels, index=values.index)


def exogenous_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(EXOG_PREFIX)]


def exogenous_text_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(EXOG_TEXT_PREFIX)]


def public_exogenous_name(column: str) -> str:
    return column.removeprefix(EXOG_PREFIX)


def public_text_exogenous_name(column: str) -> str:
    return column.removeprefix(EXOG_TEXT_PREFIX)


def exogenous_feature_overview(frame: pd.DataFrame) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    residual = frame["residual_original"].to_numpy(dtype=np.float64)
    for column in exogenous_columns(frame):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        finite_count = int(np.sum(finite))
        correlation = None
        if finite_count >= 3 and np.nanstd(values[finite]) > 0 and np.nanstd(residual[finite]) > 0:
            correlation = float(np.corrcoef(values[finite], residual[finite])[0, 1])
        features.append(
            {
                "column": public_exogenous_name(column),
                "row_count": int(len(frame)),
                "finite_count": finite_count,
                "missing_rate": float(1 - finite_count / max(len(frame), 1)),
                "min": float(np.nanmin(values[finite])) if finite_count else None,
                "max": float(np.nanmax(values[finite])) if finite_count else None,
                "mean": float(np.nanmean(values[finite])) if finite_count else None,
                "residual_correlation": correlation,
                "abs_residual_correlation": abs(correlation) if correlation is not None else None,
            }
        )
    features.sort(key=lambda item: (item["abs_residual_correlation"] is not None, item["abs_residual_correlation"] or 0), reverse=True)
    return features


def task_context_summary(task: dict[str, Any]) -> dict[str, Any]:
    context_keys = [
        "business_context",
        "csv_semantics",
        "region_info",
        "seasonality_hints",
        "holiday_effects",
        "anomaly_periods",
        "adjustment_skill_priors",
    ]
    summary = {key: task.get(key) for key in context_keys if task.get(key)}
    summary["window_contract"] = {
        "seq_len": int(task["seq_len"]),
        "forecast_gap": int(task.get("forecast_gap", 0)),
        "pred_len": int(task["pred_len"]),
        "window_stride": int(task.get("window_stride", 1)),
        "horizon_index_zero_meaning": "first evaluated point after forecast_gap",
    }
    return summary


def task_knowledge_summary(task: dict[str, Any], task_path: Path) -> dict[str, Any]:
    stable_keys = ["business_context", "csv_semantics", "region_info", "seasonality_hints"]
    summary = {key: task.get(key) for key in stable_keys if task.get(key)}
    summary["window_contract"] = task_context_summary(task)["window_contract"]
    summary["data_analysis"] = load_data_analysis_knowledge(task, task_path)
    return summary


def task_event_context_summary(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: task.get(key)
        for key in ("holiday_effects", "anomaly_periods")
        if task.get(key)
    }


def dataset_feature_ledger(task: dict[str, Any], task_path: Path) -> list[dict[str, Any]]:
    dataset = pd.read_csv(resolve_dataset_path(task, task_path), low_memory=False)
    time_col = str(task["time_col"])
    target_col = str(task["target_col"])
    safe_numeric = set(safe_exogenous_columns(task, dataset))
    safe_text = set(safe_text_exogenous_columns(task, dataset))
    rows: list[dict[str, Any]] = []
    for column in dataset.columns:
        lower = str(column).lower()
        role = "exogenous_candidate"
        blocked_reason = None
        if column == time_col:
            role = "time"
            blocked_reason = "time column is already represented by timestamp features"
        elif column == target_col:
            role = "target"
            blocked_reason = "target column would leak horizon actual values"
        elif lower in FORBIDDEN_BLIND_KEYS or lower.startswith(FORBIDDEN_BLIND_PREFIXES):
            role = "blocked"
            blocked_reason = "column name resembles actual/residual/error/metric information"
        elif column not in safe_numeric and column not in safe_text:
            role = "unsupported"
            blocked_reason = "only numeric, text, and boolean exogenous feature conditions are currently executable"
        rows.append(
            {
                "column": str(column),
                "role": role,
                "dtype": str(dataset[column].dtype),
                "usable_for_exogenous_numeric_between": column in safe_numeric,
                "usable_for_exogenous_text_conditions": column in safe_text,
                "blocked_reason": blocked_reason,
                "availability_assumption": (
                    "supplied as a visible forecast-horizon input"
                    if column in safe_numeric or column in safe_text
                    else None
                ),
            }
        )
    return rows


def exogenous_bucket_summaries(frame: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for column in exogenous_columns(frame):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = frame[np.isfinite(numeric.to_numpy(dtype=np.float64))].copy()
        if len(valid) < 2:
            continue
        try:
            valid["__exog_bucket"] = pd.qcut(pd.to_numeric(valid[column]), q=5, duplicates="drop").astype(str)
        except ValueError:
            valid["__exog_bucket"] = "all"
        for bucket_label, group in valid.groupby("__exog_bucket", dropna=False):
            summary = summarize_rows(group)
            exog_values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
            finite = exog_values[np.isfinite(exog_values)]
            if finite.size == 0:
                continue
            score = abs(float(summary["residual_mean_original"])) * np.sqrt(float(summary["row_count"]))
            buckets.append(
                {
                    "column": public_exogenous_name(column),
                    "bucket": str(bucket_label),
                    "exogenous_min": float(np.min(finite)),
                    "exogenous_max": float(np.max(finite)),
                    "signal_score": float(score),
                    **summary,
                }
            )
    buckets.sort(key=lambda item: (item["signal_score"], item["row_count"]), reverse=True)
    return buckets[:top_k]


def append_exogenous_rule_hints(
    hints: list[dict[str, Any]],
    exogenous_buckets: list[dict[str, Any]],
    evidence_ref: str,
) -> None:
    for bucket in exogenous_buckets:
        hint = rule_hint_from_bucket(
            bucket=bucket,
            conditions=[
                {
                    "kind": "exogenous_numeric_between",
                    "column": bucket["column"],
                    "min": bucket["exogenous_min"],
                    "max": bucket["exogenous_max"],
                }
            ],
            evidence_ref=evidence_ref,
            hint_id=f"exogenous-{bucket['column']}-{len(hints) + 1}",
        )
        if hint:
            hints.append(hint)


def observed_condition_domains(frame: pd.DataFrame) -> dict[str, Any]:
    raw_values = frame["raw_predicted_original"].to_numpy(dtype=np.float64)
    finite_raw = raw_values[np.isfinite(raw_values)]
    exogenous: dict[str, Any] = {}
    for column in exogenous_columns(frame):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        exogenous[public_exogenous_name(column)] = {
            "finite_count": int(finite.size),
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
    text_exogenous: dict[str, Any] = {}
    for column in exogenous_text_columns(frame):
        values = frame[column].map(lambda value: "" if pd.isna(value) else str(value).strip())
        observed = sorted(value for value in values.unique().tolist() if value)
        text_exogenous[public_text_exogenous_name(column)] = {
            "finite_count": int(np.sum(values != "")),
            "values": observed[:100],
            "truncated": len(observed) > 100,
        }
    return {
        "horizon_indices": sorted(int(value) for value in frame["horizon_index"].dropna().unique()),
        "timestamp_hours": sorted(int(value) for value in frame["timestamp_hour"].dropna().unique()),
        "timestamp_months": sorted(int(value) for value in frame["timestamp_month"].dropna().unique()),
        "timestamp_minutes_of_day": sorted(int(value) for value in frame["timestamp_minute_of_day"].dropna().unique()),
        "raw_predicted_original": {
            "min": float(np.min(finite_raw)) if finite_raw.size else None,
            "max": float(np.max(finite_raw)) if finite_raw.size else None,
        },
        "exogenous_numeric": exogenous,
        "exogenous_text": text_exogenous,
    }


def diagnostics_evidence_ref(task_path: Path, out_path: Path) -> str:
    project_root = task_path.resolve().parent.parent
    try:
        return out_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return out_path.as_posix()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task_path = Path(args.task_json)
    task = load_json(task_path)
    pred_df = pd.read_csv(args.predictions_csv)
    required = {"timestamp", "actual", "predicted"}
    missing = required.difference(pred_df.columns)
    if missing:
        raise ValueError(f"fit predictions.csv missing required columns: {sorted(missing)}")

    mean, scale = target_scale(task, task_path)
    actual_scaled = pred_df["actual"].to_numpy(dtype=np.float64)
    pred_scaled = pred_df["predicted"].to_numpy(dtype=np.float64)
    actual_original = scaled_to_original(actual_scaled, mean, scale)
    pred_original = scaled_to_original(pred_scaled, mean, scale)
    pred_len = int(task["pred_len"])
    row_index = np.arange(len(pred_df), dtype=np.int64)
    window_index, horizon_index = prediction_coordinates(pred_df, pred_len)

    frame = pd.DataFrame(
        {
            "timestamp": pred_df["timestamp"].astype(str),
            "prediction_row_index": row_index,
            "window_index": window_index,
            "horizon_index": horizon_index,
            "actual_original": actual_original,
            "raw_predicted_original": pred_original,
            "residual_original": actual_original - pred_original,
        }
    )
    frame = attach_exogenous_features(
        task,
        task_path,
        frame,
        day_offsets=(0, *DIAGNOSTIC_SHIFT_DAY_OFFSETS),
    )
    frame["timestamp_hour"] = frame["timestamp"].map(timestamp_hour)
    frame["timestamp_month"] = frame["timestamp"].map(timestamp_month)
    frame["timestamp_minute_of_day"] = frame["timestamp"].map(
        lambda value: None
        if pd.isna(pd.to_datetime(value, errors="coerce"))
        else int(pd.to_datetime(value, errors="coerce").hour) * 60
        + int(pd.to_datetime(value, errors="coerce").minute)
    )
    frame["raw_predicted_original_quantile"] = predicted_quantile_labels(frame["raw_predicted_original"])

    top_k = max(1, int(args.top_k))
    top_exogenous_buckets = exogenous_bucket_summaries(frame, top_k=top_k)
    out_path = Path(args.out)
    evidence_ref = diagnostics_evidence_ref(task_path, out_path)
    output = {
        "status": "success",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "predictions_csv": str(args.predictions_csv),
        "target_values_visible": True,
        "split_scope": "fit_only_validation_front",
        "task_context": task_context_summary(task),
        "dataset_feature_ledger": dataset_feature_ledger(task, task_path),
        "observed_condition_domains": observed_condition_domains(frame),
        "overall": summarize_rows(frame),
        "top_window_biases": grouped_summary(frame, "window_index", top_k=top_k),
        "top_horizon_biases": grouped_summary(frame, "horizon_index", top_k=top_k),
        "timestamp_hour_biases": grouped_summary(frame, "timestamp_hour", top_k=24),
        "timestamp_month_biases": grouped_summary(frame, "timestamp_month", top_k=12),
        "raw_predicted_original_quantile_biases": grouped_summary(
            frame,
            "raw_predicted_original_quantile",
            top_k=5,
        ),
        "top_horizon_hour_biases": grouped_summary_multi(
            frame,
            ["horizon_index", "timestamp_hour"],
            top_k=top_k,
        ),
        "top_month_horizon_biases": grouped_summary_multi(
            frame,
            ["timestamp_month", "horizon_index"],
            top_k=top_k,
        ),
        "exogenous_feature_overview": exogenous_feature_overview(frame),
        "top_exogenous_numeric_biases": top_exogenous_buckets,
        "guidance": {
            "safe_rule_conditions": [
                "horizon_index_in",
                "timestamp_hour_in",
                "timestamp_month_in",
                "timestamp_time_between",
                "raw_predicted_original_between",
                "exogenous_numeric_between",
                "exogenous_text_in",
                "exogenous_text_contains",
                "exogenous_bool_is",
                "all",
                "any",
            ],
            "use_validation": "Validate candidates on held-out validation metrics only; do not tune from point-level validation-back-half residuals.",
            "use_exogenous_features": "Use exogenous_numeric_between for numeric non-target columns and text conditions for supplied calendar/weather/event columns; day_offset=1 may express next-day event selectors when present in observed domains.",
            "avoid": "Do not memorize exact fit window_index values into test rules.",
        },
    }
    output["rule_hints"] = build_rule_hints(output, evidence_ref, top_k)
    append_exogenous_rule_hints(output["rule_hints"], top_exogenous_buckets, evidence_ref)
    output["rule_hints"] = sorted(
        output["rule_hints"],
        key=lambda item: abs(float(item["residual_mean_original"])) * np.sqrt(float(item["row_count"])),
        reverse=True,
    )[:top_k]

    evidence_candidates = {
        "experience": {
            "candidate_type": "experience",
            "evidence_candidate_id": "experience:validation-front",
            "definition": EVIDENCE_CANDIDATE_BOUNDARIES["experience"],
            "scope": "reusable only after held-out validation acceptance",
            "source_scope": output["split_scope"],
            "expert_prior": task.get("adjustment_skill_priors"),
            "overall_residual_summary": output["overall"],
            "exogenous_residual_overview": output["exogenous_feature_overview"],
            "rule_hints": output["rule_hints"],
        },
        "context": {
            "candidate_type": "context",
            "evidence_candidate_id": "context:validation-front",
            "definition": EVIDENCE_CANDIDATE_BOUNDARIES["context"],
            "scope": "observed validation-front evidence domain only",
            "declared_events": task_event_context_summary(task),
            "observed_condition_domains": output["observed_condition_domains"],
        },
        "knowledge": {
            "candidate_type": "knowledge",
            "evidence_candidate_id": "knowledge:frozen-task",
            "definition": EVIDENCE_CANDIDATE_BOUNDARIES["knowledge"],
            "scope": "task-wide stable contract",
            "task_semantics_and_structure": task_knowledge_summary(task, task_path),
            "dataset_feature_ledger": output["dataset_feature_ledger"],
            "safe_rule_conditions": output["guidance"]["safe_rule_conditions"],
            "validation_contract": output["guidance"]["use_validation"],
        },
    }
    candidate_storage: dict[str, dict[str, str]] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for candidate_type, payload in evidence_candidates.items():
        candidate_path = out_path.parent / f"evidence-{candidate_type}.json"
        candidate_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        candidate_storage[candidate_type] = {
            "path": str(candidate_path),
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        }
    output["evidence_candidates"] = {
        "classification_contract": EVIDENCE_CANDIDATE_BOUNDARIES,
        **evidence_candidates,
    }
    output["evidence_candidate_storage"] = candidate_storage

    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    json.dump(output, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
