"""Deterministic preparation and verification for isolated LLM window reflection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adacast_lite_ml.engine.adjustment.apply import (
    attach_exogenous_features,
    candidate_day_offsets,
    condition_mask,
    exog_numeric_column,
    exog_text_column,
    resolve_dataset_path,
    safe_exogenous_columns,
    safe_text_exogenous_columns,
    scaled_to_original,
    target_scale,
)
from adacast_lite_ml.engine.adjustment.prediction_layout import prediction_coordinates
from adacast_lite_ml.engine.evidence import EVIDENCE_CANDIDATE_BOUNDARIES, load_data_analysis_knowledge
from adacast_lite_ml.engine.data_loader import resolve_training_boundaries, resolve_validation_start_index
from adacast_lite_ml.engine.utils.metrics import mae, mase, mse, wape


CONTROLLER_VERSION = "2"
SKILL_GUIDED_ACTIVE_RECALIBRATION_SHARE = 0.20
SKILL_GUIDED_ACTIVE_RECALIBRATION_INTERVAL = int(
    round(1.0 / SKILL_GUIDED_ACTIVE_RECALIBRATION_SHARE)
)


def skill_guided_active_recalibration_priority(matched_window_ordinal: int) -> bool:
    if matched_window_ordinal < 1:
        raise ValueError("Matched skill-window ordinal must be positive")
    return (
        (matched_window_ordinal - 1)
        % SKILL_GUIDED_ACTIVE_RECALIBRATION_INTERVAL
        == 0
    )


FORBIDDEN_CONTEXT_KEYS = {
    "actual",
    "residual",
    "error",
    "metric",
    "metrics",
    "mse",
    "mae",
    "wape",
    "mase",
    "future_target",
}


def relocated_checkpoint_path(path: str | Path) -> Path:
    source = str(path)
    root = os.environ.get("ADACAST_LITE_REFLECTION_CHECKPOINT_RELOCATE_ROOT")
    normalized = source.replace("\\", "/")
    if root and re.match(r"^[A-Za-z]:/", normalized):
        relocation_root = Path(root).resolve()
        marker = f"/{relocation_root.name}/"
        marker_index = normalized.rfind(marker)
        if marker_index >= 0:
            return relocation_root / normalized[marker_index + len(marker):]
    return Path(path)


def relocate_checkpoint_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: relocate_checkpoint_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_checkpoint_json(item) for item in value]
    if isinstance(value, str) and re.match(r"^[A-Za-z]:[\\/]", value):
        return str(relocated_checkpoint_path(value))
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(relocated_checkpoint_path(path).read_text(encoding="utf-8-sig"))
    return relocate_checkpoint_json(value)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with relocated_checkpoint_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_policy_target_partitions(
    task: dict[str, Any],
    task_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = resolve_dataset_path(task, task_path)
    time_col = str(task["time_col"])
    boundary_frame = pd.read_csv(
        dataset_path,
        usecols=[time_col],
        low_memory=False,
    )
    train_end, adjustment_fit_end, val_end = resolve_training_boundaries(boundary_frame, task)
    validation_start = resolve_validation_start_index(boundary_frame, task, train_end, val_end)
    target_frame = pd.read_csv(
        dataset_path,
        usecols=[str(task["target_col"])],
        nrows=adjustment_fit_end,
        low_memory=False,
    )
    values = pd.to_numeric(
        target_frame[str(task["target_col"])], errors="coerce"
    ).to_numpy(dtype=np.float64)
    training = values[:train_end]
    validation_front = values[validation_start:adjustment_fit_end]
    if not np.isfinite(training).all() or training.size < 4:
        raise ValueError("At least four finite training target values are required to freeze reflection policy")
    if not np.isfinite(validation_front).all() or validation_front.size < 4:
        raise ValueError(
            "At least four finite validation-front target values are required to calibrate reflection policy"
        )
    return training, validation_front


def finite_training_target(task: dict[str, Any], task_path: Path) -> np.ndarray:
    dataset_path = resolve_dataset_path(task, task_path)
    time_col = str(task["time_col"])
    boundary_frame = pd.read_csv(
        dataset_path,
        usecols=[time_col],
        low_memory=False,
    )
    train_end, _, _ = resolve_training_boundaries(boundary_frame, task)
    target_frame = pd.read_csv(
        dataset_path,
        usecols=[str(task["target_col"])],
        nrows=train_end,
        low_memory=False,
    )
    training = pd.to_numeric(
        target_frame[str(task["target_col"])], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(training).all() or training.size < 4:
        raise ValueError(
            "At least four finite training target values are required to compute final metrics"
        )
    return training


def numeric_tolerance(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    magnitude = float(np.max(np.abs(finite))) if finite.size else 0.0
    return max(
        np.finfo(np.float64).eps * 1024.0,
        magnitude * 1e-9,
        1e-12,
    )


def distribution_statistics(values: np.ndarray) -> dict[str, float]:
    epsilon = numeric_tolerance(values)
    q001, q01, q25, q75, q99, q999 = np.quantile(
        values,
        [0.001, 0.01, 0.25, 0.75, 0.99, 0.999],
    )
    iqr = max(float(q75 - q25), epsilon)
    std = max(float(np.std(values)), epsilon)
    diffs = np.abs(np.diff(values))
    curvatures = np.abs(np.diff(values, n=2))

    def quantile_or_scale(samples: np.ndarray, quantile: float) -> float:
        return float(np.quantile(samples, quantile)) if samples.size else std

    return {
        "observed_min": float(np.min(values)),
        "observed_max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": std,
        "iqr": iqr,
        "range": max(float(np.max(values) - np.min(values)), epsilon),
        "q001": float(q001),
        "q01": float(q01),
        "q99": float(q99),
        "q999": float(q999),
        "diff_q99": quantile_or_scale(diffs, 0.99),
        "diff_q995": quantile_or_scale(diffs, 0.995),
        "curvature_q99": quantile_or_scale(curvatures, 0.99),
        "curvature_q995": quantile_or_scale(curvatures, 0.995),
    }


def validation_front_windows(task: dict[str, Any], values: np.ndarray) -> list[np.ndarray]:
    pred_len = int(task["pred_len"])
    forecast_gap = int(task.get("forecast_gap", 0))
    stride = int(task.get("window_stride", 1))
    starts = range(forecast_gap, len(values) - pred_len + 1, stride)
    windows = [values[start : start + pred_len].copy() for start in starts]
    if not windows:
        raise ValueError(
            "Validation-front split cannot produce one complete reflection calibration window"
        )
    return windows


def validation_front_window_statistics(windows: list[np.ndarray]) -> dict[str, Any]:
    epsilon = numeric_tolerance(np.concatenate(windows))

    def maximum_difference(window: np.ndarray, order: int) -> float:
        differences = np.abs(np.diff(window, n=order))
        return float(np.max(differences)) if differences.size else 0.0

    def summarize(samples: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(samples)),
            "q10": float(np.quantile(samples, 0.1)),
            "q50": float(np.quantile(samples, 0.5)),
            "q90": float(np.quantile(samples, 0.9)),
            "q99": float(np.quantile(samples, 0.99)),
            "max": float(np.max(samples)),
        }

    ranges = np.asarray([float(np.ptp(window)) for window in windows], dtype=np.float64)
    max_steps = np.asarray(
        [maximum_difference(window, 1) for window in windows],
        dtype=np.float64,
    )
    max_curvatures = np.asarray(
        [maximum_difference(window, 2) for window in windows],
        dtype=np.float64,
    )
    return {
        "window_count": len(windows),
        "range": summarize(ranges),
        "max_abs_step": summarize(np.maximum(max_steps, epsilon)),
        "max_abs_curvature": summarize(np.maximum(max_curvatures, epsilon)),
    }


def replay_validation_front(
    windows: list[np.ndarray],
    constraints: dict[str, Any],
    *,
    independent_soft_violation_count: int,
) -> dict[str, Any]:
    policy = {
        "constraints": constraints,
        "trigger_policy": {
            "independent_soft_violation_count": independent_soft_violation_count,
        },
    }
    reports = [detect_window(window, policy) for window in windows]
    hard_count = sum(bool(report["hard_violations"]) for report in reports)
    trigger_count = sum(bool(report["trigger"]) for report in reports)
    return {
        "window_count": len(windows),
        "hard_violation_window_count": hard_count,
        "trigger_count": trigger_count,
        "trigger_rate": trigger_count / len(windows),
    }


def calibrate_soft_constraints(
    windows: list[np.ndarray],
    constraints: dict[str, Any],
    *,
    max_reference_trigger_rate: float = 0.1,
    max_rounds: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibrated = dict(constraints)
    initial = replay_validation_front(
        windows,
        calibrated,
        independent_soft_violation_count=2,
    )
    current = initial
    adjustment_rounds = 0
    while current["trigger_rate"] > max_reference_trigger_rate and adjustment_rounds < max_rounds:
        adjustment_rounds += 1
        calibrated["soft_absolute_min"] -= 0.15 * (
            calibrated["soft_absolute_min"] - calibrated["absolute_min"]
        )
        calibrated["soft_absolute_max"] += 0.15 * (
            calibrated["absolute_max"] - calibrated["soft_absolute_max"]
        )
        calibrated["soft_abs_step"] += 0.15 * (
            calibrated["max_abs_step"] - calibrated["soft_abs_step"]
        )
        calibrated["soft_abs_curvature"] += 0.15 * (
            calibrated["max_abs_curvature"] - calibrated["soft_abs_curvature"]
        )
        calibrated["min_window_range"] *= 0.85
        current = replay_validation_front(
            windows,
            calibrated,
            independent_soft_violation_count=2,
        )

    if current["hard_violation_window_count"]:
        raise ValueError(
            "Frozen hard reflection constraints reject known validation-front target windows"
        )
    if current["trigger_rate"] > max_reference_trigger_rate:
        raise ValueError(
            "Unable to calibrate reflection soft constraints within the validation-front "
            "reference trigger-rate limit"
        )
    return calibrated, {
        "max_reference_trigger_rate": max_reference_trigger_rate,
        "initial_trigger_count": initial["trigger_count"],
        "initial_trigger_rate": initial["trigger_rate"],
        "final_trigger_count": current["trigger_count"],
        "final_trigger_rate": current["trigger_rate"],
        "hard_violation_window_count": current["hard_violation_window_count"],
        "soft_threshold_adjustment_rounds": adjustment_rounds,
    }


def derive_policy(
    task: dict[str, Any],
    task_path: Path,
    planner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training, validation_front = finite_policy_target_partitions(task, task_path)
    values = np.concatenate([training, validation_front])
    planner = planner or {}
    training_stats = distribution_statistics(training)
    validation_stats = distribution_statistics(validation_front)
    calibrated = distribution_statistics(values)
    validation_windows = validation_front_windows(task, validation_front)
    window_stats = validation_front_window_statistics(validation_windows)

    std = calibrated["std"]
    iqr = calibrated["iqr"]
    observed_min = calibrated["observed_min"]
    observed_max = calibrated["observed_max"]

    derived_min = min(
        observed_min,
        float(calibrated["q001"] - max(3.5 * iqr, 2.5 * std)),
    )
    derived_max = max(
        observed_max,
        float(calibrated["q999"] + max(3.5 * iqr, 2.5 * std)),
    )
    nonnegative = bool(planner.get("nonnegative_target", observed_min >= 0.0)) and observed_min >= 0.0
    if nonnegative:
        derived_min = 0.0

    hard_min = float(planner.get("absolute_min", derived_min))
    hard_max = float(planner.get("absolute_max", derived_max))
    soft_padding = max(0.02 * iqr, 0.01 * std, np.finfo(np.float64).eps)
    soft_min = max(hard_min, float(calibrated["q01"] - soft_padding))
    soft_max = min(hard_max, float(calibrated["q99"] + soft_padding))
    calibrated_window_step_max = float(window_stats["max_abs_step"]["max"])
    max_step = float(
        planner.get(
            "max_abs_step",
            max(
                calibrated_window_step_max * 1.25,
                calibrated["diff_q995"] * 1.5,
                numeric_tolerance(values),
            ),
        )
    )
    soft_step = min(
        max(float(window_stats["max_abs_step"]["q90"]), numeric_tolerance(values)),
        max_step * 0.75,
    )
    calibrated_window_curvature_max = float(window_stats["max_abs_curvature"]["max"])
    max_curvature = max(
        calibrated_window_curvature_max * 1.25,
        calibrated["curvature_q995"] * 1.5,
        numeric_tolerance(values),
    )
    soft_curvature = min(
        max(
            float(window_stats["max_abs_curvature"]["q90"]),
            numeric_tolerance(values),
        ),
        max_curvature * 0.75,
    )
    min_window_range = max(float(window_stats["range"]["q10"]) * 0.75, 0.0)
    hard_min_window_range = max(float(window_stats["range"]["min"]) * 0.5, 0.0)
    mean_delta = float(
        planner.get(
            "window_mean_abs_delta_limit",
            max(std * 0.5, iqr * 0.75, (observed_max - observed_min) * 0.08),
        )
    )
    point_delta = float(
        planner.get(
            "point_emergency_delta_limit",
            max(mean_delta * 3.0, std * 3.0, iqr * 4.0),
        )
    )

    if hard_min > observed_min or hard_max < observed_max:
        raise ValueError(
            "Planner value bounds may not exclude finite training or validation-front observations"
        )
    if hard_min >= hard_max:
        raise ValueError("Reflection absolute_min must be smaller than absolute_max")
    if soft_min >= soft_max:
        raise ValueError("Calibrated soft absolute range is empty")
    if max_step < calibrated_window_step_max:
        raise ValueError(
            "Planner max_abs_step may not reject known validation-front target windows"
        )
    if mean_delta < max(std * 0.1, iqr * 0.1):
        raise ValueError("window_mean_abs_delta_limit is too strict for the observed training scale")
    if point_delta < mean_delta * 2.0:
        raise ValueError("point_emergency_delta_limit must be at least twice the window mean limit")

    constraints, replay = calibrate_soft_constraints(
        validation_windows,
        {
            "nonnegative": nonnegative,
            "absolute_min": hard_min,
            "absolute_max": hard_max,
            "soft_absolute_min": soft_min,
            "soft_absolute_max": soft_max,
            "soft_abs_step": soft_step,
            "max_abs_step": max_step,
            "soft_abs_curvature": soft_curvature,
            "max_abs_curvature": max_curvature,
            "hard_min_window_range": hard_min_window_range,
            "min_window_range": min_window_range,
            "window_mean_abs_delta_limit": mean_delta,
            "point_emergency_delta_limit": point_delta,
        },
    )

    return {
        "controller_version": CONTROLLER_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "enabled": bool(task.get("reflection_enabled", True)),
        "max_reflection_rounds": 3,
        "max_format_retries": 2,
        "trigger_policy": {
            "hard_violation_count": 1,
            "independent_soft_violation_count": 2,
        },
        "target_scale": {
            "training": training_stats,
            "validation_front": validation_stats,
            "calibrated": calibrated,
        },
        "calibration": {
            "method": "training_baseline_validation_front_calibration_v2",
            "source_split": "validation_front",
            "training_row_count": int(training.size),
            "validation_front_row_count": int(validation_front.size),
            "validation_front_window_count": len(validation_windows),
            "validation_back_used": False,
            "test_used": False,
            "validation_front_window_statistics": window_stats,
            "validation_front_reference_replay": replay,
            "calibrated_rules": [
                "absolute_range",
                "adjacent_step",
                "curvature",
                "severe_flat_window",
                "flat_window",
                "window_mean_correction",
                "point_emergency_correction",
            ],
        },
        "constraints": constraints,
        "planner_rationale": str(
            planner.get(
                "rationale",
                "Moderate target-scale safety limits derived from training data and calibrated on validation-front only.",
            )
        ),
        "validation_gate": {
            "metric": "mse",
            "accept_if": "final_mse <= baseline_mse",
            "zero_trigger": "enabled_no_trigger",
        },
    }


def detect_window(
    values: np.ndarray,
    policy: dict[str, Any],
    *,
    reference: np.ndarray | None = None,
    matched_rule_ids: list[str] | None = None,
    require_adjustment_evidence: bool = False,
) -> dict[str, Any]:
    constraints = policy["constraints"]
    hard: list[dict[str, Any]] = []
    soft: list[dict[str, Any]] = []
    if values.size == 0 or not np.isfinite(values).all():
        hard.append({"id": "finite_values", "message": "Window contains missing or non-finite predictions."})
    else:
        below = int(np.sum(values < float(constraints["absolute_min"])))
        above = int(np.sum(values > float(constraints["absolute_max"])))
        if below or above:
            hard.append(
                {
                    "id": "absolute_bounds",
                    "message": "Prediction exceeds the broad frozen target range.",
                    "below_count": below,
                    "above_count": above,
                }
            )
        elif "soft_absolute_min" in constraints and "soft_absolute_max" in constraints:
            soft_below = int(np.sum(values < float(constraints["soft_absolute_min"])))
            soft_above = int(np.sum(values > float(constraints["soft_absolute_max"])))
            if soft_below or soft_above:
                soft.append(
                    {
                        "id": "soft_absolute_envelope",
                        "message": "Prediction leaves the calibrated central target envelope.",
                        "below_count": soft_below,
                        "above_count": soft_above,
                        "lower_limit": float(constraints["soft_absolute_min"]),
                        "upper_limit": float(constraints["soft_absolute_max"]),
                    }
                )
        if len(values) > 1:
            max_step = float(np.max(np.abs(np.diff(values))))
            if max_step > float(constraints["max_abs_step"]):
                hard.append(
                    {
                        "id": "extreme_step",
                        "message": "Window contains an implausibly large adjacent step.",
                        "observed": max_step,
                        "limit": float(constraints["max_abs_step"]),
                    }
                )
            elif max_step > float(constraints.get("soft_abs_step", constraints["max_abs_step"])):
                soft.append(
                    {
                        "id": "abrupt_step",
                        "message": "Window contains an unusually large adjacent step.",
                        "observed": max_step,
                        "limit": float(constraints.get("soft_abs_step", constraints["max_abs_step"])),
                    }
                )
            min_window_range = float(constraints.get("min_window_range", 0.0))
            hard_min_window_range = float(
                constraints.get("hard_min_window_range", 0.0)
            )
            observed_range = float(np.ptp(values))
            if hard_min_window_range > 0.0 and observed_range < hard_min_window_range:
                hard.append(
                    {
                        "id": "severe_flat_window",
                        "message": "Prediction window is far flatter than any validation-front reference window.",
                        "observed": observed_range,
                        "limit": hard_min_window_range,
                    }
                )
            elif min_window_range > 0.0 and observed_range < min_window_range:
                soft.append(
                    {
                        "id": "flat_window",
                        "message": "Prediction window is unusually flat for the calibrated target dynamics.",
                        "observed": observed_range,
                        "limit": min_window_range,
                    }
                )
        if len(values) > 2 and "max_abs_curvature" in constraints:
            max_curvature = float(np.max(np.abs(np.diff(values, n=2))))
            if max_curvature > float(constraints["max_abs_curvature"]):
                hard.append(
                    {
                        "id": "extreme_curvature",
                        "message": "Window contains an implausibly sharp local reversal.",
                        "observed": max_curvature,
                        "limit": float(constraints["max_abs_curvature"]),
                    }
                )
            elif max_curvature > float(
                constraints.get("soft_abs_curvature", constraints["max_abs_curvature"])
            ):
                soft.append(
                    {
                        "id": "abrupt_curvature",
                        "message": "Window contains an unusually sharp local reversal.",
                        "observed": max_curvature,
                        "limit": float(
                            constraints.get("soft_abs_curvature", constraints["max_abs_curvature"])
                        ),
                    }
                )

    if reference is not None and reference.shape == values.shape and np.isfinite(values).all():
        delta = np.abs(values - reference)
        mean_delta = float(np.mean(delta))
        max_delta = float(np.max(delta))
        if max_delta > float(constraints["point_emergency_delta_limit"]):
            hard.append(
                {
                    "id": "point_emergency_delta",
                    "message": "A single-point correction exceeds the frozen emergency limit.",
                    "observed": max_delta,
                    "limit": float(constraints["point_emergency_delta_limit"]),
                }
            )
        if mean_delta > float(constraints["window_mean_abs_delta_limit"]):
            if require_adjustment_evidence and matched_rule_ids:
                soft.append(
                    {
                        "id": "large_supported_mean_delta",
                        "message": "Large mean correction is skill-supported but requires a second independent warning to trigger.",
                        "observed": mean_delta,
                        "limit": float(constraints["window_mean_abs_delta_limit"]),
                    }
                )
            else:
                hard.append(
                    {
                        "id": "mean_absolute_delta",
                        "message": "Mean absolute correction exceeds the frozen window limit.",
                        "observed": mean_delta,
                        "limit": float(constraints["window_mean_abs_delta_limit"]),
                    }
                )

    hard_ids = {item["id"] for item in hard}
    soft_ids = {item["id"] for item in soft}
    trigger = bool(hard_ids) or len(soft_ids) >= int(
        policy["trigger_policy"]["independent_soft_violation_count"]
    )
    return {
        "trigger": trigger,
        "hard_violations": hard,
        "soft_warnings": soft,
        "matched_rule_ids": sorted(set(matched_rule_ids or [])),
    }


def serializable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)) or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


EVENT_TEXT_COLUMN_MARKERS = (
    "holiday",
    "policy",
    "environment",
    "control",
    "event",
    "节假日",
    "政策",
    "环保",
    "管控",
    "特殊日",
)
WEATHER_TEXT_COLUMN_MARKERS = ("weather", "天气")


def provider_safe_text_value(column: str, value: Any) -> str:
    """Project free-form exogenous text into stable semantics for the LLM only."""
    raw = str(value).strip()
    lowered = raw.lower()
    if not raw or lowered in {"nan", "none", "null", "0", "0.0", "false"}:
        return "none"

    column_lower = str(column).lower()
    if any(marker in column_lower for marker in WEATHER_TEXT_COLUMN_MARKERS):
        weather_patterns = (
            (("雪", "snow"), "weather:snow"),
            (("雨", "rain", "drizzle", "storm"), "weather:rain_or_storm"),
            (("阴", "多云", "cloud", "overcast"), "weather:cloudy"),
            (("晴", "sun", "clear"), "weather:clear"),
            (("雾", "fog", "haze"), "weather:fog_or_haze"),
            (("风", "wind"), "weather:wind"),
        )
        for tokens, semantic in weather_patterns:
            if any(token in lowered for token in tokens):
                return semantic
        return "weather:other_present"

    if any(marker in column_lower for marker in EVENT_TEXT_COLUMN_MARKERS):
        event_patterns = (
            (("复工", "恢复", "resume", "recovery"), "event_phase:resume_or_recovery"),
            (("后第一天", "次日", "post-event", "day after"), "event_phase:post_event"),
            (("结束", "解除", "停止", "end", "release", "stop"), "event_phase:end_or_release"),
            (("开始", "启动", "实施", "管控", "召开", "发布", "start", "launch", "restrict"), "event_phase:start_or_restrict"),
        )
        for tokens, semantic in event_patterns:
            if any(token in lowered for token in tokens):
                return semantic
        return "event_phase:special_event_present"

    digest = hashlib.sha256(f"{column}\0{raw}".encode("utf-8")).hexdigest()[:12]
    return f"category:{digest}"


HOLIDAY_CONTEXT_COLUMN_MARKERS = ("holiday", "calendar", "\u8282\u5047\u65e5")
POLICY_CONTEXT_COLUMN_MARKERS = ("policy", "\u653f\u7b56")
CONTROL_CONTEXT_COLUMN_MARKERS = (
    "environment",
    "control",
    "\u73af\u4fdd",
    "\u7ba1\u63a7",
)
SUNSHINE_CONTEXT_COLUMN_MARKERS = ("sunshine", "\u65e5\u7167")
WIND_CONTEXT_COLUMN_MARKERS = ("wind", "\u98ce\u5411\u98ce\u529b")
BOOLEAN_CONTEXT_COLUMN_MARKERS = (
    "flag",
    "is_",
    "\u662f\u5426\u6781\u7aef\u5929\u6c14",
)


def _contains_any(value: str, tokens: tuple[str, ...]) -> bool:
    return any(token in value for token in tokens)


def provider_safe_text_context(column: str, value: Any) -> dict[str, Any]:
    """Return compact, provider-safe semantics without exposing free-form text."""
    raw = str(value).strip()
    lowered = raw.lower()
    column_lower = str(column).lower()
    semantic = provider_safe_text_value(column, value)
    context: dict[str, Any] = {"semantic": semantic}

    if _contains_any(column_lower, BOOLEAN_CONTEXT_COLUMN_MARKERS):
        context["boolean"] = lowered in {"1", "1.0", "true", "yes", "y"}
        return context

    if _contains_any(column_lower, SUNSHINE_CONTEXT_COLUMN_MARKERS):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
        if match:
            context["hours"] = float(match.group(0))
        return context

    if semantic == "none":
        return context

    if _contains_any(column_lower, WIND_CONTEXT_COLUMN_MARKERS):
        direction_patterns = (
            (("\u4e1c\u5317", "northeast"), "northeast"),
            (("\u897f\u5317", "northwest"), "northwest"),
            (("\u4e1c\u5357", "southeast"), "southeast"),
            (("\u897f\u5357", "southwest"), "southwest"),
            (("\u5317", "north"), "north"),
            (("\u5357", "south"), "south"),
            (("\u4e1c", "east"), "east"),
            (("\u897f", "west"), "west"),
        )
        for tokens, label in direction_patterns:
            if _contains_any(lowered, tokens):
                context["direction"] = label
                break
        level_match = re.search(r"(\d+)\s*(?:\u7ea7|level)", lowered)
        if level_match:
            context["level"] = int(level_match.group(1))
        return context

    event_column = _contains_any(
        column_lower,
        (
            *HOLIDAY_CONTEXT_COLUMN_MARKERS,
            *POLICY_CONTEXT_COLUMN_MARKERS,
            *CONTROL_CONTEXT_COLUMN_MARKERS,
        ),
    )
    if event_column:
        if _contains_any(column_lower, HOLIDAY_CONTEXT_COLUMN_MARKERS):
            context["domain"] = "calendar_or_holiday"
        elif _contains_any(column_lower, POLICY_CONTEXT_COLUMN_MARKERS):
            context["domain"] = "policy"
        else:
            context["domain"] = "environmental_control"

        identity_patterns = (
            (("\u5143\u65e6", "new year"), "new_year"),
            (("\u6625\u8282", "spring festival"), "spring_festival"),
            (("\u4e94\u4e00", "labor day"), "labor_day"),
            (("\u7aef\u5348", "dragon boat"), "dragon_boat"),
            (("\u56fd\u5e86", "national day"), "national_day"),
            (("\u5468\u516d", "saturday"), "saturday"),
            (("\u5468\u65e5", "sunday"), "sunday"),
        )
        for tokens, label in identity_patterns:
            if _contains_any(lowered, tokens):
                context["event_type"] = label
                break

        topic_patterns = (
            (("\u4f9b\u6696", "heating"), "heating"),
            (("\u9632\u6c5b", "flood"), "flood_response"),
            (("\u91cd\u6c61\u67d3", "\u73af\u4fdd", "pollution"), "pollution_control"),
            (("\u4f11\u6e14", "fishing"), "fishing_restriction"),
            (("\u4e24\u4f1a", "conference"), "major_conference"),
            (("\u9605\u5175", "parade"), "major_public_event"),
            (("\u7528\u7535", "electricity"), "electricity_policy"),
        )
        topics = [
            label
            for tokens, label in topic_patterns
            if _contains_any(lowered, tokens)
        ]
        if topics:
            context["topics"] = topics
        return context

    return context


def provider_safe_condition(condition: dict[str, Any]) -> dict[str, Any]:
    projected = dict(condition)
    kind = projected.get("kind")
    if kind in {"all", "any"}:
        projected["conditions"] = [
            provider_safe_condition(child) for child in projected.get("conditions", [])
        ]
    elif kind in {"exogenous_text_in", "exogenous_text_contains"}:
        column = str(projected.get("column", "text_exogenous"))
        projected["values"] = list(
            dict.fromkeys(
                provider_safe_text_value(column, value)
                for value in projected.get("values", [])
            )
        )
        projected["value_representation"] = "provider_safe_semantic_projection"
    return projected


def provider_safe_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(rule["id"]),
        "condition": provider_safe_condition(rule.get("condition", {})),
        "action": rule.get("action"),
        "bounds": rule.get("bounds"),
        "matched_horizon_indices": [
            int(index) for index in rule.get("matched_horizon_indices", [])
        ],
        "guidance_role": (
            "Accepted expert-derived guidance. The original rule was matched deterministically before this "
            "provider-safe projection; use its action, scope, and bounds as important evidence rather than a rigid point command."
        ),
    }


def nested_forbidden_keys(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_CONTEXT_KEYS or lowered.startswith(tuple(f"{k}_" for k in FORBIDDEN_CONTEXT_KEYS)):
                found.add(str(key))
            found.update(nested_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(nested_forbidden_keys(child))
    return sorted(found)


def summarize_series(values: np.ndarray) -> dict[str, float | int | str | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "start": None, "end": None, "trend": "unknown"}
    slope = float(np.polyfit(np.arange(finite.size, dtype=np.float64), finite, 1)[0]) if finite.size > 1 else 0.0
    tolerance = numeric_tolerance(finite)
    trend = "upward" if slope > tolerance else "downward" if slope < -tolerance else "stable"
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "start": float(finite[0]),
        "end": float(finite[-1]),
        "trend": trend,
    }


def summarize_exogenous(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    text_columns: list[str],
) -> dict[str, Any]:
    numeric: dict[str, Any] = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        numeric[column] = summarize_series(values)
    text: dict[str, list[str]] = {}
    for column in text_columns:
        values = [str(value).strip() for value in frame[column].dropna().tolist() if str(value).strip()]
        text[column] = list(
            dict.fromkeys(
                provider_safe_text_value(column, value) for value in values
            )
        )[:20]
    return {"numeric": numeric, "text_values": text}



def positional_table(
    frame: pd.DataFrame,
    *,
    time_column: str,
    value_columns: list[str],
    timestamp_label: str = "timestamp",
    text_columns: set[str] | None = None,
) -> dict[str, Any]:
    columns = [timestamp_label, *value_columns]
    projected_text_columns = text_columns or set()
    rows = [
        [
            str(row[time_column]),
            *(
                provider_safe_text_value(column, row[column])
                if column in projected_text_columns
                else serializable(row[column])
                for column in value_columns
            ),
        ]
        for _, row in frame.iterrows()
    ]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "encoding": "Each row is positional and must be read against columns by index.",
    }


def history_target_table(
    frame: pd.DataFrame,
    *,
    time_column: str,
    target_column: str,
) -> dict[str, Any]:
    target_values = pd.to_numeric(frame[target_column], errors="coerce")
    rows = [
        [str(timestamp), serializable(value)]
        for timestamp, value in zip(frame[time_column], target_values, strict=True)
    ]
    return {
        "columns": ["timestamp", "observed_value"],
        "rows": rows,
        "row_count": len(rows),
        "encoding": "Each row is positional and must be read against columns by index.",
    }


def recent_three_day_frame(
    frame: pd.DataFrame,
    *,
    time_column: str,
    first_target: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, str]]:
    target_day = first_target.normalize()
    yesterday = target_day - pd.Timedelta(days=2)
    forecast_day = target_day - pd.Timedelta(days=1)
    end = target_day + pd.Timedelta(days=1)
    timestamps = pd.to_datetime(frame[time_column], errors="raise")
    selected = frame.loc[(timestamps >= yesterday) & (timestamps < end)].copy()
    return selected, {
        "yesterday": yesterday.strftime("%Y-%m-%d"),
        "forecast_issue_day_today": forecast_day.strftime("%Y-%m-%d"),
        "target_day_tomorrow": target_day.strftime("%Y-%m-%d"),
    }


def daily_constant_exogenous_table(
    frame: pd.DataFrame,
    *,
    time_column: str,
    exogenous_columns: list[str],
    text_columns: set[str] | None = None,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "columns": ["date"],
            "rows": [],
            "row_count": 0,
            "encoding": "Each row is positional and must be read against columns by index.",
        }
    dates = pd.to_datetime(frame[time_column], errors="raise").dt.strftime("%Y-%m-%d")
    daily_columns = [
        column
        for column in exogenous_columns
        if bool((frame.groupby(dates, sort=False)[column].nunique(dropna=False) <= 1).all())
    ]
    rows: list[list[Any]] = []
    projected_text_columns = text_columns or set()
    for date, group in frame.groupby(dates, sort=False):
        rows.append(
            [
                str(date),
                *(
                    provider_safe_text_value(column, group.iloc[0][column])
                    if column in projected_text_columns
                    else serializable(group.iloc[0][column])
                    for column in daily_columns
                ),
            ]
        )
    return {
        "columns": ["date", *daily_columns],
        "rows": rows,
        "row_count": len(rows),
        "encoding": (
            "Only exogenous columns that are constant within every represented calendar day are included; "
            "daily deduplication is lossless for this window."
        ),
    }


def condition_source_columns(condition: dict[str, Any]) -> list[str]:
    kind = condition.get("kind")
    if kind in {"all", "any"}:
        columns: list[str] = []
        for child in condition.get("conditions", []):
            columns.extend(condition_source_columns(child))
        return list(dict.fromkeys(columns))
    if kind in {
        "exogenous_numeric_between",
        "exogenous_text_in",
        "exogenous_text_contains",
        "exogenous_bool_is",
    }:
        return [str(condition["column"])]
    return []


def _day_exogenous_profile(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str],
    text_columns: list[str],
) -> dict[str, Any]:
    numeric: dict[str, Any] = {}
    numeric_means: dict[str, float] = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if not finite.size:
            numeric[column] = None
            continue
        numeric_means[column] = float(np.mean(finite))
        if float(np.max(finite) - np.min(finite)) <= 1e-12:
            numeric[column] = {"value": float(finite[0])}
        else:
            numeric[column] = summarize_series(finite)

    text: dict[str, list[dict[str, Any]]] = {}
    for column in text_columns:
        contexts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in frame[column].tolist():
            context = provider_safe_text_context(column, value)
            serialized = json.dumps(context, sort_keys=True, ensure_ascii=True)
            if serialized not in seen:
                seen.add(serialized)
                contexts.append(context)
        text[column] = contexts
    return {
        "numeric": numeric,
        "numeric_means": numeric_means,
        "text": text,
    }


def current_sample_exogenous_context(
    frame: pd.DataFrame,
    *,
    time_column: str,
    numeric_columns: list[str],
    text_columns: list[str],
    date_roles: dict[str, str],
    matching_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the current sample before historical analogue reasoning."""
    timestamps = pd.to_datetime(frame[time_column], errors="raise")
    all_columns = list(dict.fromkeys([*numeric_columns, *text_columns]))
    primary_columns = list(
        dict.fromkeys(
            column
            for rule in matching_rules
            for column in condition_source_columns(rule.get("condition", {}))
            if column in all_columns
        )
    )
    profiles: dict[str, Any] = {}
    means_by_role: dict[str, dict[str, float]] = {}
    for role, date in date_roles.items():
        selected = frame.loc[timestamps.dt.strftime("%Y-%m-%d") == date]
        profile = _day_exogenous_profile(
            selected,
            numeric_columns=numeric_columns,
            text_columns=text_columns,
        )
        means_by_role[role] = profile.pop("numeric_means")
        profiles[role] = {"date": date, **profile}

    target_role = "target_day_tomorrow"
    numeric_mean_changes: dict[str, dict[str, float]] = {}
    for source_role in ("forecast_issue_day_today", "yesterday"):
        changes: dict[str, float] = {}
        target_means = means_by_role.get(target_role, {})
        source_means = means_by_role.get(source_role, {})
        for column in numeric_columns:
            if column in target_means and column in source_means:
                changes[column] = float(target_means[column] - source_means[column])
        numeric_mean_changes[f"{target_role}_minus_{source_role}"] = changes

    return {
        "exogenous_column_count": len(all_columns),
        "all_exogenous_columns": all_columns,
        "skill_primary_columns": primary_columns,
        "secondary_context_columns": [
            column for column in all_columns if column not in primary_columns
        ],
        "day_profiles": profiles,
        "numeric_mean_changes": numeric_mean_changes,
        "reasoning_contract": (
            "The target-day joint exogenous context is the primary context. Skill columns are prioritized but never exclusive. "
            "Review every secondary context column for supporting, opposing, interaction, or neutral evidence. Treat duplicated or "
            "derived measurements of the same physical factor as one evidence family, not independent votes."
        ),
    }


def condition_exogenous_requests(
    condition: dict[str, Any],
) -> list[tuple[str, tuple[str, ...]]]:
    kind = condition.get("kind")
    if kind in {"all", "any"}:
        combined: dict[str, tuple[str, ...]] = {}
        for child in condition.get("conditions", []):
            for label, columns in condition_exogenous_requests(child):
                combined[label] = columns
        return list(combined.items())
    if kind not in {
        "exogenous_numeric_between",
        "exogenous_text_in",
        "exogenous_text_contains",
        "exogenous_bool_is",
    }:
        return []
    column = str(condition["column"])
    day_offset = int(condition.get("day_offset", 0) or 0)
    label = f"{column} [day_offset={day_offset:+d}]"
    numeric = exog_numeric_column(column, day_offset)
    text = exog_text_column(column, day_offset)
    if kind == "exogenous_bool_is":
        return [(label, (text, numeric))]
    return [(label, (numeric,))] if kind == "exogenous_numeric_between" else [(label, (text,))]


def rule_match_evidence_rows(
    feature_rows: pd.DataFrame,
    matching_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    horizon_values = pd.to_numeric(feature_rows["horizon_index"], errors="raise").astype(int)
    for rule in matching_rules:
        requests = condition_exogenous_requests(rule.get("condition", {}))
        for horizon in rule.get("matched_horizon_indices", []):
            selected = feature_rows.loc[horizon_values == int(horizon)]
            if len(selected) != 1:
                raise ValueError(
                    f"Rule {rule['id']} horizon {horizon} must map to exactly one deterministic feature row"
                )
            row = selected.iloc[0]
            timestamp = pd.to_datetime(row["timestamp"], errors="raise")
            exogenous_values: dict[str, Any] = {}
            for label, candidates in requests:
                source_column = next((column for column in candidates if column in row.index), None)
                if source_column is None:
                    raise ValueError(
                        f"Rule {rule['id']} requires unavailable deterministic feature {label}"
                    )
                value = row[source_column]
                exogenous_values[label] = (
                    provider_safe_text_value(label, value)
                    if source_column.startswith("exog_text__")
                    else serializable(value)
                )
            evidence.append(
                {
                    "rule_id": str(rule["id"]),
                    "horizon_index": int(horizon),
                    "timestamp": str(row["timestamp"]),
                    "calendar": {
                        "month": int(timestamp.month),
                        "weekday": timestamp.day_name(),
                        "hour": int(timestamp.hour),
                        "minute": int(timestamp.minute),
                    },
                    "raw_model_prediction": float(row["raw_predicted_original"]),
                    "condition_exogenous_values": exogenous_values,
                }
            )
    return evidence


PROVIDER_SAFE_SCENARIO_EVENT_FIELDS = {"holiday_effects", "anomaly_periods"}
MAX_PROVIDER_SCENARIO_CHARS = 2000

def task_scenario_overview(task: dict[str, Any]) -> dict[str, Any]:
    overview = {
        "target_column": str(task["target_col"]),
        "time_column": str(task["time_col"]),
        "history_window_steps": int(task["seq_len"]),
        "forecast_gap_steps": int(task.get("forecast_gap", 0)),
        "prediction_window_steps": int(task["pred_len"]),
        "window_stride_steps": int(task.get("window_stride", 1)),
    }
    optional_fields = (
        "business_context",
        "csv_semantics",
        "region_info",
        "seasonality_hints",
        "holiday_effects",
        "anomaly_periods",
    )
    for field in optional_fields:
        value = str(task.get(field, "")).strip()
        if value:
            if field in PROVIDER_SAFE_SCENARIO_EVENT_FIELDS:
                overview[field] = {
                    "present": True,
                    "semantic_role": field,
                    "source_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
                    "note": (
                        "Detailed free-form event text is withheld from the provider prompt; "
                        "matched skill semantics and provider-safe exogenous tags carry the usable evidence."
                    ),
                }
            else:
                overview[field] = value[:MAX_PROVIDER_SCENARIO_CHARS]
    return overview


def task_knowledge_overview(task: dict[str, Any], task_path: Path) -> dict[str, Any]:
    """Return stable task semantics without time-local events or learned rules."""
    overview = {
        "target_column": str(task["target_col"]),
        "time_column": str(task["time_col"]),
        "history_window_steps": int(task["seq_len"]),
        "forecast_gap_steps": int(task.get("forecast_gap", 0)),
        "prediction_window_steps": int(task["pred_len"]),
        "window_stride_steps": int(task.get("window_stride", 1)),
    }
    for field in ("business_context", "csv_semantics", "region_info", "seasonality_hints"):
        value = str(task.get(field, "")).strip()
        if value:
            overview[field] = value[:MAX_PROVIDER_SCENARIO_CHARS]
    overview["data_analysis"] = load_data_analysis_knowledge(task, task_path)
    return overview


def task_context_overview(task: dict[str, Any]) -> dict[str, Any]:
    """Return task-declared event state without stable semantics or expert rules."""
    overview: dict[str, Any] = {}
    for field in ("holiday_effects", "anomaly_periods"):
        value = str(task.get(field, "")).strip()
        if value:
            overview[field] = {
                "present": True,
                "semantic_role": field,
                "source_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
                "note": (
                    "Detailed free-form event text is withheld from the provider prompt; "
                    "provider-safe sample values carry the usable context evidence."
                ),
            }
    return overview


def evidence_candidate_bundle(
    task: dict[str, Any],
    task_path: Path,
    sample_context: dict[str, Any],
    lookback_window_exogenous: dict[str, Any],
    predict_window_exogenous: dict[str, Any],
    experience: dict[str, Any],
) -> dict[str, Any]:
    constraints = experience["non_negotiable_constraints"]["frozen_physical_safety"]
    experience_payload = {
        "candidate_id": experience["candidate_id"],
        "important_guidance_rules": experience["important_guidance_rules"],
        "guidance_contract": experience["guidance_contract"],
    }
    bundle = {
        "classification_contract": EVIDENCE_CANDIDATE_BOUNDARIES,
        "context": {
            "candidate_type": "context",
            "evidence_candidate_id": "context:current-window",
            "definition": EVIDENCE_CANDIDATE_BOUNDARIES["context"],
            "scope": "current forecast window only",
            "payload": {
                "declared_events": task_context_overview(task),
                "current_sample_exogenous": sample_context,
                "lookback_window_exogenous": lookback_window_exogenous,
                "predict_window_exogenous": predict_window_exogenous,
                "table_contract": (
                    "These are the complete row-level exogenous values for the exact lookback and prediction windows. "
                    "Resolve positional rows through their columns arrays; summaries are auxiliary and never replace these tables."
                ),
            },
        },
        "knowledge": {
            "candidate_type": "knowledge",
            "evidence_candidate_id": "knowledge:frozen-task",
            "definition": EVIDENCE_CANDIDATE_BOUNDARIES["knowledge"],
            "scope": "task-wide stable contract",
            "payload": {
                "task_semantics_and_structure": task_knowledge_overview(task, task_path),
                "hard_physical_constraints": constraints,
                "information_boundary": experience["non_negotiable_constraints"]["information_boundary"],
                "output_contract": experience["non_negotiable_constraints"]["output_contract"],
            },
        },
    }
    if experience["candidate_id"] and experience["important_guidance_rules"]:
        bundle["experience"] = {
            "candidate_type": "experience",
            "evidence_candidate_id": f"experience:{experience['candidate_id']}",
            "definition": EVIDENCE_CANDIDATE_BOUNDARIES["experience"],
            "scope": "reusable only under the rule's stated and validated conditions",
            "payload": experience_payload,
        }
    return bundle


def initial_critic_prediction_state(
    window_index: int,
    prediction_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the immutable initial Critic prediction state; later transitions are intentionally separate."""
    return {
        "version": 1,
        "window_index": int(window_index),
        "evidences": [],
        "predictions": [
            {
                "horizon_index": int(point["horizon_index"]),
                "timestamp": str(point["timestamp"]),
                "value": float(point["raw_anchor"]),
            }
            for point in prediction_points
        ],
        "evidence_gap": 1,
        "needed_evidence": "",
    }


def experience_rule_reference(
    initial_state: dict[str, Any],
    prediction_points: list[dict[str, Any]],
    experience_candidate: dict[str, Any],
    experience_artifact: dict[str, str],
) -> dict[str, Any]:
    """Record how the locked rules act on the immutable initial prediction.

    This is deliberately not a Critic state.  It is a reference forecast for the
    ExperienceReasoner, which must reason from the current state and may preserve
    it unchanged when the rule effect is already represented by committed evidence.
    """
    return {
        "version": 1,
        "window_index": int(initial_state["window_index"]),
        "semantics": "locked_rules_applied_to_initial_prediction_only",
        "experience_candidate_id": str(experience_candidate["evidence_candidate_id"]),
        "experience_candidate_artifact": {
            "path": str(experience_artifact["path"]),
            "sha256": str(experience_artifact["sha256"]),
        },
        "initial_predictions": initial_state["predictions"],
        "predictions": [
            {
                "horizon_index": int(point["horizon_index"]),
                "timestamp": str(point["timestamp"]),
                "value": float(point["pre_reflection_baseline"]),
            }
            for point in prediction_points
        ],
        "reasoning_contract": {
            "reference_only": True,
            "not_a_state_transition": True,
            "do_not_add_reference_delta_to_current": True,
        },
    }


def locked_skill_experience(
    candidate_id: str | None,
    matching_rules: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    constraints = policy["constraints"]
    return {
        "candidate_id": candidate_id,
        "non_negotiable_constraints": {
            "rule_identity": "Only the listed locked rule ids may be cited; do not invent or rewrite a rule.",
            "information_boundary": "Use only this sample context; never infer from observed target values in the gap or prediction window.",
            "output_contract": "Return each required horizon index and exact timestamp once with a finite value.",
            "frozen_physical_safety": {
                "nonnegative": bool(constraints.get("nonnegative", False)),
                "absolute_min": float(constraints["absolute_min"]),
                "absolute_max": float(constraints["absolute_max"]),
                "max_abs_step": float(constraints["max_abs_step"]),
                "max_abs_curvature": float(constraints.get("max_abs_curvature", constraints["max_abs_step"])),
                "hard_min_window_range": float(constraints.get("hard_min_window_range", 0.0)),
            },
        },
        "important_guidance_rules": [provider_safe_rule(rule) for rule in matching_rules],
        "guidance_contract": (
            "The accepted expert-derived skill is one important evidence source for this sample. "
            "Explicitly consider its matched conditions, action direction, magnitude, time scope, bounds, and initial-based rule reference. "
            "The rule reference applies all matching rules only to the immutable initial prediction; it is not the current state and its "
            "delta must never be mechanically added to or substituted for the current prediction. "
            "Its combined result is important reference evidence rather than a privileged target. Compare it with the current and initial "
            "predictions and committed evidence, then generate only the residual effect not already represented in the current state. "
            "Preserving the current prediction unchanged is valid when no residual numerical effect is justified."
        ),
    }

def build_window_context(
    task: dict[str, Any],
    task_path: Path,
    controller_version: str,
    dataset: pd.DataFrame,
    window_rows: pd.DataFrame,
    raw_values: np.ndarray,
    baseline_values: np.ndarray,
    detection: dict[str, Any],
    matching_rules: list[dict[str, Any]],
    *,
    reasoning_mode: str = "safety_reflection",
    gap_prediction_rows: pd.DataFrame | None = None,
    condition_feature_rows: pd.DataFrame | None = None,
    candidate_id: str | None = None,
    policy: dict[str, Any] | None = None,
    active_recalibration_priority: bool = False,
    matched_window_ordinal: int | None = None,
    minimum_meaningful_delta_original: float = 0.0,
    minimum_meaningful_points: int = 0,
) -> dict[str, Any]:
    time_col = str(task["time_col"])
    target_col = str(task["target_col"])
    seq_len = int(task["seq_len"])
    forecast_gap = int(task.get("forecast_gap", 0))
    timestamps = pd.to_datetime(dataset[time_col], errors="raise")
    first_target = pd.to_datetime(window_rows.iloc[0]["timestamp"], errors="raise")
    matches = np.flatnonzero(timestamps.eq(first_target).to_numpy())
    if len(matches) != 1:
        raise ValueError(f"Target timestamp {first_target} must map to exactly one dataset row")
    target_start = int(matches[0])
    history_end = target_start - forecast_gap
    history_start = history_end - seq_len
    if history_start < 0:
        raise ValueError("Window history would start before the dataset")

    numeric_exog = safe_exogenous_columns(task, dataset)
    text_exog = safe_text_exogenous_columns(task, dataset)
    exog_columns = list(dict.fromkeys([*numeric_exog, *text_exog]))

    def exog_payload(row: pd.Series) -> dict[str, Any]:
        return {column: serializable(row[column]) for column in exog_columns}

    history_frame = dataset.iloc[history_start:history_end]
    exogenous_history_steps = int(task.get("exogenous_history_steps", seq_len))
    if not 0 < exogenous_history_steps <= seq_len:
        raise ValueError("exogenous_history_steps must be between 1 and seq_len")
    exogenous_start = max(history_start, history_end-exogenous_history_steps)
    exogenous_history_frame = dataset.iloc[exogenous_start:history_end]
    history = []
    for position, (_, row) in enumerate(history_frame.iterrows(), start=history_start):
        history.append(
            {
                "timestamp": str(row[time_col]),
                "observed_value": serializable(pd.to_numeric(pd.Series([row[target_col]]), errors="coerce").iloc[0]),
                "known_exogenous": exog_payload(row) if position >= exogenous_start else {},
            }
        )
    future_end = target_start + len(window_rows)
    known_frame = dataset.iloc[history_end:future_end]
    future_known = []
    for _, row in known_frame.iterrows():
        future_known.append({"timestamp": str(row[time_col]), "known_exogenous": exog_payload(row)})

    points = []
    for row_offset, (_, row) in enumerate(window_rows.iterrows()):
        points.append(
            {
                "horizon_index": int(row["horizon_index"]),
                "timestamp": str(row["timestamp"]),
                "raw_anchor": float(raw_values[row_offset]),
                "pre_reflection_baseline": float(baseline_values[row_offset]),
            }
        )
    context = {
        "controller_version": controller_version,
        "window_index": int(window_rows.iloc[0]["window_index"]),
        "target_unit": "original_dataset_unit",
        "history_rows": history,
        "gap_and_future_known_rows": future_known,
        "prediction_points": points,
        "detector_report": detection,
        "supporting_skill_rules": matching_rules,
        "instructions": {
            "objective": "Make the smallest correction needed to remove the listed obvious violations.",
            "never_use": "Future observed target values, residuals, evaluation metrics, later windows, or hidden context.",
            "required_output": "Return every horizon_index and exact timestamp once, in original dataset units.",
        },
    }

    if reasoning_mode in {"skill_guided_test", "evidence_loop_test"}:
        if policy is None:
            raise ValueError("Skill-guided test reasoning requires the frozen reflection policy")
        if condition_feature_rows is None:
            raise ValueError("Skill-guided test reasoning requires deterministic condition feature rows")
        gap_points: list[dict[str, Any]] = []
        if forecast_gap > 0:
            if gap_prediction_rows is None or len(gap_prediction_rows) != forecast_gap:
                raise ValueError(
                    f"Skill-guided test window requires exactly {forecast_gap} forecast-gap predictions"
                )
            ordered_gap = gap_prediction_rows.sort_values("gap_index")
            if ordered_gap["gap_index"].astype(int).tolist() != list(range(forecast_gap)):
                raise ValueError("Forecast-gap predictions must contain contiguous gap_index values")
            expected_gap_timestamps = pd.to_datetime(
                dataset.iloc[history_end:target_start][time_col], errors="raise"
            ).reset_index(drop=True)
            supplied_gap_timestamps = pd.to_datetime(
                ordered_gap["timestamp"], errors="raise"
            ).reset_index(drop=True)
            if not expected_gap_timestamps.equals(supplied_gap_timestamps):
                raise ValueError("Forecast-gap prediction timestamps do not align with this sample")
            gap_points = [
                {
                    "gap_index": int(row["gap_index"]),
                    "timestamp": str(row["timestamp"]),
                    "intermediate_prediction": float(row["predicted_original"]),
                }
                for _, row in ordered_gap.iterrows()
            ]

        history_values = pd.to_numeric(history_frame[target_col], errors="coerce").to_numpy(dtype=np.float64)
        gap_values = np.asarray(
            [point["intermediate_prediction"] for point in gap_points], dtype=np.float64
        )
        adjustment_delta = baseline_values - raw_values
        full_sample_frame = dataset.iloc[exogenous_start:future_end]
        recent_exogenous_frame, date_roles = recent_three_day_frame(
            full_sample_frame,
            time_column=time_col,
            first_target=first_target,
        )
        if task.get("exogenous_history_steps") is not None:
            # Keep exactly the configured recent history plus all future-known gap/horizon rows.
            recent_exogenous_frame = full_sample_frame
            for day in pd.to_datetime(full_sample_frame[time_col]).dt.normalize().unique():
                date = pd.Timestamp(day).strftime("%Y-%m-%d")
                if date not in date_roles.values():
                    offset = (first_target.normalize()-pd.Timestamp(day)).days
                    date_roles[f"history_day_minus_{offset}"] = date
        context.pop("supporting_skill_rules", None)
        context.pop("history_rows", None)
        context.pop("gap_and_future_known_rows", None)
        sample_context = current_sample_exogenous_context(
            recent_exogenous_frame,
            time_column=time_col,
            numeric_columns=numeric_exog,
            text_columns=text_exog,
            date_roles=date_roles,
            matching_rules=matching_rules,
        )
        lookback_window_exogenous = positional_table(
            exogenous_history_frame,
            time_column=time_col,
            value_columns=exog_columns,
            text_columns=set(text_exog),
        )
        predict_window_exogenous = positional_table(
            dataset.iloc[target_start:future_end],
            time_column=time_col,
            value_columns=exog_columns,
            text_columns=set(text_exog),
        )
        skill_experience = locked_skill_experience(
            candidate_id,
            matching_rules,
            policy,
        )
        context.update(
            {
                "reasoning_mode": reasoning_mode,
                "active_recalibration": {
                    "priority": bool(active_recalibration_priority),
                    "minimum_share_of_skill_matched_windows": SKILL_GUIDED_ACTIVE_RECALIBRATION_SHARE,
                    "matched_window_ordinal": matched_window_ordinal,
                    "minimum_meaningful_delta_original": float(minimum_meaningful_delta_original),
                    "minimum_meaningful_points": int(minimum_meaningful_points),
                    "selection_basis": (
                        "Leakage-free chronological coverage among skill-matched test windows; no target observations, "
                        "residuals, or test metrics are used."
                    ),
                },
                "task_and_scenario": task_scenario_overview(task),
                "context_representation": {
                    "exogenous_history_rows": len(exogenous_history_frame),
                    "visible_target_history_rows": len(history_frame),
                    "exogenous_columns": exog_columns,
                    "table_contract": (
                        "All table rows are positional arrays. Resolve every value through the corresponding "
                        "columns array; never infer or reorder columns."
                    ),
                    "recent_exogenous_date_roles": date_roles,
                    "recent_exogenous_scope": (
                        "Configured recent exogenous history plus forecast-gap and prediction-horizon exogenous rows."
                        if task.get("exogenous_history_steps") is not None else
                        "Full-resolution exogenous values for yesterday, forecast issue day today, and target day tomorrow."
                    ),
                    "full_sample_scope": (
                        "The complete visible history, forecast gap, and target horizon; target observations remain hidden "
                        "outside the visible history."
                    ),
                },
                "current_sample_exogenous_context": sample_context,
                "visible_history_target_table": history_target_table(
                    history_frame,
                    time_column=time_col,
                    target_column=target_col,
                ),
                "lookback_window_exogenous_table": lookback_window_exogenous,
                "predict_window_exogenous_table": predict_window_exogenous,
                "recent_three_day_exogenous_table": positional_table(
                    recent_exogenous_frame,
                    time_column=time_col,
                    value_columns=exog_columns,
                    text_columns=set(text_exog),
                ),
                "full_sample_daily_constant_exogenous_table": daily_constant_exogenous_table(
                    full_sample_frame,
                    time_column=time_col,
                    exogenous_columns=exog_columns,
                    text_columns=set(text_exog),
                ),
                "gap_prediction_points": gap_points,
                "deterministic_sample_features": {
                    "target_window_start": str(window_rows.iloc[0]["timestamp"]),
                    "target_window_end": str(window_rows.iloc[-1]["timestamp"]),
                    "target_start_weekday": first_target.day_name(),
                    "history_target_summary": summarize_series(history_values),
                    "forecast_gap_prediction_summary": summarize_series(gap_values),
                    "raw_target_prediction_summary": summarize_series(raw_values),
                    "deterministic_skill_prediction_summary": summarize_series(baseline_values),
                    "deterministic_adjustment_summary": summarize_series(adjustment_delta),
                    "full_sample_exogenous_summary": summarize_exogenous(
                        full_sample_frame,
                        numeric_exog,
                        text_exog,
                    ),
                    "rule_match_evidence_rows": rule_match_evidence_rows(
                        condition_feature_rows,
                        matching_rules,
                    ),
                    "matched_rule_ids": [str(rule["id"]) for rule in matching_rules],
                },
                "locked_skill_experience": skill_experience,
                "evidence_candidates": evidence_candidate_bundle(
                    task,
                    task_path,
                    sample_context,
                    lookback_window_exogenous,
                    predict_window_exogenous,
                    skill_experience,
                ),
                "provider_safe_projection": {
                    "enabled": True,
                    "scope": "llm_prompt_only",
                    "deterministic_matching_uses_original_values": True,
                    "free_form_text_values_are_semantic_tags": True,
                },
            }
        )
        context["instructions"] = {
            "objective": (
                "Improve expected forecast accuracy for this sample under the accepted skill; do not optimize merely to clear a detector "
                "threshold or to demonstrate that a modification was made. Compare the raw anchor, deterministic baseline, visible history, "
                "forecast-gap trajectory, target forecast, and exogenous context before choosing the final shape and magnitude."
            ),
            "guidance_priority": (
                "The locked expert-derived skill is an important correction reference and must be considered explicitly, but its deterministic "
                "result is not the default answer. Its direction, magnitude, bounds, and time scope remain strong evidence rather than immutable "
                "commands; use all other visible sample evidence actively, and do not invent or rewrite rules."
            ),
            "baseline_semantics": (
                "pre_reflection_baseline already contains the deterministic result of every matching rule. Never apply, add, subtract, sum, "
                "net, or stack those rule actions again. If it equals raw_anchor despite matches, treat the equality as an already resolved "
                "cancellation or overlap, not as an unadjusted forecast. It is a comparison anchor and failure fallback, not a privileged baseline "
                "for a token-sized follow-up edit."
            ),
            "decision_process": (
                "Start with current_sample_exogenous_context, then compare the raw anchor, deterministic baseline, and a context-derived profile. "
                "First form a joint current-sample residual hypothesis, then calibrate morning-ramp, midday, evening-peak, and late-day segments. "
                "Moving farther than the deterministic baseline or reversing its direction requires at least two independent evidence families, "
                "including target-day joint exogenous changes or forecast-gap dynamics. A semantic trigger and its matching skill are one family."
            ),
            "exogenous_joint_reasoning": (
                "Skill-primary columns identify the main hypothesis but are never the whole context. Review every secondary_context_column and group "
                "its role as supporting, opposing, interacting, or neutral. Use weather trajectory, temperature level and change, humidity, radiation, "
                "daily weather, wind, sunshine, extreme and overlap flags, environmental control, policy, and calendar information together when present. "
                "Do not count duplicated or derived measurements of the same physical driver as separate votes. Distinguish the total scenario effect "
                "from the residual effect that the raw forecast has not already encoded."
            ),
            "active_recalibration": (
                (
                    "This window is in the leakage-free active-recalibration subset. Build a genuinely sample-specific alternative instead of copying "
                    "the deterministic skill curve or making a token-sized edit. Re-estimate both direction and magnitude from the visible sample "
                    "joint exogenous context and forecast dynamics; the skill explains why the window deserves attention but does not supply final "
                    "point values. When evidence is mixed, prefer the smallest coherent localized change that meets the contract. The accepted proposal must "
                    f"differ from the deterministic baseline by at least {minimum_meaningful_delta_original:.12g} on at least "
                    f"{minimum_meaningful_points} points while remaining smooth and within frozen safety checks."
                )
                if active_recalibration_priority
                else (
                    "This window is not in the minimum active-recalibration subset, but still compare all three candidates independently. Keep an "
                    "anchor only when it is genuinely best supported, not merely because it already exists."
                )
            ),
            "multi_rule_semantics": (
                "The deterministic baseline may legitimately contain combined effects from multiple matched rules. Do not reconstruct those actions, "
                "but also do not undo or halve their combined effect merely because tags co-occur or sound related. Treat tags as duplicates only when "
                "the supplied deterministic evidence establishes the same event and independent quantitative evidence supports reducing the magnitude."
            ),
            "analogue_semantics": (
                "Do not search for or nominate one closest historical day as the main decision basis. Past observed days are secondary distribution and "
                "shape cross-checks, not historical model residuals. A past actual level cannot establish that the current raw forecast is high or low. "
                "Use historical evidence only when multiple independently comparable periods agree after current-sample evidence has been assessed."
            ),
            "direction_and_magnitude": (
                "Do not infer correction direction from event or weather labels alone; load and energy weather effects can be nonlinear. Rule bounds are "
                "safety limits, not recommended magnitudes, so never jump to a bound because several qualitative tags point the same way. The raw anchor "
                "may already encode ordinary trend, calendar, and exogenous effects, so never apply the same effect twice. For electric-load targets, cold, "
                "rain, or snow is not evidence for lower demand by itself and may sustain or increase heating and lighting load; absent quantitative task "
                "evidence, never use those labels to amplify a downward correction. Use exogenous conditions only to estimate the residual effect not "
                "already represented by the raw anchor. A coherent correction means consistent temporal coverage, not a larger amplitude."
            ),
            "detector_semantics": (
                "Detector limits are safety diagnostics, not desired forecast statistics. A soft warning alone is not enough reason to change "
                "values, and no point may be moved merely to cross a flat-window threshold. Clear a hard violation with an evidence-supported "
                "smooth shape rather than targeting its numeric boundary."
            ),
            "shape_guidance": (
                "Prefer smooth changes over coherent ramp, peak, or plateau periods. Avoid isolated one-point spikes or troughs unless the "
                "visible current-sample evidence is localized to that timestamp. Do not spread one uncertain factor across the whole day."
            ),
            "never_use": "Future observed target values, residuals, evaluation metrics, later windows, or hidden context.",
            "required_output": "Return every horizon_index and exact timestamp once, in original dataset units, and cite matching rule ids in evidence_ids.",
            "rationale_contract": (
                "Briefly state the joint target-day context, the skill-primary evidence, secondary supporting and opposing or neutral evidence, "
                "the residual effect not already represented by the raw anchor, and each segment's chosen delta relative to both anchors. "
                "Do not present a historical day as proof of forecast bias."
            ),
            "when_uncertain": (
                "Choose whichever anchor or context-derived profile is best supported by the joint current-sample evidence. If evidence is insufficient "
                "or contradictory, stay close to the better-supported anchor or localize the change instead of making a broad directional move. "
                "For an active-recalibration priority window, satisfy its minimum contract with the smallest coherent evidence-supported adjustment."
            ),
        }

    forbidden = nested_forbidden_keys(context)
    if forbidden:
        raise ValueError(f"Blind reflection context contains forbidden keys: {forbidden}")
    return context


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    task_path = Path(args.task_json)
    task = load_json(task_path)
    policy = load_json(args.policy_json)
    controller_version = str(policy.get("controller_version", CONTROLLER_VERSION))
    predictions = pd.read_csv(args.predictions_csv)
    if not {"timestamp", "predicted"}.issubset(predictions.columns):
        raise ValueError("Reflection baseline predictions require timestamp and predicted columns")
    pred_len = int(task["pred_len"])
    forecast_gap = int(task.get("forecast_gap", 0))
    window_index, horizon_index = prediction_coordinates(predictions, pred_len)
    predictions = predictions.copy()
    predictions["window_index"] = window_index
    predictions["horizon_index"] = horizon_index
    mean, scale = target_scale(task, task_path)
    baseline_scaled = pd.to_numeric(predictions["predicted"], errors="coerce").to_numpy(dtype=np.float64)
    baseline_original = (
        pd.to_numeric(predictions["adjusted_predicted_original"], errors="coerce").to_numpy(dtype=np.float64)
        if "adjusted_predicted_original" in predictions.columns
        else scaled_to_original(baseline_scaled, mean, scale)
    )
    raw_original = (
        pd.to_numeric(predictions["raw_predicted_original"], errors="coerce").to_numpy(dtype=np.float64)
        if "raw_predicted_original" in predictions.columns
        else baseline_original.copy()
    )

    deterministic_predictions_path = getattr(
        args, "deterministic_predictions_csv", None
    )
    if deterministic_predictions_path:
        deterministic_predictions = pd.read_csv(deterministic_predictions_path)
        if len(deterministic_predictions) != len(predictions):
            raise ValueError(
                "Deterministic experience predictions must align with the raw prediction prior"
            )
        if deterministic_predictions["timestamp"].astype(str).tolist() != predictions[
            "timestamp"
        ].astype(str).tolist():
            raise ValueError(
                "Deterministic experience timestamps do not align with the raw prediction prior"
            )
        deterministic_scaled = pd.to_numeric(
            deterministic_predictions["predicted"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        baseline_original = (
            pd.to_numeric(
                deterministic_predictions["adjusted_predicted_original"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            if "adjusted_predicted_original" in deterministic_predictions.columns
            else scaled_to_original(deterministic_scaled, mean, scale)
        )

    candidate = load_json(args.candidate_json) if args.candidate_json else None
    if candidate is not None and not isinstance(candidate, dict):
        raise ValueError("candidate.json must contain a JSON object")
    evidence_loop_test = args.split == "test"
    skill_guided_test = bool(
        args.split == "test"
        and args.baseline_kind == "skill_adjusted"
        and candidate is not None
    )
    if skill_guided_test:
        expected_candidate_id = str(args.adjustment_candidate_id or "")
        actual_candidate_id = str(candidate.get("candidate_id", ""))
        if not expected_candidate_id or actual_candidate_id != expected_candidate_id:
            raise ValueError(
                "Skill-guided test candidate id does not match adjustment_candidate_id"
            )
        requires_expert_mirror = bool(
            str(task.get("adjustment_skill_priors", "")).strip()
        )
        candidate_provenance = candidate.get("llm_provenance")
        is_expert_mirror = bool(
            candidate.get("origin") == "user_prior_mirror"
            or (
                isinstance(candidate_provenance, dict)
                and candidate_provenance.get("user_prior_mirror") is True
            )
        )
        if requires_expert_mirror and not is_expert_mirror:
            raise ValueError("Skill-guided test reasoning must use the task's first expert-derived user_prior_mirror skill")
        if not np.isfinite(raw_original).all() or not np.isfinite(baseline_original).all():
            raise ValueError(
                "Skill-guided test reasoning requires finite raw and deterministic predictions"
            )
    gap_predictions_path = getattr(args, "gap_predictions_csv", None)
    gap_predictions_sha256: str | None = None
    gap_predictions: pd.DataFrame | None = None
    if gap_predictions_path:
        gap_predictions_sha256 = file_sha256(gap_predictions_path)
        gap_predictions = pd.read_csv(gap_predictions_path)
        required_gap_columns = {
            "timestamp",
            "predicted",
            "prediction_layout",
            "window_index",
            "gap_index",
        }
        missing_gap_columns = required_gap_columns.difference(gap_predictions.columns)
        if missing_gap_columns:
            raise ValueError(
                f"Forecast-gap predictions are missing columns: {sorted(missing_gap_columns)}"
            )
        if set(gap_predictions["prediction_layout"].astype(str)) != {"forecast_gap"}:
            raise ValueError("Forecast-gap predictions require prediction_layout='forecast_gap'")
        gap_scaled = pd.to_numeric(gap_predictions["predicted"], errors="coerce").to_numpy(dtype=np.float64)
        gap_window_numbers = pd.to_numeric(gap_predictions["window_index"], errors="coerce").to_numpy(dtype=np.float64)
        gap_index_numbers = pd.to_numeric(gap_predictions["gap_index"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(gap_scaled).all():
            raise ValueError("Forecast-gap predictions must be finite")
        if not np.isfinite(gap_window_numbers).all() or not np.equal(gap_window_numbers, np.floor(gap_window_numbers)).all():
            raise ValueError("Forecast-gap window_index values must be finite integers")
        if not np.isfinite(gap_index_numbers).all() or not np.equal(gap_index_numbers, np.floor(gap_index_numbers)).all():
            raise ValueError("Forecast-gap gap_index values must be finite integers")
        gap_predictions = gap_predictions.copy()
        gap_predictions["window_index"] = gap_window_numbers.astype(int)
        gap_predictions["gap_index"] = gap_index_numbers.astype(int)
        gap_predictions["predicted_original"] = scaled_to_original(gap_scaled, mean, scale)
    if evidence_loop_test and forecast_gap > 0 and gap_predictions is None:
        raise ValueError(
            "Test evidence-loop reasoning with forecast_gap requires gap_predictions.csv from the source test run"
        )
    if evidence_loop_test and forecast_gap > 0 and gap_predictions is not None:
        expected_window_ids = sorted(set(int(value) for value in window_index))
        expected_row_count = len(expected_window_ids) * forecast_gap
        if len(gap_predictions) != expected_row_count:
            raise ValueError(
                f"Forecast-gap predictions require exactly {expected_row_count} rows for all test windows"
            )
        if gap_predictions.duplicated(["window_index", "gap_index"]).any():
            raise ValueError("Forecast-gap predictions contain duplicate window_index/gap_index pairs")
        if sorted(gap_predictions["window_index"].unique().tolist()) != expected_window_ids:
            raise ValueError("Forecast-gap predictions do not cover exactly the test prediction windows")
        for window_id, group in gap_predictions.groupby("window_index", sort=True):
            if sorted(group["gap_index"].tolist()) != list(range(forecast_gap)):
                raise ValueError(
                    f"Forecast-gap window {window_id} must contain contiguous gap_index values 0..{forecast_gap - 1}"
                )

    work_rows = pd.DataFrame(
        {
            "timestamp": predictions["timestamp"].astype(str),
            "window_index": window_index,
            "horizon_index": horizon_index,
            "raw_predicted_original": raw_original,
        }
    )
    if candidate:
        work_rows = attach_exogenous_features(
            task,
            task_path,
            work_rows,
            day_offsets=candidate_day_offsets(candidate),
        )
    rule_masks: dict[str, np.ndarray] = {}
    for rule in candidate.get("rules", []) if candidate else []:
        rule_masks[str(rule["id"])] = condition_mask(rule["condition"], work_rows)

    dataset = pd.read_csv(resolve_dataset_path(task, task_path), low_memory=False)
    out_dir = Path(args.out_dir)
    contexts_dir = out_dir / "blind-contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    windows: list[dict[str, Any]] = []
    matched_skill_window_count = 0
    for window in sorted(set(int(value) for value in window_index)):
        mask = window_index == window
        rows = predictions.loc[mask].sort_values("horizon_index")
        expected_horizons = list(range(pred_len))
        if rows["horizon_index"].astype(int).tolist() != expected_horizons:
            raise ValueError(f"Window {window} does not contain exactly horizon indices 0..{pred_len - 1}")
        row_positions = np.flatnonzero(mask)[np.argsort(horizon_index[mask])]
        raw_values = raw_original[row_positions]
        baseline_values = baseline_original[row_positions]
        matched_ids = [
            rule_id
            for rule_id, rule_mask in rule_masks.items()
            if bool(np.any(rule_mask[row_positions]))
        ]
        active_recalibration_priority = False
        matched_window_ordinal: int | None = None
        minimum_meaningful_delta_original = 0.0
        minimum_meaningful_points = 0
        if skill_guided_test and matched_ids:
            matched_skill_window_count += 1
            matched_window_ordinal = matched_skill_window_count
            active_recalibration_priority = (
                skill_guided_active_recalibration_priority(
                    matched_skill_window_count
                )
            )
            skill_delta = np.abs(baseline_values - raw_values)
            nonzero_skill_delta = skill_delta[skill_delta > 1e-9]
            skill_delta_scale = (
                float(np.median(nonzero_skill_delta))
                if nonzero_skill_delta.size
                else max(
                    1e-6,
                    0.01
                    * float(
                        policy.get("constraints", {}).get(
                            "window_mean_abs_delta_limit",
                            max(float(np.nanmedian(np.abs(baseline_values))), 1.0),
                        )
                    ),
                )
            )
            minimum_meaningful_delta_original = max(1e-6, 0.10 * skill_delta_scale)
            minimum_meaningful_points = max(3, int(math.ceil(0.10 * pred_len)))
        candidate_rules = candidate.get("rules", []) if candidate else []
        matching_rules: list[dict[str, Any]] = []
        for rule in candidate_rules:
            rule_id = str(rule["id"])
            if rule_id not in matched_ids:
                continue
            if skill_guided_test:
                matched_horizons = [
                    int(horizon)
                    for horizon, matched in zip(
                        expected_horizons,
                        rule_masks[rule_id][row_positions],
                        strict=True,
                    )
                    if bool(matched)
                ]
                matching_rules.append(
                    {
                        "id": rule_id,
                        "condition": rule.get("condition", {}),
                        "action": rule.get("action", {}),
                        "bounds": rule.get("bounds", {}),
                        "rationale": str(rule.get("rationale", "")),
                        "evidence_refs": [str(value) for value in rule.get("evidence_refs", [])],
                        "matched_horizon_indices": matched_horizons,
                        "deterministic_anchor_role": (
                            "The initial-based rule reference contains this locked rule's deterministic result; it is reference evidence, not a state transition."
                        ),
                    }
                )
            else:
                matching_rules.append(
                    {
                        "id": rule_id,
                        "action": rule.get("action", {}),
                        "bounds": rule.get("bounds", {}),
                    }
                )
        detection = detect_window(
            baseline_values,
            policy,
            reference=raw_values if args.baseline_kind == "skill_adjusted" else None,
            matched_rule_ids=matched_ids,
            require_adjustment_evidence=args.baseline_kind == "skill_adjusted",
        )
        # A skill-guided Critic run is an evidence-loop run for every prediction
        # window.  Experience may have no matching rule in a particular window,
        # but context and knowledge candidates must still be assessed.
        llm_required = True if evidence_loop_test else bool(detection["trigger"])
        reasoning_mode = (
            "skill_guided_test"
            if skill_guided_test
            else "evidence_loop_test"
            if evidence_loop_test
            else "safety_reflection"
        )
        failure_fallback = (
            "raw_anchor"
            if evidence_loop_test
            else "pre_reflection_baseline"
        )
        context_path: str | None = None
        context_sha256: str | None = None
        evidence_candidate_artifacts: dict[str, dict[str, str]] = {}
        critic_prediction_state_artifact: dict[str, str] | None = None
        critic_prediction_state_history: dict[str, dict[str, str]] = {}
        experience_rule_reference_artifact: dict[str, str] | None = None
        candidate_bundle: dict[str, Any] | None = None
        if llm_required:
            window_gap_rows = None
            if gap_predictions is not None:
                window_gap_rows = gap_predictions.loc[
                    pd.to_numeric(gap_predictions["window_index"], errors="raise").astype(int) == window
                ].copy()
            context = build_window_context(
                task,
                task_path,
                controller_version,
                dataset,
                rows,
                raw_values,
                baseline_values,
                detection,
                matching_rules,
                reasoning_mode=reasoning_mode,
                gap_prediction_rows=window_gap_rows,
                condition_feature_rows=work_rows.iloc[row_positions].copy(),
                candidate_id=str(candidate.get("candidate_id")) if candidate else None,
                policy=policy,
                active_recalibration_priority=active_recalibration_priority,
                matched_window_ordinal=matched_window_ordinal,
                minimum_meaningful_delta_original=minimum_meaningful_delta_original,
                minimum_meaningful_points=minimum_meaningful_points,
            )
            context_file = contexts_dir / f"window-{window:06d}.json"
            candidate_bundle_value = context.get("evidence_candidates")
            candidate_bundle = (
                candidate_bundle_value
                if isinstance(candidate_bundle_value, dict)
                else None
            )
            if isinstance(candidate_bundle, dict):
                for candidate_type in ("context", "knowledge", "experience"):
                    candidate_payload = candidate_bundle.get(candidate_type)
                    if candidate_type in {"context", "knowledge"} and not isinstance(
                        candidate_payload, dict
                    ):
                        raise ValueError(
                            f"Evidence candidate bundle is missing {candidate_type!r}"
                        )
                    if not isinstance(candidate_payload, dict):
                        continue
                    candidate_file = contexts_dir / (
                        f"window-{window:06d}.evidence-{candidate_type}.json"
                    )
                    write_json(candidate_file, candidate_payload)
                    evidence_candidate_artifacts[candidate_type] = {
                        "path": str(candidate_file),
                        "sha256": file_sha256(candidate_file),
                    }
                context["evidence_candidate_storage"] = evidence_candidate_artifacts
            initial_prediction_state = initial_critic_prediction_state(
                window,
                context["prediction_points"],
            )
            initial_state_file = contexts_dir / (
                f"window-{window:06d}.critic-prediction-state.initial.json"
            )
            write_json(initial_state_file, initial_prediction_state)
            critic_prediction_state_history["initial"] = {
                "path": str(initial_state_file),
                "sha256": file_sha256(initial_state_file),
            }
            critic_prediction_state = initial_prediction_state
            if isinstance(candidate_bundle, dict) and isinstance(
                candidate_bundle.get("experience"), dict
            ):
                rule_reference = experience_rule_reference(
                    initial_prediction_state,
                    context["prediction_points"],
                    candidate_bundle["experience"],
                    evidence_candidate_artifacts["experience"],
                )
                rule_reference_file = contexts_dir / (
                    f"window-{window:06d}.experience-rule-reference.json"
                )
                write_json(rule_reference_file, rule_reference)
                experience_rule_reference_artifact = {
                    "path": str(rule_reference_file),
                    "sha256": file_sha256(rule_reference_file),
                }
                context["experience_rule_reference_storage"] = experience_rule_reference_artifact
            current_state_artifact = critic_prediction_state_history["initial"]
            critic_prediction_state_artifact = current_state_artifact
            context["critic_prediction_state"] = critic_prediction_state
            context["critic_prediction_state_storage"] = critic_prediction_state_artifact
            context["critic_prediction_state_history"] = critic_prediction_state_history
            write_json(context_file, context)
            context_path = str(context_file)
            context_sha256 = file_sha256(context_file) if evidence_loop_test else None
        windows.append(
            {
                "window_index": window,
                "row_positions": row_positions.astype(int).tolist(),
                "timestamps": rows["timestamp"].astype(str).tolist(),
                "horizon_indices": expected_horizons,
                "raw_anchor_original": raw_values.tolist(),
                "baseline_original": baseline_values.tolist(),
                "detection": detection,
                "skill_matched": bool(matched_ids),
                "llm_required": llm_required,
                "active_recalibration_priority": active_recalibration_priority,
                "matched_window_ordinal": matched_window_ordinal,
                "minimum_meaningful_delta_original": minimum_meaningful_delta_original,
                "minimum_meaningful_points": minimum_meaningful_points,
                "reasoning_mode": reasoning_mode,
                "failure_fallback": failure_fallback,
                "blind_context_path": context_path,
                "blind_context_sha256": context_sha256,
                "evidence_candidate_artifacts": evidence_candidate_artifacts,
                "evidence_candidate_ids": {
                    candidate_type: str(candidate_payload["evidence_candidate_id"])
                    for candidate_type, candidate_payload in (candidate_bundle or {}).items()
                    if isinstance(candidate_payload, dict)
                    and candidate_payload.get("evidence_candidate_id")
                },
                "critic_prediction_state_artifact": critic_prediction_state_artifact,
                "critic_prediction_state_history": critic_prediction_state_history,
                "experience_rule_reference_artifact": experience_rule_reference_artifact,
            }
        )

    prepared = {
        "controller_version": controller_version,
        "source_run_id": args.source_run_id,
        "split": args.split,
        "baseline_kind": args.baseline_kind,
        "adjustment_candidate_id": args.adjustment_candidate_id,
        "task_json": str(task_path),
        "task_json_sha256": file_sha256(task_path) if evidence_loop_test else None,
        "policy_json": str(Path(args.policy_json)),
        "policy_json_sha256": file_sha256(args.policy_json) if evidence_loop_test else None,
        "predictions_csv": str(Path(args.predictions_csv)),
        "predictions_csv_sha256": file_sha256(args.predictions_csv) if evidence_loop_test else None,
        "deterministic_predictions_csv": (
            str(Path(deterministic_predictions_path))
            if deterministic_predictions_path
            else None
        ),
        "deterministic_predictions_csv_sha256": (
            file_sha256(deterministic_predictions_path)
            if evidence_loop_test and deterministic_predictions_path
            else None
        ),
        "candidate_json": str(Path(args.candidate_json)) if args.candidate_json else None,
        "candidate_json_sha256": file_sha256(args.candidate_json) if evidence_loop_test and args.candidate_json else None,
        "gap_predictions_csv": str(Path(gap_predictions_path)) if gap_predictions_path else None,
        "gap_predictions_sha256": gap_predictions_sha256,
        "target_mean": mean,
        "target_scale": scale,
        "row_count": int(len(predictions)),
        "window_count": len(windows),
        "trigger_count": sum(1 for item in windows if item["detection"]["trigger"]),
        "skill_match_count": sum(1 for item in windows if item["skill_matched"]),
        "llm_required_count": sum(1 for item in windows if item["llm_required"]),
        "active_recalibration_priority_count": sum(
            1 for item in windows if item["active_recalibration_priority"]
        ),
        "active_recalibration_minimum_share": SKILL_GUIDED_ACTIVE_RECALIBRATION_SHARE,
        "windows": windows,
    }
    prepared_path = out_dir / "prepared.json"
    write_json(prepared_path, prepared)
    audit = {
        "check": "blind_context_forbidden_keys",
        "passed": True,
        "target_values_visible": False,
        "residuals_visible": False,
        "metrics_visible": False,
        "gap_observed_targets_visible": False,
        "context_count": prepared["llm_required_count"],
        "test_evidence_loop": evidence_loop_test,
    }
    write_json(out_dir / "leakage-audit.json", audit)
    return {
        **prepared,
        "prepared_path": str(prepared_path),
        "windows": [
            {
                key: value
                for key, value in item.items()
                if key not in {"raw_anchor_original", "baseline_original", "row_positions"}
            }
            for item in windows
        ],
    }


def verify_skill_guided_prepared_integrity(
    args: argparse.Namespace,
    prepared: dict[str, Any],
) -> None:
    expected_prepared_sha256 = getattr(args, "prepared_sha256", None)
    if expected_prepared_sha256 and file_sha256(args.prepared_json) != expected_prepared_sha256:
        raise ValueError("Prepared reflection manifest changed after it was returned to the orchestrator")
    if prepared.get("split") != "test":
        return
    required_sources = [
        ("task_json", "task_json_sha256"),
        ("policy_json", "policy_json_sha256"),
        ("predictions_csv", "predictions_csv_sha256"),
    ]
    if prepared.get("candidate_json"):
        required_sources.append(("candidate_json", "candidate_json_sha256"))
    if prepared.get("deterministic_predictions_csv"):
        required_sources.append(
            (
                "deterministic_predictions_csv",
                "deterministic_predictions_csv_sha256",
            )
        )
    for path_key, hash_key in required_sources:
        source_path = prepared.get(path_key)
        expected_hash = prepared.get(hash_key)
        if not source_path or not expected_hash:
            raise ValueError(f"Skill-guided test prepared manifest is missing {path_key}/{hash_key}")
        if file_sha256(source_path) != expected_hash:
            raise ValueError(f"Skill-guided test source {path_key} changed after context preparation")
    if prepared.get("gap_predictions_csv"):
        if file_sha256(prepared["gap_predictions_csv"]) != prepared.get("gap_predictions_sha256"):
            raise ValueError("Forecast-gap predictions changed after the blind contexts were prepared")
    for window in prepared.get("windows", []):
        if window.get("reasoning_mode") not in {
            "skill_guided_test",
            "evidence_loop_test",
        }:
            continue
        artifacts = window.get("evidence_candidate_artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("Prepared reflection manifest has invalid evidence candidate artifacts")
        required_candidate_types = ["context", "knowledge"]
        if window.get("evidence_candidate_ids", {}).get("experience"):
            required_candidate_types.append("experience")
        for candidate_type in required_candidate_types:
            artifact = artifacts.get(candidate_type)
            if not isinstance(artifact, dict):
                raise ValueError(
                    f"Skill-guided prepared manifest is missing {candidate_type} evidence artifact"
                )
            candidate_path = artifact.get("path")
            expected_hash = artifact.get("sha256")
            if not candidate_path or not expected_hash:
                raise ValueError(
                    f"Skill-guided prepared manifest has incomplete {candidate_type} evidence artifact"
                )
            if file_sha256(candidate_path) != expected_hash:
                raise ValueError(
                    f"Skill-guided {candidate_type} evidence artifact changed after context preparation"
                )
        state_history = window.get("critic_prediction_state_history")
        if not isinstance(state_history, dict):
            raise ValueError("Skill-guided prepared manifest is missing the Critic prediction state history")
        for transition in ("initial",):
            state_artifact = state_history.get(transition)
            if not isinstance(state_artifact, dict):
                raise ValueError(
                    f"Skill-guided prepared manifest is missing the {transition} Critic prediction state"
                )
            state_path = state_artifact.get("path")
            state_hash = state_artifact.get("sha256")
            if not state_path or not state_hash:
                raise ValueError(f"Skill-guided {transition} Critic prediction state is incomplete")
            if file_sha256(state_path) != state_hash:
                raise ValueError(
                    f"Skill-guided {transition} Critic prediction state changed after context preparation"
                )
        experience_id = window.get("evidence_candidate_ids", {}).get("experience")
        rule_reference = window.get("experience_rule_reference_artifact")
        if experience_id:
            if not isinstance(rule_reference, dict):
                raise ValueError("Skill-guided prepared manifest is missing the experience rule reference")
            reference_path = rule_reference.get("path")
            reference_hash = rule_reference.get("sha256")
            if not reference_path or not reference_hash:
                raise ValueError("Skill-guided experience rule reference is incomplete")
            if file_sha256(reference_path) != reference_hash:
                raise ValueError("Skill-guided experience rule reference changed after context preparation")


def check_response(args: argparse.Namespace) -> dict[str, Any]:
    prepared = load_json(args.prepared_json)
    verify_skill_guided_prepared_integrity(args, prepared)
    policy = load_json(prepared["policy_json"])
    response = load_json(args.response_json)
    window = next(
        (item for item in prepared["windows"] if int(item["window_index"]) == int(args.window_index)),
        None,
    )
    if window is None:
        raise ValueError(f"Unknown reflection window {args.window_index}")
    expected_count = len(window["horizon_indices"])
    points = response.get("points", [])
    format_errors: list[str] = []
    if int(response.get("window_index", -1)) != int(args.window_index):
        format_errors.append("window_index does not match the requested window")
    if not isinstance(points, list):
        points = []
        format_errors.append("points must be an array")
    if abs(len(points) - expected_count) > 2:
        format_errors.append("point count differs from the required horizon by more than 2")

    accepted_points: dict[int, float] = {}
    expected_timestamps = {int(h): str(t) for h, t in zip(window["horizon_indices"], window["timestamps"], strict=True)}
    for point in points:
        try:
            horizon = int(point["horizon_index"])
            timestamp = str(point["timestamp"])
            value = float(point["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if horizon not in expected_timestamps or timestamp != expected_timestamps[horizon] or not math.isfinite(value):
            continue
        if horizon in accepted_points:
            continue
        accepted_points[horizon] = value

    missing = [horizon for horizon in window["horizon_indices"] if horizon not in accepted_points]
    if len(missing) > 2:
        format_errors.append("more than 2 required indexed points are missing or invalid")
    if format_errors:
        return {
            "accepted": False,
            "format_retry": True,
            "format_errors": format_errors,
            "missing_count": len(missing),
            "detector_report": None,
        }

    baseline = np.asarray(window["baseline_original"], dtype=np.float64)
    raw_anchor = np.asarray(window["raw_anchor_original"], dtype=np.float64)
    for horizon in missing:
        accepted_points[int(horizon)] = float(baseline[int(horizon)])
    values = np.asarray([accepted_points[horizon] for horizon in window["horizon_indices"]], dtype=np.float64)
    initial_prior = verified_initial_prior_response(response, window, values)
    terminal_state = verified_terminal_state_response(response, window, values)
    evidence_ids = [
        str(value)
        for value in response.get("evidence_ids", [])
        if isinstance(value, (str, int, float))
    ]
    available_rule_ids = set(window["detection"].get("matched_rule_ids", []))
    supported_rule_ids = sorted(set(evidence_ids).intersection(available_rule_ids))
    available_candidate_ids = set(window.get("evidence_candidate_ids", {}).values())
    supported_candidate_ids = sorted(
        set(evidence_ids).intersection(available_candidate_ids)
    )
    supported_evidence_ids = sorted(
        set(supported_rule_ids).union(supported_candidate_ids)
    )
    if terminal_state:
        detection = {
            "trigger": False,
            "hard_violations": [],
            "soft_warnings": [],
            "acceptance": "evidence_loop_terminal",
            "semantic_safety_policy_applied": False,
        }
    else:
        detection = detect_window(
            values,
            policy,
            reference=raw_anchor,
            matched_rule_ids=supported_evidence_ids,
            require_adjustment_evidence=(
                prepared["baseline_kind"] == "skill_adjusted"
                and not initial_prior
            ),
        )
    if (
        window.get("reasoning_mode") in {"skill_guided_test", "evidence_loop_test"}
        and not initial_prior
        and not terminal_state
        and not supported_candidate_ids
    ):
        detection["hard_violations"].append(
            {
                "id": "missing_selected_evidence",
                "message": "An evidence-loop proposal must cite one prepared candidate selected for this transition.",
            }
        )
        detection["trigger"] = True
    result = {
        "accepted": not detection["trigger"],
        "format_retry": False,
        "format_errors": [],
        "missing_count": len(missing),
        "detector_report": detection,
        "window_index": int(args.window_index),
        "round": int(args.round),
        "points": [
            {"horizon_index": int(h), "timestamp": expected_timestamps[int(h)], "value": float(values[int(h)])}
            for h in window["horizon_indices"]
        ],
        "rationale": str(response.get("rationale", "")),
        "evidence_ids": evidence_ids,
        "supported_evidence_ids": supported_evidence_ids,
        "assessment_stage": response.get("assessment_stage"),
        "critic_prediction_state": response.get("critic_prediction_state"),
        "critic_prediction_state_artifact": response.get(
            "critic_prediction_state_artifact"
        ),
    }
    if args.out:
        write_json(args.out, result)
    return result


def verified_initial_prior_response(
    response: dict[str, Any],
    window: dict[str, Any],
    values: np.ndarray,
) -> bool:
    """Accept an initial-prior short circuit only when it matches the frozen initial state exactly."""
    if (
        window.get("reasoning_mode") not in {"skill_guided_test", "evidence_loop_test"}
        or response.get("assessment_stage") != "initial_prior"
    ):
        return False
    supplied_state = response.get("critic_prediction_state")
    history = window.get("critic_prediction_state_history")
    initial_artifact = history.get("initial") if isinstance(history, dict) else None
    if not isinstance(supplied_state, dict) or not isinstance(initial_artifact, dict):
        return False
    try:
        frozen_state = load_json(initial_artifact["path"])
        expected_state = dict(frozen_state)
        expected_state["evidence_gap"] = 0
        expected_state["needed_evidence"] = ""
        gap = supplied_state.get("evidence_gap")
        raw_anchor = np.asarray(window["raw_anchor_original"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return (
        isinstance(gap, int)
        and not isinstance(gap, bool)
        and gap == 0
        and supplied_state == expected_state
        and supplied_state.get("evidences") == []
        and np.array_equal(values, raw_anchor)
    )


def verified_terminal_state_response(
    response: dict[str, Any],
    window: dict[str, Any],
    values: np.ndarray,
) -> bool:
    """Verify that the final Evidence Loop response exactly matches its immutable state artifact."""
    if (
        window.get("reasoning_mode") not in {"skill_guided_test", "evidence_loop_test"}
        or response.get("assessment_stage") != "evidence_loop_terminal"
    ):
        return False
    state = response.get("critic_prediction_state")
    artifact = response.get("critic_prediction_state_artifact")
    if not isinstance(state, dict) or not isinstance(artifact, dict):
        return False
    try:
        state_path = str(artifact["path"])
        expected_hash = str(artifact["sha256"])
        if file_sha256(state_path) != expected_hash or load_json(state_path) != state:
            return False
        predictions = state["predictions"]
        state_values = np.asarray(
            [float(point["value"]) for point in predictions], dtype=np.float64
        )
        evidence_ids = {
            str(evidence["evidence_candidate_id"])
            for evidence in state.get("evidences", [])
        }
        available_ids = set(window.get("evidence_candidate_ids", {}).values())
        gap = state["evidence_gap"]
        needed = state.get("needed_evidence", "")
    except (KeyError, TypeError, ValueError, OSError):
        return False
    gap_contract_valid = (
        isinstance(gap, int)
        and not isinstance(gap, bool)
        and gap in {0, 1}
        and isinstance(needed, str)
        and ((gap == 0 and not needed.strip()) or (gap == 1 and bool(needed.strip())))
    )
    return (
        gap_contract_valid
        and evidence_ids.issubset(available_ids)
        and state_values.shape == values.shape
        and np.array_equal(state_values, values)
    )


def verified_correction_values(
    correction: dict[str, Any] | None,
    window: dict[str, Any],
    policy: dict[str, Any],
    baseline_kind: str,
) -> np.ndarray | None:
    """Revalidate a persisted correction before it can affect final predictions."""
    if not isinstance(correction, dict) or correction.get("accepted") is not True:
        return None
    try:
        if int(correction.get("window_index", -1)) != int(window["window_index"]):
            return None
        points = correction.get("points")
        if not isinstance(points, list) or len(points) != len(window["horizon_indices"]):
            return None
        expected_timestamps = {
            int(horizon): str(timestamp)
            for horizon, timestamp in zip(
                window["horizon_indices"],
                window["timestamps"],
                strict=True,
            )
        }
        indexed: dict[int, float] = {}
        for point in points:
            horizon = int(point["horizon_index"])
            timestamp = str(point["timestamp"])
            value = float(point["value"])
            if (
                horizon not in expected_timestamps
                or horizon in indexed
                or timestamp != expected_timestamps[horizon]
                or not math.isfinite(value)
            ):
                return None
            indexed[horizon] = value
        if set(indexed) != set(expected_timestamps):
            return None
    except (KeyError, TypeError, ValueError):
        return None

    values = np.asarray(
        [indexed[int(horizon)] for horizon in window["horizon_indices"]],
        dtype=np.float64,
    )
    evidence_ids = [
        str(value)
        for value in correction.get("evidence_ids", [])
        if isinstance(value, (str, int, float))
    ]
    available_rule_ids = set(window["detection"].get("matched_rule_ids", []))
    supported_rule_ids = sorted(set(evidence_ids).intersection(available_rule_ids))
    initial_prior = verified_initial_prior_response(correction, window, values)
    terminal_state = verified_terminal_state_response(correction, window, values)
    available_candidate_ids = set(window.get("evidence_candidate_ids", {}).values())
    supported_candidate_ids = sorted(
        set(evidence_ids).intersection(available_candidate_ids)
    )
    supported_evidence_ids = sorted(
        set(supported_rule_ids).union(supported_candidate_ids)
    )
    if (
        window.get("reasoning_mode") in {"skill_guided_test", "evidence_loop_test"}
        and not initial_prior
        and not terminal_state
        and not supported_evidence_ids
    ):
        return None
    if terminal_state:
        return values
    detection = detect_window(
        values,
        policy,
        reference=np.asarray(window["raw_anchor_original"], dtype=np.float64),
        matched_rule_ids=supported_evidence_ids,
        require_adjustment_evidence=baseline_kind == "skill_adjusted" and not initial_prior,
    )
    return None if detection["trigger"] else values

def compute_metrics(predicted: np.ndarray, actual: np.ndarray, train_values: np.ndarray) -> dict[str, float | None]:
    return {
        "mse": float(mse(predicted, actual)),
        "mae": float(mae(predicted, actual)),
        "wape": float(wape(predicted, actual)),
        "mase": float(mase(predicted, actual, train_values)) if len(train_values) > 1 else None,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    prepared = load_json(args.prepared_json)
    verify_skill_guided_prepared_integrity(args, prepared)
    out_dir = Path(args.out_dir).resolve()
    corrections_dir = Path(args.corrections_dir).resolve()
    if Path(args.prepared_json).resolve().parent != out_dir:
        raise ValueError("Prepared reflection manifest must be inside the output directory")
    if corrections_dir != out_dir / "corrections":
        raise ValueError("Reflection corrections directory does not match the prepared output")
    audit_path = out_dir / "leakage-audit.json"
    if not audit_path.is_file():
        raise ValueError("Reflection leakage audit is missing during finalization")

    llm_protocol_path: Path | None = None
    llm_protocol_sha256: str | None = None
    expected_llm_protocol_sha256 = getattr(args, "llm_protocol_sha256", None)
    if getattr(args, "llm_protocol_json", None):
        llm_protocol_path = Path(args.llm_protocol_json).resolve()
        if llm_protocol_path.parent != out_dir or not llm_protocol_path.is_file():
            raise ValueError("LLM protocol must be an existing artifact inside the reflection output directory")
        llm_protocol_sha256 = file_sha256(llm_protocol_path)
        if (
            expected_llm_protocol_sha256
            and llm_protocol_sha256 != expected_llm_protocol_sha256
        ):
            raise ValueError("Frozen skill-guided LLM protocol changed before finalization")
    elif expected_llm_protocol_sha256:
        raise ValueError("An expected LLM protocol hash requires --llm-protocol-json")

    predictions = pd.read_csv(prepared["predictions_csv"])
    if prepared.get("gap_predictions_csv"):
        if file_sha256(prepared["gap_predictions_csv"]) != prepared.get("gap_predictions_sha256"):
            raise ValueError("Forecast-gap predictions changed after the blind contexts were prepared")
    baseline_scaled = pd.to_numeric(predictions["predicted"], errors="coerce").to_numpy(dtype=np.float64)
    final_scaled = baseline_scaled.copy()
    mean = float(prepared["target_mean"])
    scale = float(prepared["target_scale"])
    applied_windows: list[int] = []
    fallback_windows: list[int] = []
    deterministic_fallback_windows: list[int] = []
    raw_fallback_windows: list[int] = []
    policy = load_json(prepared["policy_json"])
    for window in prepared["windows"]:
        if not bool(window.get("llm_required", window["detection"]["trigger"])):
            continue
        positions = np.asarray(window["row_positions"], dtype=np.int64)
        correction_path = corrections_dir / f"window-{int(window['window_index']):06d}.json"
        correction = load_json(correction_path) if correction_path.exists() else None
        values = verified_correction_values(
            correction,
            window,
            policy,
            str(prepared["baseline_kind"]),
        )
        if values is None:
            window_id = int(window["window_index"])
            fallback_windows.append(window_id)
            if window.get("failure_fallback") == "raw_anchor":
                raw_values = np.asarray(window["raw_anchor_original"], dtype=np.float64)
                final_scaled[positions] = (raw_values - mean) / scale
                raw_fallback_windows.append(window_id)
            else:
                deterministic_values = np.asarray(
                    window["baseline_original"], dtype=np.float64
                )
                final_scaled[positions] = (deterministic_values - mean) / scale
                deterministic_fallback_windows.append(window_id)
            continue
        final_scaled[positions] = (values - mean) / scale
        applied_windows.append(int(window["window_index"]))

    if "actual" not in predictions.columns:
        raise ValueError("Reflection finalization requires actual values for offline aggregate evaluation")
    actual = pd.to_numeric(predictions["actual"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(final_scaled).all():
        raise ValueError("Reflection metrics require finite actual and prediction arrays")
    task = load_json(prepared["task_json"])
    task_path = Path(prepared["task_json"])
    training_original = finite_training_target(task, task_path)
    training_scaled = (training_original - mean) / scale
    baseline_metrics = compute_metrics(baseline_scaled, actual, training_scaled)
    final_metrics = compute_metrics(final_scaled, actual, training_scaled)
    delta_mse = float(final_metrics["mse"] - baseline_metrics["mse"])
    if prepared["split"] == "val":
        if int(prepared["trigger_count"]) == 0:
            status = "enabled_no_trigger"
        elif delta_mse < -1e-12:
            status = "enabled_improved"
        elif delta_mse <= 1e-12:
            status = "enabled_unchanged"
        else:
            status = "disabled_negative_mse"
    else:
        status = "completed"

    out_dir.mkdir(parents=True, exist_ok=True)
    output = predictions.copy()
    output["pre_reflection_predicted"] = baseline_scaled
    output["predicted"] = final_scaled
    output["pre_reflection_predicted_original"] = baseline_scaled * scale + mean
    output["final_predicted_original"] = final_scaled * scale + mean
    reflected = np.zeros(len(output), dtype=bool)
    for window in prepared["windows"]:
        if int(window["window_index"]) in applied_windows:
            reflected[np.asarray(window["row_positions"], dtype=np.int64)] = True
    output["reflection_applied"] = reflected
    reasoning_required = np.zeros(len(output), dtype=bool)
    skill_matched = np.zeros(len(output), dtype=bool)
    fallback_kind = np.full(len(output), "none", dtype=object)
    for window in prepared["windows"]:
        positions = np.asarray(window["row_positions"], dtype=np.int64)
        if bool(window.get("llm_required", window["detection"]["trigger"])):
            reasoning_required[positions] = True
        if bool(window.get("skill_matched", False)):
            skill_matched[positions] = True
        window_id = int(window["window_index"])
        if window_id in deterministic_fallback_windows:
            fallback_kind[positions] = "deterministic_skill"
        elif window_id in raw_fallback_windows:
            fallback_kind[positions] = "raw_anchor"
    output["reflection_reasoning_required"] = reasoning_required
    output["reflection_skill_matched"] = skill_matched
    output["reflection_fallback_kind"] = fallback_kind
    output_path = out_dir / "final_predictions.csv"
    output.to_csv(output_path, index=False)

    result = {
        "controller_version": str(prepared.get("controller_version", CONTROLLER_VERSION)),
        "source_run_id": prepared["source_run_id"],
        "split": prepared["split"],
        "baseline_kind": prepared["baseline_kind"],
        "adjustment_candidate_id": prepared.get("adjustment_candidate_id"),
        "status": status,
        "trigger_count": int(prepared["trigger_count"]),
        "skill_match_count": int(prepared.get("skill_match_count", 0)),
        "llm_required_count": int(prepared.get("llm_required_count", prepared["trigger_count"])),
        "corrected_count": len(applied_windows),
        "fallback_count": len(fallback_windows),
        "deterministic_fallback_count": len(deterministic_fallback_windows),
        "raw_fallback_count": len(raw_fallback_windows),
        "applied_windows": applied_windows,
        "fallback_windows": fallback_windows,
        "deterministic_fallback_windows": deterministic_fallback_windows,
        "raw_fallback_windows": raw_fallback_windows,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "delta_mse": delta_mse,
        "final_predictions_path": str(output_path),
        "final_predictions_sha256": file_sha256(output_path),
        "leakage_audit_sha256": file_sha256(audit_path),
        "llm_protocol_path": str(llm_protocol_path) if llm_protocol_path else None,
        "llm_protocol_sha256": llm_protocol_sha256,
        "gap_predictions_sha256": prepared.get("gap_predictions_sha256"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = out_dir / "reflection_eval.json"
    write_json(result_path, result)
    result["result_path"] = str(result_path)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaCast independent window-reflection controller")
    subparsers = parser.add_subparsers(dest="action", required=True)
    configure_parser = subparsers.add_parser("configure")
    configure_parser.add_argument("--task-json", required=True)
    configure_parser.add_argument("--planner-json")
    configure_parser.add_argument("--out", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--task-json", required=True)
    prepare_parser.add_argument("--policy-json", required=True)
    prepare_parser.add_argument("--predictions-csv", required=True)
    prepare_parser.add_argument("--deterministic-predictions-csv")
    prepare_parser.add_argument("--gap-predictions-csv")
    prepare_parser.add_argument("--candidate-json")
    prepare_parser.add_argument("--adjustment-candidate-id")
    prepare_parser.add_argument("--baseline-kind", choices=["raw_anchor", "skill_adjusted"], required=True)
    prepare_parser.add_argument("--source-run-id", required=True)
    prepare_parser.add_argument("--split", choices=["val", "test"], required=True)
    prepare_parser.add_argument("--out-dir", required=True)

    check_parser = subparsers.add_parser("check-response")
    check_parser.add_argument("--prepared-json", required=True)
    check_parser.add_argument("--prepared-sha256")
    check_parser.add_argument("--response-json", required=True)
    check_parser.add_argument("--window-index", type=int, required=True)
    check_parser.add_argument("--round", type=int, required=True)
    check_parser.add_argument("--out")

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--prepared-json", required=True)
    finalize_parser.add_argument("--prepared-sha256")
    finalize_parser.add_argument("--corrections-dir", required=True)
    finalize_parser.add_argument("--out-dir", required=True)
    finalize_parser.add_argument("--llm-protocol-json")
    finalize_parser.add_argument("--llm-protocol-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")
    args = parse_args(argv)
    try:
        if args.action == "configure":
            task_path = Path(args.task_json)
            result = derive_policy(
                load_json(task_path),
                task_path,
                load_json(args.planner_json) if args.planner_json else None,
            )
            write_json(args.out, result)
            result = {**result, "policy_path": str(Path(args.out))}
        elif args.action == "prepare":
            result = prepare(args)
        elif args.action == "check-response":
            result = check_response(args)
        else:
            result = finalize(args)
        json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        json.dump({"status": "error", "error": str(exc)}, sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
