"""Data loading with frozen split enforcement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ADJUSTMENT_FIT_HOLDOUT_NUMERATOR = 1
ADJUSTMENT_FIT_HOLDOUT_DENOMINATOR = 2
ADJUSTMENT_FIT_HOLDOUT_RATIO = (
    ADJUSTMENT_FIT_HOLDOUT_NUMERATOR / ADJUSTMENT_FIT_HOLDOUT_DENOMINATOR
)


def parse_forecast_gap(spec: dict) -> int:
    """Return the frozen nonnegative gap without accepting bools or fractions."""
    value = spec.get("forecast_gap", 0)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("forecast_gap must be a nonnegative integer")
    return int(value)


def reject_if_test_in_forecast(spec: dict) -> None:
    """Reject forecast-phase specs that attempt to access the test split."""
    phase = spec.get("phase", "forecast")
    eval_split = spec.get("eval_split", "val")
    if phase == "forecast" and eval_split == "test":
        raise SystemExit(
            "SAFETY VIOLATION: Test split is inaccessible during forecast phase. "
            f"spec.phase={phase!r}, spec.eval_split={eval_split!r}"
        )


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Require one non-empty chronological partition covering the full dataset."""
    ratios = {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }
    for name, value in ratios.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be strictly between 0 and 1")
    ratio_sum = sum(ratios.values())
    if not np.isclose(ratio_sum, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1 (received {ratio_sum})"
        )


def compute_split_indices(
    n_rows: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float | None = None,
) -> tuple[int, int]:
    """Compute absolute split boundaries from the spec ratios."""
    resolved_test_ratio = (
        1.0 - float(train_ratio) - float(val_ratio)
        if test_ratio is None
        else float(test_ratio)
    )
    validate_split_ratios(train_ratio, val_ratio, resolved_test_ratio)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    return train_end, val_end


def _timestamp_boundary_index(
    timestamps: pd.Series,
    boundary: object,
    label: str,
) -> int:
    """Resolve one inclusive timestamp boundary to its unique row index."""
    raw_values = timestamps.astype(str).str.strip()
    raw_boundary = str(boundary).strip()
    exact_matches = np.flatnonzero(raw_values.to_numpy() == raw_boundary)
    if len(exact_matches) == 1:
        return int(exact_matches[0])
    if len(exact_matches) > 1:
        raise ValueError(
            f"{label} {raw_boundary!r} is ambiguous because it appears multiple times"
        )

    parsed_values = pd.to_datetime(timestamps, errors="coerce", utc=True)
    parsed_boundary = pd.to_datetime(raw_boundary, errors="coerce", utc=True)
    if pd.isna(parsed_boundary):
        raise ValueError(f"{label} {raw_boundary!r} is not a valid timestamp")
    parsed_matches = np.flatnonzero((parsed_values == parsed_boundary).to_numpy())
    if len(parsed_matches) == 0:
        raise ValueError(
            f"{label} {raw_boundary!r} was not found in the dataset time column"
        )
    if len(parsed_matches) > 1:
        raise ValueError(
            f"{label} {raw_boundary!r} is ambiguous because it matches multiple rows"
        )
    return int(parsed_matches[0])


def resolve_split_indices(frame: pd.DataFrame, spec: dict) -> tuple[int, int]:
    """Resolve chronological forecast-origin boundaries from a frozen task spec.

    Ratio mode keeps the historical row-count behavior. In timestamp mode, the
    two timestamps identify the final row whose value may belong to the
    preceding split's history/origin context. Later evaluation windows may still
    use earlier rows as leakage-safe history context.
    """
    split_mode = str(spec.get("split_mode", "ratio"))
    if split_mode == "ratio":
        if (
            spec.get("train_end_timestamp") is not None
            or spec.get("validation_start_timestamp") is not None
            or spec.get("validation_end_timestamp") is not None
        ):
            raise ValueError("Ratio split mode cannot include timestamp boundaries")
        return compute_split_indices(
            len(frame),
            float(spec["train_ratio"]),
            float(spec["val_ratio"]),
            float(
                spec.get(
                    "test_ratio",
                    1.0 - float(spec["train_ratio"]) - float(spec["val_ratio"]),
                )
            ),
        )
    if split_mode != "timestamp":
        raise ValueError(f"Unknown split_mode: {split_mode!r}")

    time_col = str(spec["time_col"])
    if time_col not in frame.columns:
        raise ValueError(f"Time column {time_col!r} not found in dataset")
    train_boundary = spec.get("train_end_timestamp")
    validation_boundary = spec.get("validation_end_timestamp")
    if train_boundary is None or validation_boundary is None:
        raise ValueError(
            "Timestamp split mode requires train_end_timestamp and "
            "validation_end_timestamp"
        )

    parsed_timestamps = pd.to_datetime(frame[time_col], errors="coerce", utc=True)
    if parsed_timestamps.isna().any():
        raise ValueError(
            "Timestamp split mode requires every value in the time column to be parseable"
        )
    if not parsed_timestamps.is_unique:
        raise ValueError(
            "Timestamp split mode requires unique values in the time column"
        )
    if not parsed_timestamps.is_monotonic_increasing:
        raise ValueError(
            "Timestamp split mode requires the dataset rows to be strictly chronological"
        )

    train_end = (
        _timestamp_boundary_index(
            frame[time_col],
            train_boundary,
            "train_end_timestamp",
        )
        + 1
    )
    val_end = (
        _timestamp_boundary_index(
            frame[time_col],
            validation_boundary,
            "validation_end_timestamp",
        )
        + 1
    )
    if train_end >= val_end:
        raise ValueError(
            "train_end_timestamp must occur before validation_end_timestamp"
        )
    if val_end >= len(frame):
        raise ValueError(
            "validation_end_timestamp must leave at least one later row for the test split"
        )

    expected_ratios = (
        train_end / len(frame),
        (val_end - train_end) / len(frame),
        (len(frame) - val_end) / len(frame),
    )
    configured_ratios = (
        spec.get("train_ratio"),
        spec.get("val_ratio"),
        spec.get("test_ratio"),
    )
    if all(value is not None for value in configured_ratios):
        if not all(
            np.isclose(float(configured), expected, rtol=0.0, atol=1e-9)
            for configured, expected in zip(configured_ratios, expected_ratios)
        ):
            raise ValueError(
                "Frozen split ratios do not match the timestamp boundary row counts"
            )
    return train_end, val_end


def normalize_adjustment_fit_holdout_ratio(spec: dict) -> dict:
    """Return a spec carrying the fixed validation-front evidence contract."""
    normalized = dict(spec)
    configured = float(
        normalized.get("adjustment_fit_holdout_ratio", ADJUSTMENT_FIT_HOLDOUT_RATIO)
    )
    if not np.isclose(configured, ADJUSTMENT_FIT_HOLDOUT_RATIO, rtol=0.0, atol=1e-12):
        raise ValueError(
            "adjustment_fit_holdout_ratio is fixed at 1/2 of the validation split and cannot be overridden"
        )
    normalized["adjustment_fit_holdout_ratio"] = ADJUSTMENT_FIT_HOLDOUT_RATIO
    return normalized


def compute_training_boundaries(
    n_rows: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float | None = None,
) -> tuple[int, int, int]:
    """Return train end, adjustment-fit end, and validation end.

    The full user training split is used for model fitting. The front half of
    the validation split is reserved as optional adjustment-fit evidence, and
    the back half remains the validation evaluation interval.
    """
    train_end, val_end = compute_split_indices(
        n_rows,
        train_ratio,
        val_ratio,
        test_ratio,
    )
    val_rows = val_end - train_end
    adjustment_rows = (
        val_rows
        * ADJUSTMENT_FIT_HOLDOUT_NUMERATOR
        // ADJUSTMENT_FIT_HOLDOUT_DENOMINATOR
    )
    adjustment_fit_end = train_end + adjustment_rows
    if train_end <= 0:
        raise ValueError("Training split is empty under the frozen ratios")
    if adjustment_fit_end <= train_end or adjustment_fit_end >= val_end:
        raise ValueError(
            "Validation split is too short to reserve its front 1/2 for adjustment evidence"
        )
    return train_end, adjustment_fit_end, val_end


def resolve_validation_start_index(frame: pd.DataFrame, spec: dict, train_end: int, val_end: int) -> int:
    """Leave an optional post-training bridge available only as rolling history."""
    boundary = spec.get("validation_start_timestamp")
    if boundary is None:
        return train_end
    if spec.get("split_mode", "ratio") != "timestamp":
        raise ValueError("validation_start_timestamp requires timestamp split mode")
    start = _timestamp_boundary_index(frame[str(spec["time_col"])], boundary, "validation_start_timestamp")
    if not train_end <= start < val_end:
        raise ValueError("Validation must start after training and before validation ends")
    return start


def resolve_training_boundaries(
    frame: pd.DataFrame,
    spec: dict,
) -> tuple[int, int, int]:
    """Return model train, adjustment-fit, and held-out validation boundaries."""
    train_end, val_end = resolve_split_indices(frame, spec)
    validation_start = resolve_validation_start_index(frame, spec, train_end, val_end)
    val_rows = val_end - validation_start
    adjustment_rows = (
        val_rows
        * ADJUSTMENT_FIT_HOLDOUT_NUMERATOR
        // ADJUSTMENT_FIT_HOLDOUT_DENOMINATOR
    )
    adjustment_fit_end = validation_start + adjustment_rows
    if train_end <= 0:
        raise ValueError("Training split is empty under the frozen split contract")
    if adjustment_fit_end <= validation_start or adjustment_fit_end >= val_end:
        raise ValueError(
            "Validation split is too short to reserve its front 1/2 for adjustment evidence"
    )
    return train_end, adjustment_fit_end, val_end


def numeric_feature_columns(frame: pd.DataFrame, time_col: str, target_col: str) -> list[str]:
    """Return complete numeric model-input columns while preserving CSV order.

    Text/calendar fields may still be present in the task CSV so adjustment
    rules can read them later. Model training and UserPrior scaling only use
    columns that are fully numeric across the frozen dataset, with the target
    column required to be finite.
    """
    feature_cols: list[str] = []
    for column in frame.columns:
        if column == time_col:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if column == target_col:
            if not np.isfinite(values).all():
                raise ValueError(f"Target column {target_col!r} must be fully numeric and finite")
            feature_cols.append(str(column))
            continue
        if np.isfinite(values).all():
            feature_cols.append(str(column))
    if target_col not in feature_cols:
        raise ValueError(f"Target column {target_col!r} not found in dataset")
    return feature_cols


def load_dataset(spec: dict, allowed_splits: list[str]) -> dict[str, dict[str, object]]:
    """Load a CSV dataset and return only the requested split payloads."""
    if not allowed_splits:
        raise ValueError("allowed_splits must not be empty")

    spec = normalize_adjustment_fit_holdout_ratio(spec)
    df = pd.read_csv(spec["dataset_path"], low_memory=False)
    time_col = spec["time_col"]
    target_col = spec["target_col"]
    feature_cols = numeric_feature_columns(df, str(time_col), str(target_col))
    target_index = feature_cols.index(str(target_col))
    feature_data = df[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    timestamps = df[time_col].to_numpy()

    train_end, adjustment_fit_end, val_end = resolve_training_boundaries(df, spec)
    validation_start = resolve_validation_start_index(df, spec, train_end, val_end)

    scaler = StandardScaler()
    scaler.fit(feature_data[:train_end])

    seq_len = int(spec.get("seq_len", 0))
    forecast_gap = parse_forecast_gap(spec)
    pred_len = int(spec.get("pred_len", 1))

    def evaluation_context(
        *,
        origin_abs: int,
        target_end_abs: int,
        split_name: str,
    ) -> tuple[int, int, int]:
        target_start_abs = origin_abs + forecast_gap
        if target_end_abs - target_start_abs < pred_len:
            required_window = (
                "pred_len" if forecast_gap == 0 else "forecast_gap + pred_len"
            )
            raise ValueError(
                f"Split {split_name!r} is shorter than {required_window} and cannot produce metrics"
            )
        if origin_abs < seq_len:
            raise ValueError(
                f"Rows before split {split_name!r} are shorter than seq_len"
            )
        context_start = origin_abs - seq_len
        return (
            context_start,
            origin_abs - context_start,
            target_start_abs - context_start,
        )

    if "train" in allowed_splits and train_end < seq_len + forecast_gap + pred_len:
        raise ValueError(
            "The training split is shorter than seq_len + forecast_gap + pred_len"
        )
    if "adjustment_fit" in allowed_splits:
        if seq_len > 0 and train_end < seq_len:
            raise ValueError("The training split is shorter than seq_len")
        evaluation_context(
            origin_abs=validation_start,
            target_end_abs=adjustment_fit_end,
            split_name="adjustment_fit",
        )
    if "val" in allowed_splits:
        evaluation_context(
            origin_abs=adjustment_fit_end,
            target_end_abs=val_end,
            split_name="val",
        )
    if "test" in allowed_splits:
        evaluation_context(
            origin_abs=val_end,
            target_end_abs=len(df),
            split_name="test",
        )

    def split_payload_indices(split_name: str) -> tuple[int, int, int, int]:
        if split_name == "train":
            return 0, train_end, seq_len, seq_len + forecast_gap
        if split_name == "adjustment_fit":
            start, origin, target_start = evaluation_context(
                origin_abs=validation_start,
                target_end_abs=adjustment_fit_end,
                split_name=split_name,
            )
            return start, adjustment_fit_end, origin, target_start
        if split_name == "val":
            start, origin, target_start = evaluation_context(
                origin_abs=adjustment_fit_end,
                target_end_abs=val_end,
                split_name=split_name,
            )
            return start, val_end, origin, target_start
        if split_name == "test":
            start, origin, target_start = evaluation_context(
                origin_abs=val_end,
                target_end_abs=len(df),
                split_name=split_name,
            )
            return start, len(df), origin, target_start
        raise ValueError(f"Unknown split: {split_name!r}")

    forecast_offsets = {
        "train": 0,
        "adjustment_fit": validation_start - train_end + forecast_gap,
        "val": adjustment_fit_end - train_end + forecast_gap,
        "test": val_end - train_end + forecast_gap,
    }

    result: dict[str, dict[str, object]] = {}
    for split_name in allowed_splits:
        start, end, forecast_origin_index, target_start_index = split_payload_indices(
            split_name
        )
        if start >= end:
            raise ValueError(
                f"Split {split_name!r} is empty under the frozen split contract"
            )
        result[split_name] = {
            "data": scaler.transform(feature_data[start:end]),
            "raw": feature_data[start:end],
            "timestamps": timestamps[start:end],
            "absolute_start_index": start,
            "forecast_origin_index": forecast_origin_index,
            "target_start_index": target_start_index,
            "forecast_offset": forecast_offsets[split_name],
            "model_train_end": train_end,
            "outer_train_end": train_end,
            "adjustment_fit_end": adjustment_fit_end,
            "validation_eval_start_index": adjustment_fit_end,
            "feature_columns": list(feature_cols),
            "target_index": target_index,
        }

    return result
