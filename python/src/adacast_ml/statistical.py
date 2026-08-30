"""Statsmodels-backed forecasting registry and fitted-result helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing as StatsmodelsExponentialSmoothing,
)
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing


STATISTICAL_MODEL_NAMES = (
    "ARIMA",
    "AutoARIMA",
    "ETS",
    "ExponentialSmoothing",
    "SimpleExponentialSmoothing",
    "Holt",
    "HoltWinters",
    "Theta",
)

_CANONICAL_BY_NORMALIZED = {
    "".join(character for character in name.casefold() if character.isalnum()): name
    for name in STATISTICAL_MODEL_NAMES
}


def normalize_statistical_model_name(model_name: str) -> str | None:
    normalized = "".join(
        character for character in str(model_name).casefold() if character.isalnum()
    )
    return _CANONICAL_BY_NORMALIZED.get(normalized)


def is_statistical_model(model_name: str) -> bool:
    return normalize_statistical_model_name(model_name) is not None


def _tuple_param(
    hyperparams: dict[str, Any],
    key: str,
    default: tuple[int, ...],
    length: int,
) -> tuple[int, ...]:
    raw = hyperparams.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise ValueError(f"Statistical hyperparams.{key} must contain {length} integers")
    return tuple(int(value) for value in raw)


def _positive_period(hyperparams: dict[str, Any], *, required: bool) -> int | None:
    raw = hyperparams.get("seasonal_periods", hyperparams.get("seasonal_period"))
    if raw is None:
        if required:
            raise ValueError(
                "This statistical model requires hyperparams.seasonal_periods >= 2"
            )
        return None
    period = int(raw)
    if period < 2:
        raise ValueError("hyperparams.seasonal_periods must be at least 2")
    return period


def _fit_auto_arima(
    train_series: np.ndarray,
    hyperparams: dict[str, Any],
) -> Any:
    candidate_orders_raw = hyperparams.get("candidate_orders")
    if candidate_orders_raw is None:
        candidate_orders = [
            (0, 0, 1),
            (1, 0, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 0),
            (1, 1, 1),
            (2, 1, 0),
            (0, 1, 2),
        ]
    else:
        if not isinstance(candidate_orders_raw, list) or not candidate_orders_raw:
            raise ValueError(
                "AutoARIMA hyperparams.candidate_orders must be a non-empty list"
            )
        candidate_orders = []
        for raw_order in candidate_orders_raw:
            if not isinstance(raw_order, (list, tuple)) or len(raw_order) != 3:
                raise ValueError(
                    "Each AutoARIMA candidate order must contain three integers"
                )
            candidate_orders.append(tuple(int(value) for value in raw_order))

    best_result = None
    best_aic = float("inf")
    failures: list[str] = []
    for order in candidate_orders:
        try:
            result = StatsmodelsARIMA(
                train_series,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append(f"{order}: {exc}")
            continue
        aic = float(result.aic)
        if np.isfinite(aic) and aic < best_aic:
            best_result = result
            best_aic = aic

    if best_result is None:
        detail = "; ".join(failures[-3:]) or "no finite AIC candidate"
        raise ValueError(f"AutoARIMA could not fit any candidate order: {detail}")
    return best_result


def fit_statistical_model(
    model_name: str,
    train_series: np.ndarray,
    hyperparams: dict[str, Any],
) -> tuple[str, Any]:
    """Fit one registered statistical model on the frozen training prefix."""
    canonical = normalize_statistical_model_name(model_name)
    if canonical is None:
        raise ValueError(
            f"Unsupported statistical model {model_name!r}. "
            f"Available: {', '.join(STATISTICAL_MODEL_NAMES)}"
        )
    values = np.asarray(train_series, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError(
            "Statistical models require a finite one-dimensional training target"
        )

    if canonical == "ARIMA":
        order = _tuple_param(hyperparams, "order", (1, 1, 1), 3)
        fitted = StatsmodelsARIMA(
            values,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
    elif canonical == "AutoARIMA":
        fitted = _fit_auto_arima(values, hyperparams)
    elif canonical in {"ETS", "ExponentialSmoothing", "HoltWinters"}:
        seasonal = hyperparams.get("seasonal")
        period = _positive_period(hyperparams, required=seasonal is not None)
        fitted = StatsmodelsExponentialSmoothing(
            values,
            trend=hyperparams.get("trend", "add"),
            damped_trend=bool(hyperparams.get("damped_trend", False)),
            seasonal=seasonal,
            seasonal_periods=period,
            initialization_method=str(
                hyperparams.get("initialization_method", "estimated")
            ),
        ).fit(optimized=True)
    elif canonical == "SimpleExponentialSmoothing":
        fitted = SimpleExpSmoothing(
            values,
            initialization_method=str(
                hyperparams.get("initialization_method", "estimated")
            ),
        ).fit(optimized=True)
    elif canonical == "Holt":
        fitted = Holt(
            values,
            damped_trend=bool(hyperparams.get("damped_trend", False)),
            exponential=bool(hyperparams.get("exponential", False)),
            initialization_method=str(
                hyperparams.get("initialization_method", "estimated")
            ),
        ).fit(optimized=True)
    elif canonical == "Theta":
        period = _positive_period(hyperparams, required=False)
        fitted = ThetaModel(
            values,
            period=period,
            deseasonalize=bool(hyperparams.get("deseasonalize", period is not None)),
            use_test=bool(hyperparams.get("use_test", False)),
            difference=bool(hyperparams.get("difference", False)),
        ).fit()
    else:
        raise AssertionError(f"Unhandled statistical model {canonical}")
    return canonical, fitted


def save_statistical_result(fitted: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_statistical_result(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def forecast_statistical_result(fitted: Any, steps: int) -> np.ndarray:
    if steps < 0:
        raise ValueError("Statistical forecast steps must be non-negative")
    values = np.asarray(fitted.forecast(steps=steps), dtype=np.float64).reshape(-1)
    if len(values) != steps or not np.isfinite(values).all():
        raise ValueError("Statistical model returned invalid forecast values")
    return values
