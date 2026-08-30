"""Tests for the statsmodels-backed forecast registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adacast_ml.statistical import (
    STATISTICAL_MODEL_NAMES,
    fit_statistical_model,
    forecast_statistical_result,
    is_statistical_model,
    load_statistical_result,
    normalize_statistical_model_name,
    save_statistical_result,
)


@pytest.fixture
def seasonal_series() -> np.ndarray:
    index = np.arange(96, dtype=np.float64)
    return np.sin(index * 2 * np.pi / 12) + index * 0.01


def model_hyperparams(model_name: str) -> dict[str, object]:
    if model_name == "AutoARIMA":
        return {"candidate_orders": [[1, 0, 0], [0, 1, 1]]}
    if model_name == "HoltWinters":
        return {"trend": "add", "seasonal": "add", "seasonal_periods": 12}
    if model_name == "Theta":
        return {"seasonal_periods": 12}
    return {}


def test_statistical_registry_is_case_and_separator_insensitive() -> None:
    assert normalize_statistical_model_name("auto_arima") == "AutoARIMA"
    assert normalize_statistical_model_name("simple-exponential-smoothing") == (
        "SimpleExponentialSmoothing"
    )
    assert is_statistical_model("HOLT WINTERS")
    assert not is_statistical_model("SARIMA")
    assert not is_statistical_model("Prophet")


def test_sarima_fit_is_rejected_as_unsupported(
    seasonal_series: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="Unsupported statistical model"):
        fit_statistical_model("SARIMA", seasonal_series, {})


@pytest.mark.parametrize("model_name", STATISTICAL_MODEL_NAMES)
def test_every_registered_statistical_model_fits_and_forecasts(
    model_name: str,
    seasonal_series: np.ndarray,
) -> None:
    canonical, fitted = fit_statistical_model(
        model_name,
        seasonal_series,
        model_hyperparams(model_name),
    )
    forecast = forecast_statistical_result(fitted, 6)

    assert canonical == model_name
    assert forecast.shape == (6,)
    assert np.isfinite(forecast).all()


def test_fitted_statistical_result_round_trips_without_refit(
    tmp_path: Path,
    seasonal_series: np.ndarray,
) -> None:
    _, fitted = fit_statistical_model("ETS", seasonal_series, {})
    artifact_path = tmp_path / "statistical-results.pkl"
    save_statistical_result(fitted, artifact_path)
    loaded = load_statistical_result(artifact_path)

    assert np.allclose(
        forecast_statistical_result(loaded, 4),
        forecast_statistical_result(fitted, 4),
    )
