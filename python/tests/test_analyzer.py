"""Tests for the fixed quantitative analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from adacast_ml import analyzer


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SYNTHETIC_CSV = FIXTURES_DIR / "synthetic_etth1.csv"


def make_ar1_series(length: int = 512, phi: float = 0.6, seed: int = 42) -> np.ndarray:
    """Generate a deterministic stationary AR(1) sample."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.5, size=length)
    series = np.zeros(length, dtype=np.float64)
    for index in range(1, length):
        series[index] = phi * series[index - 1] + noise[index]
    return series


def test_imports_resolve() -> None:
    assert hasattr(analyzer, "adfuller")
    assert hasattr(analyzer, "kpss")
    assert hasattr(analyzer, "linregress")
    assert hasattr(analyzer, "ks_2samp")


def test_analyze_trend_returns_expected_shape() -> None:
    result = analyzer.analyze_trend(np.linspace(0.0, 10.0, 200))

    assert set(result) == {"slope", "r_squared", "direction"}
    assert result["direction"] == "upward"
    assert result["slope"] > 0
    assert 0.99 <= result["r_squared"] <= 1.0


def test_analyze_stationarity_returns_expected_shape() -> None:
    result = analyzer.analyze_stationarity(make_ar1_series())

    expected = {
        "adf_statistic",
        "adf_pvalue",
        "adf_conclusion",
        "kpss_statistic",
        "kpss_pvalue",
        "kpss_conclusion",
        "overall",
    }
    assert expected.issubset(result)
    assert result["adf_conclusion"] in {"stationary", "non-stationary"}
    assert result["kpss_conclusion"] in {"stationary", "non-stationary"}
    assert result["overall"] in {"stationary", "non-stationary", "uncertain"}


def test_analyze_seasonality_returns_expected_shape() -> None:
    x = np.arange(0, 24 * 20, dtype=np.float64)
    series = np.sin(2 * np.pi * x / 24.0)

    result = analyzer.analyze_seasonality(series)

    assert set(result) == {"dominant_period", "strength", "method"}
    assert result["method"] == "autocorrelation"
    assert result["dominant_period"] == 24
    assert 0.0 <= result["strength"] <= 1.0


def test_analyze_volatility_returns_expected_shape() -> None:
    result = analyzer.analyze_volatility(np.linspace(1.0, 100.0, 100))

    assert set(result) == {"std", "cv", "rolling_std_mean"}
    assert result["std"] > 0
    assert result["cv"] > 0
    assert result["rolling_std_mean"] >= 0


def test_analyze_missing_data_returns_expected_shape() -> None:
    frame = pd.DataFrame({"OT": [1.0, np.nan, np.nan, 4.0, np.nan]})

    result = analyzer.analyze_missing_data(frame, "OT")

    assert set(result) == {"count", "rate", "longest_gap"}
    assert result == {"count": 3, "rate": 0.6, "longest_gap": 2}


def test_analyze_anomaly_density_returns_expected_shape() -> None:
    series = np.concatenate([np.zeros(99, dtype=np.float64), np.array([1000.0])])

    result = analyzer.analyze_anomaly_density(series)

    assert set(result) == {"count", "rate", "method"}
    assert result["method"] == "iqr_3sigma"
    assert result["count"] == 1
    assert result["rate"] == 0.01


def test_analyze_distribution_shifts_returns_expected_shape() -> None:
    rng = np.random.default_rng(42)
    series = np.concatenate(
        [
            rng.normal(loc=0.0, scale=1.0, size=200),
            rng.normal(loc=5.0, scale=1.0, size=200),
        ]
    )

    result = analyzer.analyze_distribution_shifts(series)

    assert set(result) == {"detected", "n_changepoints", "method", "changepoint_indices"}
    assert result["method"] == "ks_test_windows"
    assert result["detected"] is True
    assert result["n_changepoints"] >= 1
    assert result["changepoint_indices"]


def test_analyzer(tmp_path: Path) -> None:
    out_path = tmp_path / "analysis.json"

    result = analyzer.analyze(
        dataset_path=SYNTHETIC_CSV,
        target_col="OT",
        time_col="date",
        out_path=out_path,
    )

    assert set(result) == {
        "dataset_path",
        "target_col",
        "time_col",
        "split_mode",
        "n_rows",
        "n_features",
        "analysis_scope",
        "split_time_ranges",
        "analysis",
        "analysis_timestamp",
        "analyzer_version",
    }
    assert set(result["analysis"]) == {
        "trend",
        "seasonality",
        "stationarity",
        "volatility",
        "missing_data",
        "anomaly_density",
        "distribution_shifts",
    }
    assert result["dataset_path"] == str(SYNTHETIC_CSV)
    assert result["target_col"] == "OT"
    assert result["time_col"] == "date"
    assert result["n_rows"] == 500
    assert result["n_features"] == 1
    assert result["analyzer_version"] == "1.3.0"
    assert result["split_mode"] == "ratio"
    assert result["analysis_scope"]["target_scope"] == "training_split_only"
    assert result["analysis_scope"]["target_row_count"] == 350
    assert result["analysis_scope"]["reserved_adjustment_row_count"] == 25
    assert result["split_time_ranges"]["validation"]["target_values_visible"] is False
    assert result["split_time_ranges"]["test"]["target_values_visible"] is False
    assert out_path.exists()

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == result


def test_preforecast_analysis_ignores_reserved_validation_and_test_targets(tmp_path: Path) -> None:
    row_count = 140
    timestamps = pd.date_range("2025-01-01", periods=row_count, freq="h")
    non_chronological_order = np.concatenate(
        [
            np.arange(56, row_count),
            np.arange(56),
        ]
    )
    frame = pd.DataFrame(
        {
            "date": timestamps[non_chronological_order],
            "target": np.sin(np.arange(row_count, dtype=np.float64) / 4.0),
            "known_exogenous": np.arange(row_count, dtype=np.float64),
        }
    )
    original_path = tmp_path / "original.csv"
    mutated_path = tmp_path / "mutated.csv"
    frame.to_csv(original_path, index=False)

    train_end = 98
    mutated = frame.copy()
    mutated.loc[train_end:, "target"] = 1_000_000 + np.arange(
        row_count - train_end,
        dtype=np.float64,
    )
    mutated.to_csv(mutated_path, index=False)

    original = analyzer.analyze(
        original_path,
        target_col="target",
        time_col="date",
        train_ratio=0.7,
        val_ratio=0.1,
    )
    changed = analyzer.analyze(
        mutated_path,
        target_col="target",
        time_col="date",
        train_ratio=0.7,
        val_ratio=0.1,
    )

    assert original["analysis_scope"]["target_end_index_exclusive"] == train_end
    assert original["analysis"] == changed["analysis"]
    assert original["analysis_scope"] == changed["analysis_scope"]
    assert original["split_time_ranges"] == changed["split_time_ranges"]


def test_analyzer_uses_inclusive_timestamp_boundaries(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="h"),
            "target": np.arange(40, dtype=np.float64),
        }
    )
    dataset_path = tmp_path / "timestamp-analysis.csv"
    frame.to_csv(dataset_path, index=False)

    result = analyzer.analyze(
        dataset_path=dataset_path,
        target_col="target",
        time_col="date",
        train_ratio=0.5,
        val_ratio=0.3,
        test_ratio=0.2,
        split_mode="timestamp",
        train_end_timestamp=str(frame["date"].iloc[19]),
        validation_end_timestamp=str(frame["date"].iloc[31]),
    )

    assert result["split_mode"] == "timestamp"
    assert result["analysis_scope"]["target_end_index_exclusive"] == 20
    assert result["analysis_scope"]["reserved_adjustment_end_index_exclusive"] == 26
    assert result["analysis_scope"]["validation_end_index_exclusive"] == 32
    assert result["analysis_scope"]["test_start_index"] == 32
