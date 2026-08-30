"""Tests for runner model-name normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from adacast_ml import runner


MODELS_DIR = Path(runner.__file__).resolve().parent / "models"


def test_lazy_model_dict_resolves_all_registered_models_case_insensitively() -> None:
    registry = runner.LazyModelDict(str(MODELS_DIR))

    assert "_foundation" not in registry.available_model_names
    assert registry.available_model_names

    for model_name in registry.available_model_names:
        assert registry.resolve_key(model_name.lower()) == model_name
        assert registry.resolve_key(model_name.upper()) == model_name
        assert registry.resolve_key(model_name.swapcase()) == model_name


def test_lazy_model_dict_getitem_uses_canonical_module_name() -> None:
    registry = runner.LazyModelDict(str(MODELS_DIR))

    assert registry["dlinear"] is registry["DLinear"]
    assert registry["PATCHTST"] is registry["PatchTST"]


@pytest.mark.parametrize("model_name", ["ARIMA", "arima", "ArImA"])
def test_arima_detection_is_case_insensitive(model_name: str) -> None:
    assert runner.is_arima_model(model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        "auto_arima",
        "ETS",
        "exponential-smoothing",
        "SimpleExponentialSmoothing",
        "Holt",
        "holt_winters",
        "Theta",
    ],
)
def test_statistical_model_detection_covers_the_registered_family(
    model_name: str,
) -> None:
    assert runner.is_statistical_model(model_name)
    assert runner.expected_reuse_artifact_kind(model_name) == (
        "statsmodels_statistical_results"
    )


def test_sarima_is_not_registered_as_a_statistical_model() -> None:
    assert not runner.is_statistical_model("SARIMA")


def test_prediction_layout_contract_rejects_legacy_arima_cache(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "actual": [1.0, 2.0],
            "predicted": [1.1, 2.1],
        }
    ).to_csv(tmp_path / "predictions.csv", index=False)

    assert (
        runner.prediction_layout_matches_contract(
            tmp_path,
            {"model": "ARIMA", "input_contract_version": 2},
        )
        is False
    )
    assert (
        runner.prediction_layout_matches_contract(
            tmp_path,
            {"model": "DLinear", "input_contract_version": 2},
        )
        is True
    )


def test_prediction_layout_contract_accepts_continuous_arima_coordinates(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "prediction_layout": ["continuous_origin", "continuous_origin"],
            "window_index": [0, 0],
            "horizon_index": [0, 1],
        }
    ).to_csv(tmp_path / "predictions.csv", index=False)

    assert runner.prediction_layout_matches_contract(
        tmp_path,
        {"model": "ARIMA", "input_contract_version": 2},
    )


def test_prediction_layout_contract_rejects_continuous_arima_in_current_contract(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "prediction_layout": ["continuous_origin", "continuous_origin"],
            "window_index": [0, 0],
            "horizon_index": [0, 1],
        }
    ).to_csv(tmp_path / "predictions.csv", index=False)

    assert not runner.prediction_layout_matches_contract(
        tmp_path,
        {"model": "ARIMA", "pred_len": 2, "input_contract_version": 4},
    )


def test_prediction_layout_contract_accepts_strided_statistical_coordinates(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "prediction_layout": ["sliding_window"] * 6,
            "window_index": [0, 0, 0, 1, 1, 1],
            "horizon_index": [0, 1, 2, 0, 1, 2],
        }
    ).to_csv(tmp_path / "predictions.csv", index=False)

    assert runner.prediction_layout_matches_contract(
        tmp_path,
        {
            "model": "ARIMA",
            "pred_len": 3,
            "window_stride": 2,
            "input_contract_version": 3,
        },
    )


def test_prediction_layout_contract_rejects_incomplete_sliding_window(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "prediction_layout": ["sliding_window"] * 5,
            "window_index": [0, 0, 0, 1, 1],
            "horizon_index": [0, 1, 2, 0, 1],
        }
    ).to_csv(tmp_path / "predictions.csv", index=False)

    assert not runner.prediction_layout_matches_contract(
        tmp_path,
        {"model": "DLinear", "pred_len": 3, "input_contract_version": 3},
    )


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("chronos", "Chronos"),
        ("CHRONOS2", "Chronos2"),
        ("timesfm", "TimesFM"),
        ("tirex", "TiRex"),
        ("timemoe", "TimeMoE"),
    ],
)
def test_foundation_model_names_are_canonicalized_case_insensitively(
    model_name: str,
    expected: str,
) -> None:
    assert runner.canonical_foundation_model_name(model_name) == expected
    assert runner.is_zero_shot_model(model_name)


def test_build_model_config_treats_lowercase_foundation_model_as_zero_shot() -> None:
    config = runner.build_model_config(
        {
            "model": "chronos",
            "seq_len": 24,
            "label_len": 12,
            "pred_len": 12,
        },
        num_features=1,
    )

    assert config.task_name == "zero_shot_forecast"


def test_flattened_model_params_are_treated_as_hyperparams() -> None:
    hyperparams = runner.extract_hyperparams(
        {
            "model": "patchtst",
            "dataset_path": "/tmp/data.csv",
            "target_col": "OT",
            "time_col": "date",
            "seq_len": 24,
            "label_len": 12,
            "pred_len": 12,
            "d_model": 64,
            "patch_len": 24,
            "stride": 12,
            "hyperparams": {"d_model": 128},
        }
    )

    assert hyperparams["d_model"] == 128
    assert hyperparams["patch_len"] == 24
    assert hyperparams["stride"] == 12
    assert "dataset_path" not in hyperparams


def test_model_constructor_receives_supported_hyperparams() -> None:
    class PatchLikeModel(nn.Module):
        def __init__(self, configs: object, patch_len: int = 16, stride: int = 8) -> None:
            super().__init__()
            self.configs = configs
            self.patch_len = patch_len
            self.stride = stride

    config = object()
    model = runner.instantiate_model(
        PatchLikeModel,
        config,
        {"patch_len": 24, "stride": 12, "unused": True},
    )

    assert model.configs is config
    assert model.patch_len == 24
    assert model.stride == 12


def test_model_specific_defaults_cover_required_ref_configuration() -> None:
    frets = runner.build_model_config(
        {"model": "FreTS", "seq_len": 24, "label_len": 12, "pred_len": 12},
        num_features=3,
    )
    segrnn = runner.build_model_config(
        {"model": "SegRNN", "seq_len": 24, "label_len": 12, "pred_len": 12},
        num_features=3,
    )
    timemixer = runner.build_model_config(
        {
            "model": "TimeMixer",
            "seq_len": 24,
            "label_len": 12,
            "pred_len": 12,
            "d_model": 8,
            "d_ff": 16,
            "e_layers": 1,
        },
        num_features=3,
    )

    assert frets.channel_independence == 1
    assert segrnn.seg_len == 12
    assert timemixer.decomp_method == "moving_avg"
    assert timemixer.use_norm == 1
    assert timemixer.down_sampling_method == "avg"
    assert timemixer.c_out == 3


def test_forecast_window_dataset_builds_frequency_aligned_time_features() -> None:
    payload = {
        "data": np.zeros((48, 2), dtype=np.float32),
        "timestamps": np.asarray(
            [f"2024-01-01 {hour // 4:02d}:{(hour % 4) * 15:02d}:00" for hour in range(48)]
        ),
    }
    dataset = runner.ForecastWindowDataset(
        payload,
        seq_len=24,
        label_len=12,
        pred_len=12,
        target_index=0,
        freq="t",
        input_contract_version=2,
    )

    _, _, seq_x_mark, seq_y_mark = dataset[0]
    assert seq_x_mark.shape == torch.Size([24, 5])
    assert seq_y_mark.shape == torch.Size([24, 5])
    assert torch.count_nonzero(seq_x_mark) > 0


def test_legacy_window_contract_preserves_four_zero_time_features() -> None:
    payload = {
        "data": np.zeros((48, 2), dtype=np.float32),
        "timestamps": np.asarray(
            [f"2024-01-01 {hour // 4:02d}:{(hour % 4) * 15:02d}:00" for hour in range(48)]
        ),
    }
    dataset = runner.ForecastWindowDataset(
        payload,
        seq_len=24,
        label_len=12,
        pred_len=12,
        target_index=0,
        freq="t",
        input_contract_version=1,
    )

    _, _, seq_x_mark, seq_y_mark = dataset[0]
    assert seq_x_mark.shape == torch.Size([24, 4])
    assert seq_y_mark.shape == torch.Size([24, 4])
    assert torch.count_nonzero(seq_x_mark) == 0


def test_forecast_window_dataset_applies_unified_window_stride() -> None:
    payload = {
        "data": np.arange(60, dtype=np.float32).reshape(30, 2),
        "timestamps": np.asarray([f"t-{index}" for index in range(30)]),
    }
    dataset = runner.ForecastWindowDataset(
        payload,
        seq_len=8,
        label_len=4,
        pred_len=4,
        target_index=0,
        window_stride=3,
    )

    assert len(dataset) == 7
    assert dataset.window_start(0) == 0
    assert dataset.window_start(1) == 3
    assert dataset.window_start(6) == 18
    timestamps, targets = dataset.prediction_targets(1)
    assert timestamps.tolist() == ["t-11", "t-12", "t-13", "t-14"]
    assert targets.tolist() == [22.0, 24.0, 26.0, 28.0]


def test_forecast_window_dataset_keeps_only_the_evaluated_tail_after_gap(
    tmp_path: Path,
) -> None:
    payload = {
        "data": np.arange(15, dtype=np.float32).reshape(15, 1),
        "timestamps": np.asarray([f"t-{index}" for index in range(15)]),
    }
    dataset = runner.ForecastWindowDataset(
        payload,
        seq_len=8,
        label_len=4,
        pred_len=4,
        target_index=0,
        forecast_gap=3,
    )

    _, seq_y, _, _ = dataset[0]
    timestamps, targets = dataset.prediction_targets(0)
    assert len(dataset) == 1
    assert dataset.model_pred_len == 7
    assert seq_y.shape == torch.Size([11, 1])
    assert timestamps.tolist() == ["t-11", "t-12", "t-13", "t-14"]
    assert targets.tolist() == [11.0, 12.0, 13.0, 14.0]

    class OrderedForecast(torch.nn.Module):
        def forward(self, seq_x, seq_x_mark, decoder_input, seq_y_mark):
            del seq_x, seq_x_mark, decoder_input, seq_y_mark
            return torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)

    predictions_path = runner.write_predictions(
        model=OrderedForecast(),
        dataset=dataset,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        pred_len=dataset.model_pred_len,
        include_gap_predictions=True,
    )
    predictions = pd.read_csv(predictions_path)
    assert predictions["timestamp"].tolist() == timestamps.tolist()
    assert predictions["predicted"].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert predictions["horizon_index"].tolist() == [0, 1, 2, 3]
    gap_predictions = pd.read_csv(tmp_path / "gap_predictions.csv")
    assert gap_predictions["timestamp"].tolist() == ["t-8", "t-9", "t-10"]
    assert gap_predictions["predicted"].tolist() == [0.0, 1.0, 2.0]
    assert gap_predictions["gap_index"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("window_stride", [0, -1, 1.5, True, "2"])
def test_forecast_window_dataset_rejects_invalid_window_stride(
    window_stride: object,
) -> None:
    payload = {
        "data": np.zeros((30, 1), dtype=np.float32),
        "timestamps": np.asarray([f"t-{index}" for index in range(30)]),
    }

    with pytest.raises(ValueError, match="window_stride must be a positive integer"):
        runner.ForecastWindowDataset(
            payload,
            seq_len=8,
            label_len=4,
            pred_len=4,
            target_index=0,
            window_stride=window_stride,
        )


@pytest.mark.parametrize("forecast_gap", [-1, 1.5, True, "2"])
def test_forecast_window_dataset_rejects_invalid_forecast_gap(
    forecast_gap: object,
) -> None:
    payload = {
        "data": np.zeros((30, 1), dtype=np.float32),
        "timestamps": np.asarray([f"t-{index}" for index in range(30)]),
    }

    with pytest.raises(ValueError, match="forecast_gap must be a nonnegative integer"):
        runner.ForecastWindowDataset(
            payload,
            seq_len=8,
            label_len=4,
            pred_len=4,
            target_index=0,
            forecast_gap=forecast_gap,
        )


def test_window_stride_is_not_forwarded_as_a_model_hyperparameter() -> None:
    hyperparams = runner.extract_hyperparams(
        {
            "model": "PatchTST",
            "seq_len": 24,
            "label_len": 12,
            "pred_len": 12,
            "forecast_gap": 5,
            "window_stride": 24,
            "stride": 8,
        }
    )

    assert "window_stride" not in hyperparams
    assert "forecast_gap" not in hyperparams
    assert hyperparams["stride"] == 8


def test_model_config_predicts_gap_plus_evaluated_horizon() -> None:
    config = runner.build_model_config(
        {
            "model": "DLinear",
            "seq_len": 24,
            "label_len": 12,
            "pred_len": 12,
            "forecast_gap": 5,
        },
        num_features=1,
    )

    assert config.pred_len == 17


def test_statistical_predictions_skip_gap_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "data": np.arange(15, dtype=np.float32).reshape(15, 1),
        "timestamps": np.asarray([f"t-{index}" for index in range(15)]),
        "target_start_index": 0,
        "forecast_offset": 0,
    }
    monkeypatch.setattr(
        runner,
        "forecast_statistical_result",
        lambda _fitted, steps: np.arange(steps, dtype=np.float64),
    )

    predictions_path = runner.write_statistical_predictions(
        fitted=object(),
        eval_payload=payload,
        run_dir=tmp_path,
        target_index=0,
        seq_len=8,
        label_len=4,
        pred_len=4,
        window_stride=1,
        forecast_gap=3,
        include_gap_predictions=True,
    )

    predictions = pd.read_csv(predictions_path)
    assert predictions["timestamp"].tolist() == ["t-11", "t-12", "t-13", "t-14"]
    assert predictions["predicted"].tolist() == [11.0, 12.0, 13.0, 14.0]
    gap_predictions = pd.read_csv(tmp_path / "gap_predictions.csv")
    assert gap_predictions["timestamp"].tolist() == ["t-8", "t-9", "t-10"]
    assert gap_predictions["predicted"].tolist() == [8.0, 9.0, 10.0]


def test_timemixer_channel_independent_output_matches_multivariate_width() -> None:
    config = runner.build_model_config(
        {
            "model": "TimeMixer",
            "seq_len": 24,
            "label_len": 12,
            "pred_len": 12,
            "d_model": 8,
            "d_ff": 16,
            "e_layers": 1,
            "freq": "t",
        },
        num_features=3,
    )
    model_class = runner.LazyModelDict(str(MODELS_DIR))["TimeMixer"]
    model = runner.instantiate_model(model_class, config, runner.extract_hyperparams(vars(config)))
    x = torch.zeros((2, 24, 3), dtype=torch.float32)
    x_mark = torch.zeros((2, 24, 5), dtype=torch.float32)
    y = torch.zeros((2, 24, 3), dtype=torch.float32)
    y_mark = torch.zeros((2, 24, 5), dtype=torch.float32)

    output = model(x, x_mark, y, y_mark)
    assert output.shape == torch.Size([2, 12, 3])
