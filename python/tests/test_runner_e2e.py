"""End-to-end tests for the experiment runner."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd

from adacast_ml.runner import load_spec, validate_reuse_contract
from adacast_ml.utils.hash import compute_file_hash, compute_spec_hash


PYTHON_DIR = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SYNTHETIC_CSV = FIXTURES_DIR / "synthetic_etth1.csv"


@pytest.fixture
def spec_path(tmp_path: Path) -> tuple[Path, dict, Path]:
    """Create a minimal DLinear spec for fast testing."""
    spec = {
        "model": "DLinear",
        "dataset_path": str(SYNTHETIC_CSV),
        "target_col": "OT",
        "time_col": "date",
        "freq": "h",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "adjustment_fit_holdout_ratio": 1 / 2,
        "seq_len": 24,
        "pred_len": 12,
        "label_len": 12,
        "phase": "forecast",
        "eval_split": "val",
        "seed": 42,
        "hyperparams": {
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 1,
            "d_layers": 1,
            "d_ff": 128,
            "dropout": 0.1,
            "train_epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "moving_avg": 25,
            "individual": False,
        },
        "metrics": ["mse", "mae", "wape", "mase"],
    }
    spec_file = tmp_path / "test_spec.json"
    spec_file.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec_file, spec, tmp_path


def run_runner(
    spec_path: Path,
    out_dir: Path,
    timeout: int = 120,
    assigned_run_id: str | None = None,
    reuse_model_from: Path | None = None,
) -> dict:
    """Invoke the runner as a subprocess and parse the final JSON line."""
    command = [
        sys.executable,
        "-m",
        "adacast_ml.runner",
        "--spec",
        str(spec_path),
        "--out-dir",
        str(out_dir / "runs"),
        "--evaluator-hash-store",
        str(out_dir / ".forecast" / "evaluator.sha256"),
    ]
    if assigned_run_id is not None:
        command.extend(["--run-id", assigned_run_id])
    if reuse_model_from is not None:
        command.extend(["--reuse-model-from", str(reuse_model_from)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PYTHON_DIR),
    )

    stdout_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    if stdout_lines:
        try:
            payload = json.loads(stdout_lines[-1])
            payload.setdefault("returncode", result.returncode)
            return payload
        except json.JSONDecodeError:
            pass

    return {
        "status": "error",
        "error": result.stderr,
        "stdout": result.stdout,
        "returncode": result.returncode,
    }


class TestRunnerE2E:
    def test_load_spec_defaults_validation_phase_to_forecast(
        self,
        tmp_path: Path,
    ) -> None:
        spec = {
            "model": "DLinear",
            "eval_split": "val",
            "adjustment_fit_holdout_ratio": 1 / 2,
        }
        spec_file = tmp_path / "missing-phase.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")

        normalized, canonical, run_id = load_spec(spec_file)

        assert normalized["phase"] == "forecast"
        assert normalized["eval_split"] == "val"
        assert normalized.get("window_stride", 1) == 1
        assert json.loads(canonical) == normalized
        assert run_id == compute_spec_hash(normalized)

    @pytest.mark.parametrize("window_stride", [0, -1, 1.5, True, "2", None])
    def test_load_spec_rejects_invalid_window_stride(
        self,
        tmp_path: Path,
        window_stride: object,
    ) -> None:
        spec_file = tmp_path / "invalid-window-stride.json"
        spec_file.write_text(
            json.dumps(
                {
                    "model": "DLinear",
                    "phase": "forecast",
                    "eval_split": "val",
                    "window_stride": window_stride,
                    "adjustment_fit_holdout_ratio": 1 / 2,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="window_stride must be a positive integer"):
            load_spec(spec_file)

    @pytest.mark.parametrize("forecast_gap", [-1, 1.5, True, "2", None])
    def test_load_spec_rejects_invalid_forecast_gap(
        self,
        tmp_path: Path,
        forecast_gap: object,
    ) -> None:
        spec_file = tmp_path / "invalid-forecast-gap.json"
        spec_file.write_text(
            json.dumps(
                {
                    "model": "DLinear",
                    "phase": "forecast",
                    "eval_split": "val",
                    "forecast_gap": forecast_gap,
                    "adjustment_fit_holdout_ratio": 1 / 2,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="forecast_gap must be a nonnegative integer"):
            load_spec(spec_file)

    @pytest.mark.parametrize(
        ("phase", "eval_split"),
        [
            ("forecast", "test"),
            ("post-forecast", "val"),
            ("pre-forecast", "val"),
        ],
    )
    def test_load_spec_rejects_invalid_execution_contract(
        self,
        tmp_path: Path,
        phase: str,
        eval_split: str,
    ) -> None:
        spec_file = tmp_path / f"invalid-{phase}-{eval_split}.json"
        spec_file.write_text(
            json.dumps(
                {
                    "model": "DLinear",
                    "phase": phase,
                    "eval_split": eval_split,
                    "adjustment_fit_holdout_ratio": 1 / 2,
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Invalid experiment execution contract"):
            load_spec(spec_file)

    def test_reuse_contract_normalizes_legacy_missing_source_phase(self) -> None:
        source_spec = {
            "model": "DLinear",
            "eval_split": "val",
            "adjustment_fit_holdout_ratio": 1 / 2,
        }
        replay_spec = {
            **source_spec,
            "eval_split": "adjustment_fit",
        }

        validate_reuse_contract(replay_spec, source_spec)

    def test_reuse_contract_requires_the_same_forecast_gap(self) -> None:
        source_spec = {
            "model": "DLinear",
            "phase": "forecast",
            "eval_split": "val",
            "forecast_gap": 3,
            "adjustment_fit_holdout_ratio": 1 / 2,
        }
        replay_spec = {**source_spec, "eval_split": "adjustment_fit"}

        validate_reuse_contract(replay_spec, source_spec)
        with pytest.raises(ValueError, match="must exactly match"):
            validate_reuse_contract({**replay_spec, "forecast_gap": 4}, source_spec)

    @pytest.mark.slow
    def test_full_pipeline_dlinear(self, spec_path: tuple[Path, dict, Path]) -> None:
        spec_file, _spec, tmp_dir = spec_path
        result = run_runner(spec_file, tmp_dir)

        assert result["status"] in ("success", "cached"), f"Runner failed: {result}"
        assert "run_id" in result
        assert len(result["run_id"]) == 64

    @pytest.mark.slow
    def test_full_pipeline_dlinear_with_timestamp_split(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        spec_file, spec, tmp_dir = spec_path
        source_frame = pd.read_csv(SYNTHETIC_CSV)
        timestamp_spec = {
            **spec,
            "split_mode": "timestamp",
            "train_end_timestamp": str(source_frame["date"].iloc[349]),
            "validation_end_timestamp": str(source_frame["date"].iloc[399]),
        }
        spec_file.write_text(json.dumps(timestamp_spec, indent=2), encoding="utf-8")

        result = run_runner(spec_file, tmp_dir)

        assert result["status"] == "success", f"Runner failed: {result}"
        run_dir = Path(result["out_dir"])
        predictions = pd.read_csv(run_dir / "predictions.csv")
        expected_validation_timestamps = set(
            source_frame["date"].iloc[375:400].astype(str)
        )
        assert set(predictions["timestamp"]).issubset(
            expected_validation_timestamps
        )

    @pytest.mark.slow
    def test_full_pipeline_dlinear_applies_window_stride(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        spec_file, spec, tmp_dir = spec_path
        spec_file.write_text(
            json.dumps({**spec, "window_stride": 5}, indent=2),
            encoding="utf-8",
        )

        result = run_runner(spec_file, tmp_dir)

        assert result["status"] == "success", f"Runner failed: {result}"
        predictions = pd.read_csv(Path(result["out_dir"]) / "predictions.csv")
        assert predictions["window_index"].unique().tolist() == [0, 1, 2]
        assert predictions.groupby("window_index").size().tolist() == [12, 12, 12]
        window_starts = (
            predictions.groupby("window_index")["timestamp"].first().tolist()
        )
        source_frame = pd.read_csv(SYNTHETIC_CSV)
        assert window_starts == [
            str(source_frame["date"].iloc[375]),
            str(source_frame["date"].iloc[380]),
            str(source_frame["date"].iloc[385]),
        ]

    @pytest.mark.slow
    def test_full_pipeline_dlinear_predicts_gap_but_evaluates_only_tail(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        spec_file, spec, tmp_dir = spec_path
        spec_file.write_text(
            json.dumps({**spec, "forecast_gap": 3}, indent=2),
            encoding="utf-8",
        )

        result = run_runner(spec_file, tmp_dir)

        assert result["status"] == "success", result
        predictions = pd.read_csv(Path(result["out_dir"]) / "predictions.csv")
        source_frame = pd.read_csv(SYNTHETIC_CSV)
        assert predictions.groupby("window_index").size().unique().tolist() == [12]
        assert predictions.iloc[0]["timestamp"] == str(source_frame["date"].iloc[378])
        assert predictions["timestamp"].min() == str(source_frame["date"].iloc[378])
        assert predictions["timestamp"].max() == str(source_frame["date"].iloc[399])
        assert predictions["horizon_index"].min() == 0
        assert predictions["horizon_index"].max() == 11

    @pytest.mark.slow
    def test_explicit_default_stride_matches_legacy_omission(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        spec_file, spec, tmp_dir = spec_path
        legacy_result = run_runner(spec_file, tmp_dir / "legacy")
        explicit_spec_file = tmp_dir / "explicit-stride-one.json"
        explicit_spec_file.write_text(
            json.dumps({**spec, "window_stride": 1}, indent=2),
            encoding="utf-8",
        )
        explicit_result = run_runner(explicit_spec_file, tmp_dir / "explicit")

        assert legacy_result["status"] == "success", legacy_result
        assert explicit_result["status"] == "success", explicit_result
        legacy_predictions = pd.read_csv(
            Path(legacy_result["out_dir"]) / "predictions.csv"
        )
        explicit_predictions = pd.read_csv(
            Path(explicit_result["out_dir"]) / "predictions.csv"
        )
        pd.testing.assert_frame_equal(legacy_predictions, explicit_predictions)
        assert legacy_result["metrics"] == explicit_result["metrics"]

    @pytest.mark.slow
    def test_explicit_zero_forecast_gap_matches_legacy_omission(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        spec_file, spec, tmp_dir = spec_path
        legacy_result = run_runner(spec_file, tmp_dir / "legacy-gap")
        explicit_spec_file = tmp_dir / "explicit-gap-zero.json"
        explicit_spec_file.write_text(
            json.dumps({**spec, "forecast_gap": 0}, indent=2),
            encoding="utf-8",
        )
        explicit_result = run_runner(explicit_spec_file, tmp_dir / "explicit-gap")

        assert legacy_result["status"] == "success", legacy_result
        assert explicit_result["status"] == "success", explicit_result
        legacy_predictions = pd.read_csv(
            Path(legacy_result["out_dir"]) / "predictions.csv"
        )
        explicit_predictions = pd.read_csv(
            Path(explicit_result["out_dir"]) / "predictions.csv"
        )
        pd.testing.assert_frame_equal(legacy_predictions, explicit_predictions)
        assert legacy_result["metrics"] == explicit_result["metrics"]

    @pytest.mark.slow
    def test_full_pipeline_dlinear_defaults_missing_phase(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        spec_file, spec, tmp_dir = spec_path
        spec_without_phase = {key: value for key, value in spec.items() if key != "phase"}
        spec_file.write_text(json.dumps(spec_without_phase, indent=2), encoding="utf-8")

        result = run_runner(spec_file, tmp_dir)

        assert result["status"] == "success", f"Runner failed: {result}"
        run_dir = Path(result["out_dir"])
        stored_spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        metadata = json.loads(
            (run_dir / "model" / "metadata.json").read_text(encoding="utf-8")
        )
        assert stored_spec["phase"] == "forecast"
        assert stored_spec["eval_split"] == "val"
        assert metadata["phase"] == "forecast"
        assert metadata["eval_split"] == "val"

    @pytest.mark.slow
    def test_produces_all_artifacts(self, spec_path: tuple[Path, dict, Path]) -> None:
        spec_file, _spec, tmp_dir = spec_path
        result = run_runner(spec_file, tmp_dir)

        assert result["status"] in ("success", "cached")
        run_dir = Path(result["out_dir"])
        assert (run_dir / "spec.json").exists()
        assert (run_dir / "predictions.csv").exists()
        assert (run_dir / "actual.npy").exists()
        assert (run_dir / "pred.npy").exists()
        assert (run_dir / "eval.json").exists()
        assert (run_dir / "train.log").exists()
        assert (run_dir / "train_scale.json").exists()
        predictions = pd.read_csv(run_dir / "predictions.csv")
        assert set(
            ["prediction_layout", "window_index", "horizon_index"]
        ).issubset(predictions.columns)
        assert predictions["prediction_layout"].unique().tolist() == [
            "sliding_window"
        ]
        assert predictions.groupby("window_index")["horizon_index"].apply(
            list
        ).iloc[0] == list(range(12))

        checkpoint_path = run_dir / "model" / "checkpoint.pth"
        metadata_path = run_dir / "model" / "metadata.json"
        assert checkpoint_path.exists()
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["kind"] == "checkpoint"
        assert metadata["path"] == "model/checkpoint.pth"
        assert metadata["sha256"] == compute_file_hash(checkpoint_path)
        assert metadata["model"] == "DLinear"
        assert metadata["eval_split"] == "val"
        assert metadata["phase"] == "forecast"
        assert metadata["trained_on"] == "train"
        assert metadata["format"] == "torch_state_dict"

        cached_result = run_runner(spec_file, tmp_dir)
        assert cached_result["status"] == "cached"
        assert Path(cached_result["out_dir"]) == run_dir
        assert checkpoint_path.exists()
        assert metadata_path.exists()

    @pytest.mark.slow
    def test_eval_json_has_metrics(self, spec_path: tuple[Path, dict, Path]) -> None:
        spec_file, _spec, tmp_dir = spec_path
        result = run_runner(spec_file, tmp_dir)

        assert result["status"] in ("success", "cached")
        eval_json = json.loads((Path(result["out_dir"]) / "eval.json").read_text(encoding="utf-8"))
        assert "metrics" in eval_json
        for metric in ("mse", "mae", "wape"):
            assert isinstance(eval_json["metrics"][metric], float)

    @pytest.mark.slow
    def test_deterministic_scores(self, spec_path: tuple[Path, dict, Path]) -> None:
        spec_file, _spec, tmp_dir = spec_path
        first_result = run_runner(spec_file, tmp_dir)
        second_result = run_runner(spec_file, tmp_dir)

        assert first_result["status"] in ("success", "cached")
        assert second_result["status"] == "cached"
        assert first_result["run_id"] == second_result["run_id"]

    @pytest.mark.slow
    def test_run_id_is_spec_hash(self, spec_path: tuple[Path, dict, Path]) -> None:
        spec_file, spec, tmp_dir = spec_path
        result = run_runner(spec_file, tmp_dir)
        assert result["run_id"] == compute_spec_hash(spec)

    @pytest.mark.slow
    def test_adjustment_fit_reuses_checkpoint_without_training(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        source_spec_file, source_spec, tmp_dir = spec_path
        source_result = run_runner(source_spec_file, tmp_dir)
        assert source_result["status"] == "success"

        source_run_dir = Path(source_result["out_dir"])
        checkpoint_path = source_run_dir / "model" / "checkpoint.pth"
        checkpoint_mtime = checkpoint_path.stat().st_mtime_ns

        fit_spec = {**source_spec, "eval_split": "adjustment_fit"}
        fit_spec_file = tmp_dir / "adjustment-fit-spec.json"
        fit_spec_file.write_text(json.dumps(fit_spec, indent=2), encoding="utf-8")
        fit_result = run_runner(
            fit_spec_file,
            tmp_dir / "adjustment",
            reuse_model_from=source_run_dir,
        )

        assert fit_result["status"] == "success", fit_result
        fit_run_dir = Path(fit_result["out_dir"])
        provenance = json.loads(
            (fit_run_dir / "model" / "reuse.json").read_text(encoding="utf-8")
        )
        train_log = (fit_run_dir / "train.log").read_text(encoding="utf-8")
        assert provenance["source_run_id"] == source_result["run_id"]
        assert provenance["training_performed"] is False
        assert provenance["source_artifact_kind"] == "torch_state_dict"
        assert provenance["source_artifact_sha256"] == compute_file_hash(checkpoint_path)
        assert '"type": "train_start"' not in train_log
        assert '"type": "epoch"' not in train_log
        assert checkpoint_path.stat().st_mtime_ns == checkpoint_mtime
        assert (fit_run_dir / "predictions.csv").exists()
        assert (fit_run_dir / "eval.json").exists()
        source_frame = pd.read_csv(SYNTHETIC_CSV)
        fit_predictions = pd.read_csv(fit_run_dir / "predictions.csv")
        expected_evidence_timestamps = set(source_frame["date"].iloc[350:375].astype(str))
        assert set(fit_predictions["timestamp"]).issubset(expected_evidence_timestamps)
        assert fit_predictions["timestamp"].iloc[0] == str(source_frame["date"].iloc[350])
        assert fit_predictions["timestamp"].iloc[-1] == str(source_frame["date"].iloc[374])

    @pytest.mark.slow
    def test_post_forecast_test_reuses_selected_checkpoint_without_training(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        source_spec_file, source_spec, tmp_dir = spec_path
        source_result = run_runner(source_spec_file, tmp_dir)
        assert source_result["status"] == "success"

        source_run_dir = Path(source_result["out_dir"])
        checkpoint_path = source_run_dir / "model" / "checkpoint.pth"
        checkpoint_mtime = checkpoint_path.stat().st_mtime_ns
        test_spec = {
            **source_spec,
            "phase": "post-forecast",
            "eval_split": "test",
        }
        test_spec_file = tmp_dir / "post-forecast-test-spec.json"
        test_spec_file.write_text(json.dumps(test_spec, indent=2), encoding="utf-8")
        stale_test_model_dir = tmp_dir / "runs" / compute_spec_hash(test_spec) / "model"
        stale_test_model_dir.mkdir(parents=True, exist_ok=True)
        (stale_test_model_dir / "checkpoint.pth").write_text(
            "stale-retrained-test-checkpoint",
            encoding="utf-8",
        )

        test_result = run_runner(
            test_spec_file,
            tmp_dir,
            reuse_model_from=source_run_dir,
        )

        assert test_result["status"] == "success", test_result
        test_run_dir = Path(test_result["out_dir"])
        provenance = json.loads(
            (test_run_dir / "model" / "reuse.json").read_text(encoding="utf-8")
        )
        train_log = (test_run_dir / "train.log").read_text(encoding="utf-8")
        assert provenance["source_run_id"] == source_result["run_id"]
        assert provenance["training_performed"] is False
        assert provenance["evaluated_split"] == "test"
        assert provenance["source_artifact_kind"] == "torch_state_dict"
        assert provenance["source_artifact_sha256"] == compute_file_hash(checkpoint_path)
        assert '"type": "train_start"' not in train_log
        assert '"type": "epoch"' not in train_log
        assert checkpoint_path.stat().st_mtime_ns == checkpoint_mtime
        assert not (test_run_dir / "model" / "checkpoint.pth").exists()
        assert (test_run_dir / "predictions.csv").exists()
        assert (test_run_dir / "eval.json").exists()

        cached_result = run_runner(
            test_spec_file,
            tmp_dir,
            reuse_model_from=source_run_dir,
        )
        assert cached_result["status"] == "cached"

    @pytest.mark.slow
    def test_forecast_gap_survives_test_reuse_without_retraining(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        source_spec_file, source_spec, tmp_dir = spec_path
        gap_spec = {**source_spec, "forecast_gap": 3}
        source_spec_file.write_text(json.dumps(gap_spec, indent=2), encoding="utf-8")
        source_result = run_runner(source_spec_file, tmp_dir)
        assert source_result["status"] == "success", source_result

        source_run_dir = Path(source_result["out_dir"])
        checkpoint_path = source_run_dir / "model" / "checkpoint.pth"
        checkpoint_mtime = checkpoint_path.stat().st_mtime_ns
        test_spec = {**gap_spec, "phase": "post-forecast", "eval_split": "test"}
        test_spec_file = tmp_dir / "post-forecast-gap-test-spec.json"
        test_spec_file.write_text(json.dumps(test_spec, indent=2), encoding="utf-8")
        test_result = run_runner(
            test_spec_file,
            tmp_dir,
            reuse_model_from=source_run_dir,
        )

        assert test_result["status"] == "success", test_result
        test_run_dir = Path(test_result["out_dir"])
        provenance = json.loads(
            (test_run_dir / "model" / "reuse.json").read_text(encoding="utf-8")
        )
        train_log = (test_run_dir / "train.log").read_text(encoding="utf-8")
        predictions = pd.read_csv(test_run_dir / "predictions.csv")
        assert provenance["source_run_id"] == source_result["run_id"]
        assert provenance["training_performed"] is False
        assert checkpoint_path.stat().st_mtime_ns == checkpoint_mtime
        assert '"type": "train_start"' not in train_log
        assert '"type": "epoch"' not in train_log
        assert predictions.groupby("window_index").size().unique().tolist() == [12]
        assert predictions["horizon_index"].min() == 0
        assert predictions["horizon_index"].max() == 11

    @pytest.mark.slow
    def test_arima_adjustment_fit_reuses_saved_results_without_refit(
        self,
        tmp_path: Path,
    ) -> None:
        source_spec = {
            "model": "ARIMA",
            "dataset_path": str(SYNTHETIC_CSV),
            "target_col": "OT",
            "time_col": "date",
            "freq": "h",
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
            "adjustment_fit_holdout_ratio": 1 / 2,
            "seq_len": 24,
            "pred_len": 12,
            "label_len": 12,
            "window_stride": 5,
            "input_contract_version": 4,
            "phase": "forecast",
            "eval_split": "val",
            "seed": 42,
            "hyperparams": {"order": [1, 0, 0]},
            "metrics": ["mse", "mae", "wape", "mase"],
        }
        source_spec_file = tmp_path / "arima-source.json"
        source_spec_file.write_text(json.dumps(source_spec, indent=2), encoding="utf-8")
        source_result = run_runner(source_spec_file, tmp_path)
        assert source_result["status"] == "success", source_result

        source_run_dir = Path(source_result["out_dir"])
        cached_source_result = run_runner(source_spec_file, tmp_path)
        assert cached_source_result["status"] == "cached", cached_source_result
        fitted_path = source_run_dir / "model" / "arima-results.pkl"
        fitted_mtime = fitted_path.stat().st_mtime_ns
        fit_spec_file = tmp_path / "arima-adjustment-fit.json"
        fit_spec_file.write_text(
            json.dumps({**source_spec, "eval_split": "adjustment_fit"}, indent=2),
            encoding="utf-8",
        )

        fit_result = run_runner(
            fit_spec_file,
            tmp_path / "arima-adjustment",
            reuse_model_from=source_run_dir,
        )

        assert fit_result["status"] == "success", fit_result
        fit_run_dir = Path(fit_result["out_dir"])
        provenance = json.loads(
            (fit_run_dir / "model" / "reuse.json").read_text(encoding="utf-8")
        )
        train_log = (fit_run_dir / "train.log").read_text(encoding="utf-8")
        assert provenance["source_artifact_kind"] == "statsmodels_arima_results"
        assert provenance["training_performed"] is False
        assert '"type": "train_start"' not in train_log
        assert fitted_path.stat().st_mtime_ns == fitted_mtime
        source_frame = pd.read_csv(SYNTHETIC_CSV)
        fit_predictions = pd.read_csv(fit_run_dir / "predictions.csv")
        assert fit_predictions["timestamp"].iloc[0] == str(source_frame["date"].iloc[350])
        assert fit_predictions["timestamp"].iloc[-1] == str(source_frame["date"].iloc[371])
        assert fit_predictions["prediction_layout"].unique().tolist() == [
            "sliding_window"
        ]
        assert fit_predictions["window_index"].unique().tolist() == [0, 1, 2]
        assert fit_predictions.groupby("window_index").size().tolist() == [12, 12, 12]
        assert fit_predictions.groupby("window_index")["horizon_index"].apply(list).tolist() == [
            list(range(12)),
            list(range(12)),
            list(range(12)),
        ]

        test_spec_file = tmp_path / "arima-post-forecast-test.json"
        test_spec_file.write_text(
            json.dumps(
                {
                    **source_spec,
                    "phase": "post-forecast",
                    "eval_split": "test",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        test_result = run_runner(
            test_spec_file,
            tmp_path,
            reuse_model_from=source_run_dir,
        )
        assert test_result["status"] == "success", test_result
        test_run_dir = Path(test_result["out_dir"])
        test_provenance = json.loads(
            (test_run_dir / "model" / "reuse.json").read_text(encoding="utf-8")
        )
        test_log = (test_run_dir / "train.log").read_text(encoding="utf-8")
        assert test_provenance["source_artifact_kind"] == "statsmodels_arima_results"
        assert test_provenance["source_artifact_sha256"] == compute_file_hash(fitted_path)
        assert test_provenance["training_performed"] is False
        assert test_provenance["evaluated_split"] == "test"
        assert '"type": "train_start"' not in test_log
        assert fitted_path.stat().st_mtime_ns == fitted_mtime

    @pytest.mark.slow
    def test_ets_reuses_saved_results_for_adjustment_fit_and_test(
        self,
        tmp_path: Path,
    ) -> None:
        source_spec = {
            "model": "ETS",
            "dataset_path": str(SYNTHETIC_CSV),
            "target_col": "OT",
            "time_col": "date",
            "freq": "h",
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
            "adjustment_fit_holdout_ratio": 1 / 2,
            "seq_len": 24,
            "pred_len": 12,
            "label_len": 12,
            "input_contract_version": 4,
            "phase": "forecast",
            "eval_split": "val",
            "seed": 42,
            "hyperparams": {"trend": "add"},
            "metrics": ["mse", "mae", "wape", "mase"],
        }
        source_spec_file = tmp_path / "ets-source.json"
        source_spec_file.write_text(json.dumps(source_spec, indent=2), encoding="utf-8")
        source_result = run_runner(source_spec_file, tmp_path)
        assert source_result["status"] == "success", source_result

        source_run_dir = Path(source_result["out_dir"])
        fitted_path = source_run_dir / "model" / "statistical-results.pkl"
        fitted_mtime = fitted_path.stat().st_mtime_ns
        metadata = json.loads(
            (source_run_dir / "model" / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["kind"] == "statistical_results"
        assert metadata["format"] == "statsmodels_results_pickle"

        fit_spec_file = tmp_path / "ets-adjustment-fit.json"
        fit_spec_file.write_text(
            json.dumps({**source_spec, "eval_split": "adjustment_fit"}, indent=2),
            encoding="utf-8",
        )
        fit_result = run_runner(
            fit_spec_file,
            tmp_path / "ets-adjustment",
            reuse_model_from=source_run_dir,
        )
        assert fit_result["status"] == "success", fit_result
        fit_run_dir = Path(fit_result["out_dir"])
        fit_provenance = json.loads(
            (fit_run_dir / "model" / "reuse.json").read_text(encoding="utf-8")
        )
        assert fit_provenance["source_artifact_kind"] == (
            "statsmodels_statistical_results"
        )
        assert fit_provenance["training_performed"] is False
        assert '"type": "train_start"' not in (
            fit_run_dir / "train.log"
        ).read_text(encoding="utf-8")

        test_spec_file = tmp_path / "ets-post-forecast-test.json"
        test_spec_file.write_text(
            json.dumps(
                {**source_spec, "phase": "post-forecast", "eval_split": "test"},
                indent=2,
            ),
            encoding="utf-8",
        )
        test_result = run_runner(
            test_spec_file,
            tmp_path,
            reuse_model_from=source_run_dir,
        )
        assert test_result["status"] == "success", test_result
        test_run_dir = Path(test_result["out_dir"])
        test_provenance = json.loads(
            (test_run_dir / "model" / "reuse.json").read_text(encoding="utf-8")
        )
        assert test_provenance["source_artifact_kind"] == (
            "statsmodels_statistical_results"
        )
        assert test_provenance["source_artifact_sha256"] == compute_file_hash(
            fitted_path
        )
        assert test_provenance["training_performed"] is False
        assert '"type": "train_start"' not in (
            test_run_dir / "train.log"
        ).read_text(encoding="utf-8")
        assert fitted_path.stat().st_mtime_ns == fitted_mtime

    @pytest.mark.slow
    def test_post_forecast_test_rejects_tampered_source_checkpoint(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        source_spec_file, source_spec, tmp_dir = spec_path
        source_result = run_runner(source_spec_file, tmp_dir)
        assert source_result["status"] == "success"

        source_run_dir = Path(source_result["out_dir"])
        checkpoint_path = source_run_dir / "model" / "checkpoint.pth"
        checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
        test_spec_file = tmp_dir / "tampered-source-test.json"
        test_spec_file.write_text(
            json.dumps(
                {
                    **source_spec,
                    "phase": "post-forecast",
                    "eval_split": "test",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        result = run_runner(
            test_spec_file,
            tmp_dir,
            reuse_model_from=source_run_dir,
        )

        assert result["status"] == "error"
        assert "does not match its validation metadata" in result["error"]

    def test_post_forecast_test_rejects_missing_reuse_source(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        _source_spec_file, source_spec, tmp_dir = spec_path
        test_spec_file = tmp_dir / "test-without-reuse.json"
        test_spec_file.write_text(
            json.dumps(
                {
                    **source_spec,
                    "phase": "post-forecast",
                    "eval_split": "test",
                }
            ),
            encoding="utf-8",
        )

        result = run_runner(test_spec_file, tmp_dir)

        assert result["status"] == "error"
        assert "requires --reuse-model-from" in result["error"]

    @pytest.mark.slow
    def test_post_forecast_test_rejects_spec_that_does_not_match_source(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        source_spec_file, source_spec, tmp_dir = spec_path
        source_result = run_runner(source_spec_file, tmp_dir)
        assert source_result["status"] == "success"

        test_spec_file = tmp_dir / "mismatched-post-forecast-test.json"
        test_spec_file.write_text(
            json.dumps(
                {
                    **source_spec,
                    "phase": "post-forecast",
                    "eval_split": "test",
                    "pred_len": int(source_spec["pred_len"]) + 1,
                }
            ),
            encoding="utf-8",
        )

        result = run_runner(
            test_spec_file,
            tmp_dir,
            reuse_model_from=Path(source_result["out_dir"]),
        )

        assert result["status"] == "error"
        assert "must exactly match its source validation spec" in result["error"]

    @pytest.mark.slow
    def test_assigned_run_id_keeps_unicode_path_artifacts_in_one_directory(
        self,
        spec_path: tuple[Path, dict, Path],
    ) -> None:
        spec_file, spec, tmp_dir = spec_path
        unicode_dir = tmp_dir / "桌面大文件" / "学习资料"
        unicode_dir.mkdir(parents=True)
        unicode_dataset = unicode_dir / "光伏数据.csv"
        shutil.copy2(SYNTHETIC_CSV, unicode_dataset)
        spec = {**spec, "dataset_path": str(unicode_dataset)}
        spec_file.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        assigned_run_id = "d" * 64

        result = run_runner(spec_file, tmp_dir, assigned_run_id=assigned_run_id)

        run_dir = tmp_dir / "runs" / assigned_run_id
        eval_result = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
        assert result["status"] == "success"
        assert result["run_id"] == assigned_run_id
        assert Path(result["out_dir"]) == run_dir
        assert eval_result["run_id"] == assigned_run_id
        assert eval_result["spec_hash"] == assigned_run_id
        assert (run_dir / "spec.json").exists()
        assert (run_dir / "predictions.csv").exists()

    def test_test_split_rejected(self, tmp_path: Path) -> None:
        spec = {
            "model": "DLinear",
            "dataset_path": str(SYNTHETIC_CSV),
            "target_col": "OT",
            "time_col": "date",
            "freq": "h",
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "test_ratio": 0.2,
            "seq_len": 24,
            "pred_len": 12,
            "label_len": 12,
            "phase": "forecast",
            "eval_split": "test",
            "seed": 42,
            "hyperparams": {"train_epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        }
        spec_file = tmp_path / "bad_spec.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")

        result = run_runner(spec_file, tmp_path)
        assert result.get("status") == "error"
        assert result.get("returncode") != 0

    @pytest.mark.slow
    def test_eval_json_has_evaluator_hash(self, spec_path: tuple[Path, dict, Path]) -> None:
        spec_file, _spec, tmp_dir = spec_path
        result = run_runner(spec_file, tmp_dir)

        eval_json = json.loads((Path(result["out_dir"]) / "eval.json").read_text(encoding="utf-8"))
        assert len(eval_json["evaluator_hash"]) == 64

    @pytest.mark.slow
    def test_spec_json_in_run_dir_is_canonical(self, spec_path: tuple[Path, dict, Path]) -> None:
        spec_file, spec, tmp_dir = spec_path
        result = run_runner(spec_file, tmp_dir)

        stored_spec = (Path(result["out_dir"]) / "spec.json").read_text(encoding="utf-8")
        assert compute_spec_hash(spec) == compute_spec_hash(json.loads(stored_spec))
