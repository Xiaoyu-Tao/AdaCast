"""Tests for zero-shot foundation model runner behavior."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from adacast_ml import runner


FOUNDATION_MODELS = {
    "Chronos",
    "Chronos2",
    "TimesFM",
    "Moirai",
    "Sundial",
    "TiRex",
    "TimeMoE",
}
FOUNDATION_REVISION = "a" * 40


class FakeFoundationModel(nn.Module):
    """Small model stand-in so tests never import real heavy dependencies."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("write_predictions is patched and should not call forward")


class FakeLazyModelDict(dict):
    """LazyModelDict replacement that avoids filesystem and network imports."""

    def __init__(self, models_dir: str) -> None:
        super().__init__()
        self.models_dir = models_dir

    def __getitem__(self, key: str) -> type[FakeFoundationModel]:
        return FakeFoundationModel


def foundation_spec(tmp_path: Path, model: str = "Chronos") -> tuple[Path, dict[str, Any]]:
    spec = {
        "model": model,
        "dataset_path": str(tmp_path / "series.csv"),
        "target_col": "OT",
        "time_col": "date",
        "freq": "h",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "adjustment_fit_holdout_ratio": 1 / 2,
        "seq_len": 24,
        "label_len": 12,
        "pred_len": 12,
        "eval_split": "val",
        "phase": "forecast",
        "seed": 42,
        "hyperparams": {
            "train_epochs": 99,
            "learning_rate": 0.5,
            "batch_size": 8,
        },
        "metrics": ["mse", "mae", "wape", "mase"],
    }
    spec_path = tmp_path / f"{model.lower()}_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path, spec


def fake_splits() -> dict[str, dict[str, np.ndarray]]:
    timestamps = np.arange(80)
    return {
        "train": {
            "data": np.linspace(0.0, 79.0, 80, dtype=np.float32).reshape(-1, 1),
            "timestamps": timestamps,
        },
        "val": {
            "data": np.linspace(100.0, 179.0, 80, dtype=np.float32).reshape(-1, 1),
            "timestamps": timestamps,
        },
    }


def parse_events(stderr: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stderr.splitlines() if line.strip().startswith("{")]


def install_fake_foundation_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, str | None]] | None = None,
) -> None:
    snapshot_path = tmp_path / "hub" / "snapshots" / FOUNDATION_REVISION
    snapshot_path.mkdir(parents=True)
    (snapshot_path / "model.ckpt").write_bytes(b"fake-tirex-checkpoint")

    def fake_resolve(model_name: str, revision: str | None = None) -> tuple[Path, str]:
        if calls is not None:
            calls.append({"model": model_name, "revision": revision})
        if revision is not None:
            assert revision == FOUNDATION_REVISION
        return snapshot_path, FOUNDATION_REVISION

    monkeypatch.setattr(runner, "resolve_foundation_snapshot", fake_resolve)


def test_zero_shot_models_match_current_foundation_loader_set() -> None:
    assert runner.ZERO_SHOT_MODELS == FOUNDATION_MODELS


def test_zero_shot_branch_skips_training_dataloader_and_hyperparams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_calls: list[dict[str, Any]] = []
    data_loader_calls: list[Any] = []
    write_prediction_calls: list[dict[str, Any]] = []

    def fake_train_model(**kwargs: Any) -> None:
        train_calls.append(kwargs)
        on_epoch = kwargs.get("on_epoch")
        if on_epoch is not None:
            on_epoch(0, kwargs["train_epochs"], 0.123456)

    class SpyDataLoader:
        def __init__(self, dataset: Any, **kwargs: Any) -> None:
            data_loader_calls.append({"dataset": dataset, **kwargs})

    def fake_write_predictions(
        model: nn.Module,
        dataset: runner.ForecastWindowDataset,
        run_dir: Path,
        device: torch.device,
        pred_len: int,
        include_gap_predictions: bool = False,
    ) -> Path:
        write_prediction_calls.append(
            {
                "model": model,
                "dataset": dataset,
                "device": device,
                "pred_len": pred_len,
            }
        )
        predictions_path = run_dir / "predictions.csv"
        predictions_path.write_text(
            "timestamp,actual,predicted\n0,1.0,1.0\n",
            encoding="utf-8",
        )
        np.save(run_dir / "actual.npy", np.asarray([1.0]), allow_pickle=False)
        np.save(run_dir / "pred.npy", np.asarray([1.0]), allow_pickle=False)
        return predictions_path

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        return {"metrics": {"mse": 0.0, "mae": 0.0, "wape": 0.0, "mase": 0.0}}

    monkeypatch.setattr(runner, "LazyModelDict", FakeLazyModelDict)
    monkeypatch.setattr(runner, "train_model", fake_train_model)
    monkeypatch.setattr(runner, "DataLoader", SpyDataLoader)
    monkeypatch.setattr(runner, "write_predictions", fake_write_predictions)
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner, "load_dataset", lambda *args, **kwargs: fake_splits())
    monkeypatch.setattr(runner, "detect_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runner, "get_target_index", lambda spec: 0)
    install_fake_foundation_snapshot(tmp_path, monkeypatch)

    spec_path, _spec = foundation_spec(tmp_path, "Chronos")
    result, exit_code = runner.run_experiment(
        Namespace(
            spec=str(spec_path),
            out_dir=str(tmp_path / "runs"),
            evaluator_hash_store=str(tmp_path / ".forecast" / "evaluator.sha256"),
        )
    )

    events = parse_events(capsys.readouterr().err)
    event_types = [event["type"] for event in events]

    assert exit_code == 0
    assert result["status"] == "success"
    assert train_calls == []
    assert data_loader_calls == []
    assert write_prediction_calls
    assert write_prediction_calls[0]["pred_len"] == 12
    assert write_prediction_calls[0]["model"].config.foundation_model_path.endswith(
        FOUNDATION_REVISION
    )
    assert (
        write_prediction_calls[0]["model"].config.foundation_model_repo
        == runner.FOUNDATION_MODEL_REPOS["Chronos"]
    )
    assert write_prediction_calls[0]["model"].config.foundation_model_revision == FOUNDATION_REVISION
    assert "epoch" not in event_types
    assert {"type": "predict_start", "zero_shot": True, "model": "Chronos"}.items() <= events[-2].items()


def test_tirex_uses_checkpoint_file_inside_pinned_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / FOUNDATION_REVISION
    snapshot.mkdir()
    checkpoint = snapshot / "model.ckpt"
    checkpoint.write_bytes(b"weights")

    assert runner.foundation_model_load_path("TiRex", snapshot) == checkpoint


def test_tirex_rejects_snapshot_without_checkpoint_file(tmp_path: Path) -> None:
    snapshot = tmp_path / FOUNDATION_REVISION
    snapshot.mkdir()

    with pytest.raises(ValueError, match="missing its required model.ckpt"):
        runner.foundation_model_load_path("TiRex", snapshot)


def test_foundation_metadata_rejects_movable_branch_revision(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    repo_id = runner.FOUNDATION_MODEL_REPOS["Chronos"]
    revision = "main"
    (model_dir / "foundation.json").write_text(
        json.dumps(
            {
                "kind": "foundation_model_reference",
                "model": "Chronos",
                "repo_id": repo_id,
                "revision": revision,
                "sha256": runner.foundation_identity_sha256(repo_id, revision),
                "format": "huggingface_snapshot",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not self-consistent"):
        runner.read_source_artifact_identity(tmp_path, "Chronos")


def test_zero_shot_post_forecast_test_records_reuse_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_calls: list[dict[str, Any]] = []
    snapshot_calls: list[dict[str, str | None]] = []

    def fake_write_predictions(
        model: nn.Module,
        dataset: runner.ForecastWindowDataset,
        run_dir: Path,
        device: torch.device,
        pred_len: int,
        include_gap_predictions: bool = False,
    ) -> Path:
        predictions_path = run_dir / "predictions.csv"
        predictions_path.write_text(
            "timestamp,actual,predicted\n0,1.0,1.0\n",
            encoding="utf-8",
        )
        np.save(run_dir / "actual.npy", np.asarray([1.0]), allow_pickle=False)
        np.save(run_dir / "pred.npy", np.asarray([1.0]), allow_pickle=False)
        return predictions_path

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        return {
            "eval_split": kwargs["spec"]["eval_split"],
            "metrics": {"mse": 0.0, "mae": 0.0, "wape": 0.0, "mase": 0.0},
        }

    splits = fake_splits()
    splits["test"] = {
        "data": np.linspace(200.0, 279.0, 80, dtype=np.float32).reshape(-1, 1),
        "timestamps": np.arange(80),
    }
    monkeypatch.setattr(runner, "LazyModelDict", FakeLazyModelDict)
    monkeypatch.setattr(runner, "train_model", lambda **kwargs: train_calls.append(kwargs))
    monkeypatch.setattr(runner, "write_predictions", fake_write_predictions)
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda spec, allowed_splits: {name: splits[name] for name in allowed_splits},
    )
    monkeypatch.setattr(runner, "detect_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runner, "get_target_index", lambda spec: 0)
    install_fake_foundation_snapshot(tmp_path, monkeypatch, snapshot_calls)

    source_spec_path, source_spec = foundation_spec(tmp_path, "Chronos")
    source_result, source_exit_code = runner.run_experiment(
        Namespace(
            spec=str(source_spec_path),
            out_dir=str(tmp_path / "runs"),
            evaluator_hash_store=str(tmp_path / ".forecast" / "evaluator.sha256"),
        )
    )
    assert source_exit_code == 0
    source_identity = json.loads(
        (Path(source_result["out_dir"]) / "model" / "foundation.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_identity["revision"] == FOUNDATION_REVISION

    test_spec_path = tmp_path / "chronos_test_spec.json"
    test_spec_path.write_text(
        json.dumps(
            {
                **source_spec,
                "phase": "post-forecast",
                "eval_split": "test",
            }
        ),
        encoding="utf-8",
    )
    test_result, test_exit_code = runner.run_experiment(
        Namespace(
            spec=str(test_spec_path),
            out_dir=str(tmp_path / "runs"),
            evaluator_hash_store=str(tmp_path / ".forecast" / "evaluator.sha256"),
            reuse_model_from=source_result["out_dir"],
        )
    )

    assert test_exit_code == 0, test_result
    assert train_calls == []
    reuse = json.loads(
        (Path(test_result["out_dir"]) / "model" / "reuse.json").read_text(encoding="utf-8")
    )
    assert reuse["source_run_id"] == source_result["run_id"]
    assert reuse["source_artifact_kind"] == "foundation_model_reference"
    assert reuse["source_artifact_sha256"] == source_identity["sha256"]
    assert reuse["foundation_revision"] == FOUNDATION_REVISION
    assert reuse["training_performed"] is False
    assert reuse["evaluated_split"] == "test"
    assert snapshot_calls == [
        {"model": "Chronos", "revision": None},
        {"model": "Chronos", "revision": FOUNDATION_REVISION},
    ]

    source_identity["revision"] = "d" * 40
    (Path(source_result["out_dir"]) / "model" / "foundation.json").write_text(
        json.dumps(source_identity),
        encoding="utf-8",
    )
    failed_result, failed_exit_code = runner.run_experiment(
        Namespace(
            spec=str(test_spec_path),
            out_dir=str(tmp_path / "runs"),
            evaluator_hash_store=str(tmp_path / ".forecast" / "evaluator.sha256"),
            reuse_model_from=source_result["out_dir"],
        )
    )
    failed_run_dir = Path(failed_result["out_dir"])

    assert failed_exit_code == 1
    assert "Foundation source metadata is not self-consistent" in failed_result["error"]
    assert not (failed_run_dir / "eval.json").exists()
    assert not (failed_run_dir / "predictions.csv").exists()
    assert not (failed_run_dir / "model" / "reuse.json").exists()


def test_optional_foundation_dependency_missing_emits_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class MissingDependencyLazyModelDict(dict):
        def __init__(self, models_dir: str) -> None:
            super().__init__()

        def __getitem__(self, key: str) -> type[FakeFoundationModel]:
            raise ModuleNotFoundError("No module named 'uni2ts'")

    monkeypatch.setattr(runner, "LazyModelDict", MissingDependencyLazyModelDict)
    monkeypatch.setattr(runner, "load_dataset", lambda *args, **kwargs: fake_splits())
    monkeypatch.setattr(runner, "detect_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runner, "get_target_index", lambda spec: 0)

    spec_path, _spec = foundation_spec(tmp_path, "Moirai")
    result, exit_code = runner.run_experiment(
        Namespace(
            spec=str(spec_path),
            out_dir=str(tmp_path / "runs"),
            evaluator_hash_store=str(tmp_path / ".forecast" / "evaluator.sha256"),
        )
    )

    events = parse_events(capsys.readouterr().err)
    dependency_events = [event for event in events if event["type"] == "dependency_missing"]
    train_log = Path(result["out_dir"]) / "train.log"

    assert exit_code == 1
    assert result["status"] == "error"
    assert dependency_events
    assert dependency_events[-1]["module"] == "uni2ts"
    assert dependency_events[-1]["package"] == "uni2ts"
    assert "uv run --project python --extra heavy-models" in dependency_events[-1]["guidance"]
    assert "ModuleNotFoundError" in train_log.read_text(encoding="utf-8")
