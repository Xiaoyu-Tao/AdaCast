"""Tests for cache diagnostic events without real network downloads."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from adacast_ml import model_cache, runner
from adacast_ml.env import STANDARD_HF_CACHE_ENV_KEYS
from adacast_ml.utils.hash import compute_spec_hash


REPO_ID = "amazon/chronos-bolt-base"
FOUNDATION_REVISION = "d" * 40


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


def clear_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*STANDARD_HF_CACHE_ENV_KEYS, "HF_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)


def foundation_spec(tmp_path: Path, model: str = "Chronos") -> Path:
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
        "metrics": ["mse", "mae", "wape", "mase"],
    }
    spec_path = tmp_path / f"{model.lower()}_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


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


def parse_events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]


def normalized_path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def patch_successful_foundation_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "hub" / "snapshots" / FOUNDATION_REVISION
    snapshot_path.mkdir(parents=True)
    (snapshot_path / "model.ckpt").write_bytes(b"fake-tirex-checkpoint")

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

    monkeypatch.setattr(runner, "LazyModelDict", FakeLazyModelDict)
    monkeypatch.setattr(runner, "write_predictions", fake_write_predictions)
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda **kwargs: {"metrics": {"mse": 0.0, "mae": 0.0, "wape": 0.0, "mase": 0.0}},
    )
    monkeypatch.setattr(runner, "load_dataset", lambda *args, **kwargs: fake_splits())
    monkeypatch.setattr(runner, "detect_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runner, "get_target_index", lambda spec: 0)
    monkeypatch.setattr(
        runner,
        "resolve_foundation_snapshot",
        lambda model_name, revision=None: (snapshot_path, FOUNDATION_REVISION),
    )


def run_mocked_foundation(tmp_path: Path, model: str = "Chronos") -> tuple[dict[str, Any], int]:
    spec_path = foundation_spec(tmp_path, model)
    return runner.run_experiment(
        Namespace(
            spec=str(spec_path),
            out_dir=str(tmp_path / ".forecast" / "runs"),
            evaluator_hash_store=str(tmp_path / ".forecast" / "evaluator.sha256"),
        )
    )


def test_runner_load_spec_uses_assigned_run_id_and_preserves_unicode(tmp_path: Path) -> None:
    spec = {
        "model": "SegRNN",
        "dataset_path": r"D:\桌面大文件\学习资料\光伏数据.csv",
        "context": {"description": "光伏功率预测"},
    }
    spec_path = tmp_path / "unicode-spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    assigned_run_id = "b" * 64

    loaded, canonical, run_id = runner.load_spec(spec_path, assigned_run_id)

    assert loaded == {
        **spec,
        "adjustment_fit_holdout_ratio": 1 / 2,
        "phase": "forecast",
        "eval_split": "val",
    }
    assert run_id == assigned_run_id
    assert "桌面大文件" in canonical
    assert "\\u684c" not in canonical


def test_runner_writes_artifacts_under_orchestrator_assigned_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_successful_foundation_run(tmp_path, monkeypatch)
    spec_path = foundation_spec(tmp_path, "Chronos")
    assigned_run_id = "c" * 64

    result, exit_code = runner.run_experiment(
        Namespace(
            spec=str(spec_path),
            run_id=assigned_run_id,
            out_dir=str(tmp_path / ".forecast" / "runs"),
            evaluator_hash_store=str(tmp_path / ".forecast" / "evaluator.sha256"),
        )
    )

    run_dir = tmp_path / ".forecast" / "runs" / assigned_run_id
    eval_result = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["run_id"] == assigned_run_id
    assert Path(result["out_dir"]) == run_dir
    assert eval_result["run_id"] == assigned_run_id
    assert eval_result["spec_hash"] == assigned_run_id


def test_runner_does_not_cache_eval_without_prediction_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_successful_foundation_run(tmp_path, monkeypatch)
    spec_path = foundation_spec(tmp_path, "Chronos")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    run_id = compute_spec_hash(spec)
    run_dir = tmp_path / ".forecast" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "eval.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "evaluator_hash": "eval-hash",
                "spec_hash": run_id,
                "metrics": {"mse": 9.0, "mae": 3.0, "wape": 1.0, "mase": None},
                "eval_split": "val",
                "timestamp": "2026-06-03T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result, exit_code = runner.run_experiment(
        Namespace(
            spec=str(spec_path),
            out_dir=str(tmp_path / ".forecast" / "runs"),
            evaluator_hash_store=str(tmp_path / ".forecast" / "evaluator.sha256"),
        )
    )

    assert exit_code == 0
    assert result["status"] == "success"
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "actual.npy").exists()
    assert (run_dir / "pred.npy").exists()


def test_cache_rejects_replaced_checkpoint_even_when_metadata_is_updated(
    tmp_path: Path,
) -> None:
    spec = {
        "model": "DLinear",
        "eval_split": "val",
        "phase": "forecast",
    }
    run_id = "a" * 64
    run_dir = tmp_path / "runs" / run_id
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(
        runner.canonical_spec_json(spec),
        encoding="utf-8",
    )
    prediction_contents = {
        "predictions.csv": b"timestamp,actual,predicted\n0,1,1\n",
        "actual.npy": b"actual",
        "pred.npy": b"pred",
        "train_scale.json": b'{"train_scale": 1}',
    }
    for filename, content in prediction_contents.items():
        (run_dir / filename).write_bytes(content)

    checkpoint = model_dir / "checkpoint.pth"
    checkpoint.write_bytes(b"original-checkpoint")
    original_sha256 = runner.compute_file_hash(checkpoint)
    metadata = {
        "kind": "checkpoint",
        "path": "model/checkpoint.pth",
        "sha256": original_sha256,
        "model": "DLinear",
        "eval_split": "val",
        "phase": "forecast",
        "trained_on": "train",
        "format": "torch_state_dict",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    binding = runner.build_eval_artifact_binding(
        model_artifact={
            "kind": "torch_state_dict",
            "sha256": original_sha256,
        },
        model_source_run_id=run_id,
        prediction_artifacts=runner.compute_prediction_artifact_hashes(run_dir),
    )
    (run_dir / "eval.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "evaluator_hash": "e" * 64,
                "spec_hash": run_id,
                "metrics": {"mse": 0.0, "mae": 0.0, "wape": 0.0, "mase": None},
                "eval_split": "val",
                "timestamp": "2026-06-10T00:00:00+00:00",
                "artifact_binding": binding,
            }
        ),
        encoding="utf-8",
    )

    assert runner.completed_run_artifacts_exist(run_dir, run_id, spec)

    checkpoint.write_bytes(b"replacement-checkpoint")
    metadata["sha256"] = runner.compute_file_hash(checkpoint)
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert not runner.completed_run_artifacts_exist(run_dir, run_id, spec)


def test_exclusive_run_lock_blocks_a_second_process(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    run_id = "b" * 64
    ready = tmp_path / "child-ready"
    acquired = tmp_path / "child-acquired"
    child_code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from adacast_ml.runner import exclusive_run_lock\n"
        "out_dir, run_id, ready, acquired = sys.argv[1:]\n"
        "Path(ready).write_text('ready', encoding='utf-8')\n"
        "with exclusive_run_lock(Path(out_dir), run_id):\n"
        "    Path(acquired).write_text('acquired', encoding='utf-8')\n"
    )

    with runner.exclusive_run_lock(out_dir, run_id):
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                child_code,
                str(out_dir),
                run_id,
                str(ready),
                str(acquired),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        time.sleep(0.3)
        assert not acquired.exists()

    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert acquired.exists()


def test_inspect_repo_cache_reports_miss_then_hit_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []
    cache_states = [
        SimpleNamespace(repos=[]),
        SimpleNamespace(repos=[SimpleNamespace(repo_id=REPO_ID)]),
    ]

    def fake_scan_cache_dir(**kwargs: str) -> SimpleNamespace:
        calls.append(kwargs)
        return cache_states.pop(0)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(scan_cache_dir=fake_scan_cache_dir),
    )

    first = model_cache.inspect_repo_cache(REPO_ID, "/tmp/hf-cache/hub")
    second = model_cache.inspect_repo_cache(REPO_ID, "/tmp/hf-cache/hub")

    assert first == {"type": "cache_miss", "repo_id": REPO_ID}
    assert second == {"type": "cache_hit", "repo_id": REPO_ID}
    assert calls == [{"cache_dir": "/tmp/hf-cache/hub"}, {"cache_dir": "/tmp/hf-cache/hub"}]


def test_runner_emits_cache_config_and_miss_to_stderr_and_train_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
    patch_successful_foundation_run(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "inspect_repo_cache", lambda repo_id, hub_cache: {"type": "cache_miss", "repo_id": repo_id})

    result, exit_code = run_mocked_foundation(tmp_path, "Chronos")

    stderr_events = parse_events(capsys.readouterr().err)
    train_log_events = parse_events((Path(result["out_dir"]) / "train.log").read_text(encoding="utf-8"))

    cache_config = next(event for event in stderr_events if event["type"] == "cache_config")
    assert exit_code == 0
    assert result["status"] == "success"
    assert cache_config["endpoint"] == "https://hf-mirror.example"
    assert cache_config["source"] == "project_default"
    assert ".forecast/model-cache" in normalized_path_text(cache_config["cache_root"])
    assert ".forecast/model-cache/hub" in normalized_path_text(cache_config["hub_cache"])
    assert ".forecast/model-cache/xet" in normalized_path_text(cache_config["xet_cache"])
    assert ".forecast/model-cache/assets" in normalized_path_text(cache_config["assets_cache"])
    assert cache_config["transformers_cache"] is None
    assert cache_config["warnings"] == []
    assert any(event["type"] == "cache_miss" and event["repo_id"] == REPO_ID for event in stderr_events)
    assert {"cache_config", "cache_miss"} <= {event["type"] for event in train_log_events}
    assert {"HF_ENDPOINT", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "TRANSFORMERS_CACHE"}


def test_unwritable_project_cache_warns_and_uses_hf_default_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".forecast").write_text("not a directory", encoding="utf-8")
    clear_hf_env(monkeypatch)
    patch_successful_foundation_run(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "resolve_project_root", lambda out_dir: project_root)
    monkeypatch.setattr(runner, "inspect_repo_cache", lambda repo_id, hub_cache: {"type": "cache_hit", "repo_id": repo_id})

    result, exit_code = run_mocked_foundation(tmp_path, "Chronos")

    stderr_events = parse_events(capsys.readouterr().err)
    train_log_events = parse_events((Path(result["out_dir"]) / "train.log").read_text(encoding="utf-8"))
    cache_config = next(event for event in stderr_events if event["type"] == "cache_config")
    warnings = [event for event in stderr_events if event["type"] == "cache_warning"]

    assert exit_code == 0
    assert cache_config["source"] == "hf_default"
    assert cache_config["cache_root"] is None
    assert cache_config["warnings"][0]["code"] == "project_cache_unwritable"
    assert ".forecast/model-cache" in cache_config["warnings"][0]["message"]
    assert warnings[-1]["code"] == "project_cache_unwritable"
    assert any(event["type"] == "cache_warning" for event in train_log_events)


def test_dependency_missing_event_includes_guidance_and_train_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class MissingDependencyLazyModelDict(dict):
        def __init__(self, models_dir: str) -> None:
            super().__init__()

        def __getitem__(self, key: str) -> type[FakeFoundationModel]:
            raise ModuleNotFoundError("No module named 'uni2ts'")

    clear_hf_env(monkeypatch)
    monkeypatch.setattr(runner, "LazyModelDict", MissingDependencyLazyModelDict)
    monkeypatch.setattr(runner, "load_dataset", lambda *args, **kwargs: fake_splits())
    monkeypatch.setattr(runner, "detect_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(runner, "get_target_index", lambda spec: 0)
    monkeypatch.setattr(runner, "inspect_repo_cache", lambda repo_id, hub_cache: {"type": "cache_hit", "repo_id": repo_id})

    result, exit_code = run_mocked_foundation(tmp_path, "Moirai")

    stderr_events = parse_events(capsys.readouterr().err)
    train_log_events = parse_events((Path(result["out_dir"]) / "train.log").read_text(encoding="utf-8"))
    dependency_events = [event for event in stderr_events if event["type"] == "dependency_missing"]

    assert exit_code == 1
    assert result["status"] == "error"
    assert dependency_events[-1]["module"] == "uni2ts"
    assert dependency_events[-1]["package"] == "uni2ts"
    assert "uv run --project python --extra heavy-models" in dependency_events[-1]["guidance"]
    assert any(event["type"] == "dependency_missing" for event in train_log_events)
