"""Unified experiment runner for AdaCast ML experiments."""

from __future__ import annotations

import argparse
import contextlib
import errno
import importlib
import inspect
import json
import math
import os
import re
import shutil
import sys
import time as _time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from adacast_lite_ml.engine.data_loader import (
    load_dataset,
    normalize_adjustment_fit_holdout_ratio,
    numeric_feature_columns,
    reject_if_test_in_forecast,
)
from adacast_lite_ml.engine.env import apply_hf_environment
from adacast_lite_ml.engine.evaluator import evaluate
from adacast_lite_ml.engine.model_cache import inspect_repo_cache
from adacast_lite_ml.engine.models._foundation import FOUNDATION_MODEL_NAMES, FOUNDATION_MODEL_REPOS
from adacast_lite_ml.engine.statistical import (
    STATISTICAL_MODEL_NAMES,
    fit_statistical_model,
    forecast_statistical_result,
    is_statistical_model,
    load_statistical_result,
    normalize_statistical_model_name,
    save_statistical_result,
)
from adacast_lite_ml.engine.utils.device import detect_device, get_device_info
from adacast_lite_ml.engine.utils.hash import canonical_spec_json, compute_file_hash, compute_spec_hash
from adacast_lite_ml.engine.utils.seed import set_seed
from adacast_lite_ml.engine.utils.timefeatures import time_features


class LazyModelDict(dict):
    """Lazy-load models from the copied adacast_lite_ml.engine.models package."""

    def __init__(self, models_dir: str) -> None:
        self._map = {
            path.stem: f"adacast_lite_ml.engine.models.{path.stem}"
            for path in Path(models_dir).iterdir()
            if path.suffix == ".py" and path.name != "__init__.py" and not path.stem.startswith("_")
        }
        self._casefold_map: dict[str, str] = {}
        for model_name in self._map:
            folded = model_name.casefold()
            existing = self._casefold_map.get(folded)
            if existing is not None:
                raise ValueError(f"Ambiguous model names: {existing!r} and {model_name!r}")
            self._casefold_map[folded] = model_name
        super().__init__()

    @property
    def available_model_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._map))

    def resolve_key(self, key: str) -> str:
        return self._casefold_map.get(key.casefold(), key)

    def __getitem__(self, key: str) -> type:
        resolved_key = self.resolve_key(key)
        if resolved_key in self:
            return super().__getitem__(resolved_key)
        if resolved_key not in self._map:
            raise KeyError(f"Model '{key}' not found. Available: {self.available_model_names}")
        module = importlib.import_module(self._map[resolved_key])
        model_class = getattr(module, "Model", None) or getattr(module, resolved_key)
        self[resolved_key] = model_class
        if key != resolved_key:
            self[key] = model_class
        return model_class


class ForecastWindowDataset(Dataset):
    """Sliding-window dataset for long-term forecast models."""

    def __init__(
        self,
        split_payload: dict[str, object],
        seq_len: int,
        label_len: int,
        pred_len: int,
        target_index: int,
        freq: str = "h",
        input_contract_version: int = 1,
        window_stride: int = 1,
        forecast_gap: int = 0,
    ) -> None:
        self.data = np.asarray(split_payload["data"], dtype=np.float32)
        self.timestamps = np.asarray(split_payload["timestamps"])
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_index = target_index
        if (
            isinstance(forecast_gap, bool)
            or not isinstance(forecast_gap, (int, np.integer))
            or forecast_gap < 0
        ):
            raise ValueError("forecast_gap must be a nonnegative integer")
        self.forecast_gap = int(forecast_gap)
        self.model_pred_len = self.forecast_gap + self.pred_len
        if (
            isinstance(window_stride, bool)
            or not isinstance(window_stride, (int, np.integer))
            or window_stride <= 0
        ):
            raise ValueError("window_stride must be a positive integer")
        self.window_stride = int(window_stride)
        if input_contract_version >= 2:
            try:
                datetime_index = pd.DatetimeIndex(pd.to_datetime(self.timestamps))
                feature_frequency = "min" if freq.casefold() == "t" else freq
                self.time_marks = time_features(
                    datetime_index, freq=feature_frequency
                ).transpose().astype(np.float32)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    f"Could not build time features for freq={freq!r}: {exc}"
                ) from exc
        else:
            self.time_marks = np.zeros((len(self.timestamps), 4), dtype=np.float32)

        forecast_origin_index = int(
            split_payload.get(
                "forecast_origin_index",
                split_payload.get("target_start_index", 0),
            )
        )
        self.window_start_index = max(0, forecast_origin_index - self.seq_len)
        available = (
            len(self.data)
            - self.window_start_index
            - self.seq_len
            - self.model_pred_len
            + 1
        )
        if available <= 0:
            raise ValueError(
                "Split is too short for the requested seq_len/forecast_gap/pred_len window sizes"
            )
        self._length = (available - 1) // self.window_stride + 1

    def __len__(self) -> int:
        return self._length

    def window_start(self, index: int) -> int:
        """Return the split-local input-window start for a dataset index."""
        return self.window_start_index + index * self.window_stride

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        s_begin = self.window_start(index)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.model_pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.time_marks[s_begin:s_end]
        seq_y_mark = self.time_marks[r_begin:r_end]

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(seq_x_mark),
            torch.from_numpy(seq_y_mark),
        )

    def prediction_targets(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return timestamps and target values for the prediction horizon."""
        s_end = self.window_start(index) + self.seq_len
        target_start = s_end + self.forecast_gap
        target_end = target_start + self.pred_len
        target_timestamps = self.timestamps[target_start:target_end]
        target_values = self.data[target_start:target_end, self.target_index]
        return target_timestamps, target_values

    def forecast_gap_timestamps(self, index: int) -> np.ndarray:
        """Return timestamps for unevaluated model outputs between history and target."""
        s_end = self.window_start(index) + self.seq_len
        return self.timestamps[s_end : s_end + self.forecast_gap]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the experiment runner."""
    parser = argparse.ArgumentParser(description="AdaCast ML experiment runner")
    parser.add_argument("--spec", required=True, help="Path to an experiment spec JSON")
    parser.add_argument("--out-dir", default="runs", help="Base directory for run artifacts")
    parser.add_argument(
        "--run-id",
        help="Optional orchestrator-assigned run id. When provided, it is the artifact directory identity.",
    )
    parser.add_argument(
        "--evaluator-hash-store",
        default=".forecast/evaluator.sha256",
        help="Path to the persisted evaluator hash file",
    )
    parser.add_argument(
        "--reuse-model-from",
        help="Existing validation run directory whose fitted model must be reused without training",
    )
    return parser.parse_args(argv)


def normalize_execution_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate the phase/split combinations supported by the runner."""
    normalized = dict(spec)
    phase = str(normalized.get("phase", "forecast"))
    eval_split = str(normalized.get("eval_split", "val"))
    if not (
        (phase == "forecast" and eval_split in {"val", "adjustment_fit"})
        or (phase == "post-forecast" and eval_split == "test")
    ):
        raise ValueError(
            "Invalid experiment execution contract: expected "
            "phase='forecast' with eval_split='val' or 'adjustment_fit', "
            "or phase='post-forecast' with eval_split='test'"
        )
    if "window_stride" in normalized:
        window_stride = normalized["window_stride"]
        if (
            isinstance(window_stride, bool)
            or not isinstance(window_stride, (int, np.integer))
            or window_stride <= 0
        ):
            raise ValueError("window_stride must be a positive integer")
        normalized["window_stride"] = int(window_stride)
    if "forecast_gap" in normalized:
        forecast_gap = normalized["forecast_gap"]
        if (
            isinstance(forecast_gap, bool)
            or not isinstance(forecast_gap, (int, np.integer))
            or forecast_gap < 0
        ):
            raise ValueError("forecast_gap must be a nonnegative integer")
        normalized["forecast_gap"] = int(forecast_gap)
    normalized["phase"] = phase
    normalized["eval_split"] = eval_split
    return normalized


def load_spec(spec_path: Path, assigned_run_id: str | None = None) -> tuple[dict[str, Any], str, str]:
    """Load the experiment spec and its canonical JSON representation."""
    spec = normalize_execution_contract(
        normalize_adjustment_fit_holdout_ratio(
            json.loads(spec_path.read_text(encoding="utf-8"))
        )
    )
    canonical = canonical_spec_json(spec)
    if assigned_run_id is not None and re.fullmatch(r"[A-Fa-f0-9]{64}", assigned_run_id) is None:
        raise ValueError("--run-id must be a 64-character hexadecimal spec hash")
    run_id = assigned_run_id or compute_spec_hash(spec)
    return spec, canonical, run_id


def resolve_project_root(out_dir: Path) -> Path:
    """Infer the project root for .forecast/runs output, else use cwd."""
    resolved_out_dir = out_dir.resolve()
    if resolved_out_dir.name == "runs" and resolved_out_dir.parent.name == ".forecast":
        return resolved_out_dir.parent.parent
    return Path.cwd()


@contextlib.contextmanager
def exclusive_run_lock(out_dir: Path, run_id: str):
    """Serialize writers for one deterministic run id across processes."""
    lock_dir = out_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{run_id}.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EDEADLK}:
                        raise
                    _time.sleep(0.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


PREDICTION_ARTIFACT_FILES = {
    "predictions_sha256": "predictions.csv",
    "actual_sha256": "actual.npy",
    "pred_sha256": "pred.npy",
    "train_scale_sha256": "train_scale.json",
}


def compute_prediction_artifact_hashes(run_dir: Path) -> dict[str, str]:
    """Hash every output that the fixed evaluator result depends on."""
    return {
        key: compute_file_hash(run_dir / filename)
        for key, filename in PREDICTION_ARTIFACT_FILES.items()
    }


def build_eval_artifact_binding(
    *,
    model_artifact: dict[str, str],
    model_source_run_id: str,
    prediction_artifacts: dict[str, str],
) -> dict[str, object]:
    """Build the immutable identity linking metrics, predictions, and model."""
    return {
        "model_artifact": {
            "kind": model_artifact["kind"],
            "sha256": model_artifact["sha256"],
            "source_run_id": model_source_run_id,
            "foundation_revision": model_artifact.get("revision"),
        },
        "prediction_artifacts": prediction_artifacts,
    }


def eval_artifact_binding_matches(
    *,
    eval_result: dict[str, Any],
    run_dir: Path,
    model_artifact: dict[str, str],
    model_source_run_id: str,
) -> bool:
    """Verify that stored metrics still describe the current model and outputs."""
    try:
        current_prediction_hashes = compute_prediction_artifact_hashes(run_dir)
    except OSError:
        return False
    expected = build_eval_artifact_binding(
        model_artifact=model_artifact,
        model_source_run_id=model_source_run_id,
        prediction_artifacts=current_prediction_hashes,
    )
    return eval_result.get("artifact_binding") == expected


def prediction_layout_matches_contract(
    run_dir: Path,
    spec: dict[str, Any],
) -> bool:
    """Validate explicit row coordinates required by current and ARIMA outputs."""
    requires_explicit_layout = (
        int(spec.get("input_contract_version", 1)) >= 3
        or is_statistical_model(str(spec.get("model", "")))
    )
    if not requires_explicit_layout:
        return True

    try:
        predictions = pd.read_csv(run_dir / "predictions.csv")
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    required = {"prediction_layout", "window_index", "horizon_index"}
    if not required.issubset(predictions.columns):
        return False

    window_values = pd.to_numeric(predictions["window_index"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    horizon_values = pd.to_numeric(
        predictions["horizon_index"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(window_values).all()
        or not np.isfinite(horizon_values).all()
        or (window_values < 0).any()
        or (horizon_values < 0).any()
        or not np.equal(window_values, np.floor(window_values)).all()
        or not np.equal(horizon_values, np.floor(horizon_values)).all()
    ):
        return False

    layouts = predictions["prediction_layout"].astype(str)
    if is_statistical_model(str(spec.get("model", ""))):
        legacy_continuous = (
            int(spec.get("input_contract_version", 1)) < 4
            and
            layouts.eq("continuous_origin").all()
            and np.equal(window_values, 0).all()
            and np.array_equal(
                horizon_values.astype(np.int64),
                np.arange(len(predictions), dtype=np.int64),
            )
        )
        if legacy_continuous:
            return True

    if not layouts.eq("sliding_window").all():
        return False
    pred_len = int(spec.get("pred_len", 0))
    if pred_len <= 0 or len(predictions) == 0 or len(predictions) % pred_len != 0:
        return False
    expected_windows = np.repeat(
        np.arange(len(predictions) // pred_len, dtype=np.int64),
        pred_len,
    )
    expected_horizons = np.tile(
        np.arange(pred_len, dtype=np.int64),
        len(predictions) // pred_len,
    )
    return np.array_equal(
        window_values.astype(np.int64),
        expected_windows,
    ) and np.array_equal(
        horizon_values.astype(np.int64),
        expected_horizons,
    )


def completed_run_artifacts_exist(
    run_dir: Path,
    run_id: str,
    spec: dict[str, Any],
    reuse_source_dir: Path | None = None,
) -> bool:
    """Return true only when cached metrics and prediction artifacts are all present."""
    required = (
        "spec.json",
        "eval.json",
        "predictions.csv",
        "actual.npy",
        "pred.npy",
        "train_scale.json",
    )
    if not all((run_dir / name).exists() for name in required):
        return False
    if reuse_source_dir is not None and not (run_dir / "model" / "reuse.json").exists():
        return False
    try:
        stored_spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        cached_eval = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
        reuse_provenance = (
            json.loads((run_dir / "model" / "reuse.json").read_text(encoding="utf-8"))
            if reuse_source_dir is not None
            else None
        )
        source_identity = (
            read_source_artifact_identity(
                reuse_source_dir,
                str(spec.get("model", "")),
            )
            if reuse_source_dir is not None
            else read_source_artifact_identity(run_dir, str(spec.get("model", "")))
        )
    except (OSError, json.JSONDecodeError):
        return False
    except ValueError:
        return False
    return (
        canonical_spec_json(stored_spec) == canonical_spec_json(spec)
        and cached_eval.get("run_id") == run_id
        and cached_eval.get("spec_hash") == run_id
        and prediction_layout_matches_contract(run_dir, spec)
        and eval_artifact_binding_matches(
            eval_result=cached_eval,
            run_dir=run_dir,
            model_artifact=source_identity,
            model_source_run_id=(
                reuse_source_dir.name if reuse_source_dir is not None else run_id
            ),
        )
        and (
            reuse_source_dir is None
            or (
                reuse_provenance is not None
                and reuse_provenance.get("source_run_id") == reuse_source_dir.name
                and reuse_provenance.get("training_performed") is False
                and reuse_provenance.get("evaluated_split") == spec.get("eval_split", "val")
                and reuse_provenance.get("source_artifact_kind")
                == expected_reuse_artifact_kind(str(spec.get("model", "")))
                and reuse_provenance.get("source_artifact_sha256")
                == source_identity["sha256"]
            )
        )
    )


def clear_incomplete_run_artifacts(run_dir: Path) -> None:
    """Remove stale outputs before rebuilding an invalid or incomplete run."""
    for name in (
        "spec.json",
        "eval.json",
        "predictions.csv",
        "actual.npy",
        "pred.npy",
        "train_scale.json",
        "train.log",
    ):
        path = run_dir / name
        if path.exists():
            path.unlink()
    model_dir = run_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)


def get_target_index(spec: dict[str, Any]) -> int:
    """Resolve the target column index within the non-time feature matrix."""
    df = pd.read_csv(spec["dataset_path"], low_memory=False)
    feature_cols = numeric_feature_columns(df, str(spec["time_col"]), str(spec["target_col"]))
    return feature_cols.index(str(spec["target_col"]))


# Foundation models that use zero-shot inference and require task_name='zero_shot_forecast'
# in their forward() method. All other models use the standard 'long_term_forecast' path.
ZERO_SHOT_MODELS: frozenset[str] = FOUNDATION_MODEL_NAMES
FOUNDATION_MODEL_NAME_BY_CASEFOLD = {name.casefold(): name for name in FOUNDATION_MODEL_NAMES}
SPEC_CONTROL_KEYS = frozenset(
    {
        "model",
        "dataset_path",
        "target_col",
        "time_col",
        "freq",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "split_mode",
        "train_end_timestamp",
        "validation_start_timestamp",
        "prediction_target_start_timestamp",
        "exogenous_columns",
        "exogenous_history_steps",
        "validation_end_timestamp",
        "adjustment_fit_holdout_ratio",
        "seq_len",
        "label_len",
        "pred_len",
        "forecast_gap",
        "window_stride",
        "phase",
        "eval_split",
        "seed",
        "metrics",
        "window_features",
        "hyperparams",
        "input_contract_version",
    }
)


def canonical_foundation_model_name(model_name: str) -> str:
    """Return the canonical foundation model spelling when model_name differs only by case."""
    return FOUNDATION_MODEL_NAME_BY_CASEFOLD.get(model_name.casefold(), model_name)


def is_zero_shot_model(model_name: str) -> bool:
    return canonical_foundation_model_name(model_name) in ZERO_SHOT_MODELS


def is_arima_model(model_name: str) -> bool:
    return normalize_statistical_model_name(model_name) == "ARIMA"


def expected_reuse_artifact_kind(model_name: str) -> str:
    if is_arima_model(model_name):
        return "statsmodels_arima_results"
    if is_statistical_model(model_name):
        return "statsmodels_statistical_results"
    if is_zero_shot_model(canonical_foundation_model_name(model_name)):
        return "foundation_model_reference"
    return "torch_state_dict"


def foundation_identity_sha256(repo_id: str, revision: str) -> str:
    """Hash the immutable repository revision used by a foundation model."""
    return compute_spec_hash({"repo_id": repo_id, "revision": revision})


def resolve_foundation_snapshot(
    model_name: str,
    revision: str | None = None,
) -> tuple[Path, str]:
    """Resolve a foundation repository to an immutable local snapshot."""
    from huggingface_hub import snapshot_download

    repo_id = FOUNDATION_MODEL_REPOS[model_name]
    snapshot_path = Path(snapshot_download(repo_id=repo_id, revision=revision)).resolve()
    resolved_revision = snapshot_path.name
    if not re.fullmatch(r"[A-Fa-f0-9]{40,64}", resolved_revision):
        raise ValueError(
            f"Could not resolve an immutable commit revision for foundation model {model_name}"
        )
    if revision is not None and resolved_revision.casefold() != revision.casefold():
        raise ValueError(
            f"Foundation snapshot revision mismatch: expected {revision}, got {resolved_revision}"
        )
    return snapshot_path, resolved_revision


def foundation_model_load_path(model_name: str, snapshot_path: Path) -> Path:
    """Return the library-specific local path used to load a pinned snapshot."""
    if model_name == "TiRex":
        checkpoint_path = snapshot_path / "model.ckpt"
        if not checkpoint_path.is_file():
            raise ValueError(
                f"TiRex snapshot is missing its required model.ckpt: {checkpoint_path}"
            )
        return checkpoint_path
    return snapshot_path


def write_foundation_artifact(
    *,
    run_dir: Path,
    model_name: str,
    revision: str,
) -> dict[str, str]:
    """Persist the immutable repository identity used for zero-shot inference."""
    repo_id = FOUNDATION_MODEL_REPOS[model_name]
    identity = {
        "kind": "foundation_model_reference",
        "model": model_name,
        "repo_id": repo_id,
        "revision": revision,
        "sha256": foundation_identity_sha256(repo_id, revision),
        "format": "huggingface_snapshot",
    }
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "foundation.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return identity


def read_source_artifact_identity(
    run_dir: Path,
    model_name: str,
) -> dict[str, str]:
    """Read and verify the fitted artifact identity stored by a validation run."""
    canonical_model_name = canonical_foundation_model_name(model_name)
    if is_zero_shot_model(canonical_model_name):
        path = run_dir / "model" / "foundation.json"
        if not path.exists():
            raise ValueError(f"Foundation source metadata is missing: {path}")
        identity = json.loads(path.read_text(encoding="utf-8"))
        repo_id = FOUNDATION_MODEL_REPOS[canonical_model_name]
        revision = identity.get("revision")
        expected_sha256 = (
            foundation_identity_sha256(repo_id, revision)
            if isinstance(revision, str)
            and re.fullmatch(r"[A-Fa-f0-9]{40,64}", revision)
            else None
        )
        if (
            identity.get("kind") != "foundation_model_reference"
            or identity.get("model") != canonical_model_name
            or identity.get("repo_id") != repo_id
            or identity.get("format") != "huggingface_snapshot"
            or identity.get("sha256") != expected_sha256
        ):
            raise ValueError("Foundation source metadata is not self-consistent")
        return {
            "kind": "foundation_model_reference",
            "sha256": str(expected_sha256),
            "repo_id": repo_id,
            "revision": revision,
        }

    metadata_path = run_dir / "model" / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Reusable model metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if is_arima_model(model_name):
        artifact_path = run_dir / "model" / "arima-results.pkl"
        expected_kind = "arima_results"
        expected_format = "statsmodels_arima_results"
        reuse_kind = "statsmodels_arima_results"
    elif is_statistical_model(model_name):
        artifact_path = run_dir / "model" / "statistical-results.pkl"
        expected_kind = "statistical_results"
        expected_format = "statsmodels_results_pickle"
        reuse_kind = "statsmodels_statistical_results"
    else:
        artifact_path = run_dir / "model" / "checkpoint.pth"
        expected_kind = "checkpoint"
        expected_format = "torch_state_dict"
        reuse_kind = "torch_state_dict"
    if not artifact_path.exists():
        raise ValueError(f"Reusable model artifact is missing: {artifact_path}")
    actual_sha256 = compute_file_hash(artifact_path)
    metadata_model_matches = (
        normalize_statistical_model_name(str(metadata.get("model", "")))
        == normalize_statistical_model_name(model_name)
        if is_statistical_model(model_name)
        else str(metadata.get("model", "")).casefold() == model_name.casefold()
    )
    if (
        metadata.get("kind") != expected_kind
        or metadata.get("format") != expected_format
        or not metadata_model_matches
        or metadata.get("eval_split") != "val"
        or metadata.get("phase") != "forecast"
        or metadata.get("sha256") != actual_sha256
    ):
        raise ValueError("Reusable model artifact does not match its validation metadata")
    return {
        "kind": reuse_kind,
        "sha256": actual_sha256,
        "path": str(artifact_path),
    }


def extract_hyperparams(spec: dict[str, Any]) -> dict[str, Any]:
    """Accept both flattened model params and a nested hyperparams object."""
    hyperparams = {key: value for key, value in spec.items() if key not in SPEC_CONTROL_KEYS}
    nested = spec.get("hyperparams", {})
    if nested is None:
        return hyperparams
    if not isinstance(nested, dict):
        raise ValueError("spec.hyperparams must be an object when provided")
    hyperparams.update(nested)
    return hyperparams


def instantiate_model(model_class: type, config: SimpleNamespace, hyperparams: dict[str, Any]) -> nn.Module:
    """Instantiate a model, forwarding supported constructor hyperparameters."""
    signature = inspect.signature(model_class)
    kwargs: dict[str, Any] = {}
    for index, (name, parameter) in enumerate(signature.parameters.items()):
        if index == 0:
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if name in hyperparams:
            kwargs[name] = hyperparams[name]
    return model_class(config, **kwargs)

DEPENDENCY_PACKAGE_BY_MODULE = {
    "chronos": "chronos-forecasting",
    "timesfm": "timesfm",
    "uni2ts": "uni2ts",
    "transformers": "transformers",
    "tirex": "tirex-ts[gluonts]",
}

HEAVY_MODEL_DEPENDENCY_GUIDANCE = (
    "Install heavy model dependencies with: uv run --project python --extra heavy-models "
    "python -m adacast_lite_ml.engine.runner ..."
)


def _missing_module_name(exc: ModuleNotFoundError) -> str:
    """Resolve a useful missing module name from ModuleNotFoundError variants."""
    if exc.name:
        return exc.name
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(exc))
    if match:
        return match.group(1)
    return str(exc)


def _recommended_dependency_package(module_name: str) -> str:
    root_module = module_name.split(".", maxsplit=1)[0]
    return DEPENDENCY_PACKAGE_BY_MODULE.get(root_module, root_module)


def config_flag_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "off"}
    return bool(value)


def build_model_config(spec: dict[str, Any], num_features: int) -> SimpleNamespace:
    """Build a minimal argparse-like config object for REF models."""
    hyperparams = extract_hyperparams(spec)
    model_pred_len = int(spec["pred_len"]) + int(spec.get("forecast_gap", 0))
    default_task_name = (
        "zero_shot_forecast"
        if is_zero_shot_model(str(spec.get("model", "")))
        else "long_term_forecast"
    )
    config_values: dict[str, Any] = {
        "task_name": default_task_name,
        "seq_len": spec["seq_len"],
        "label_len": spec.get("label_len", spec["seq_len"] // 2),
        "pred_len": model_pred_len,
        "enc_in": num_features,
        "dec_in": num_features,
        "c_out": hyperparams.get("c_out", num_features if num_features > 1 else 1),
        "d_model": hyperparams.get("d_model", 512),
        "n_heads": hyperparams.get("n_heads", 8),
        "e_layers": hyperparams.get("e_layers", 2),
        "d_layers": hyperparams.get("d_layers", 1),
        "d_ff": hyperparams.get("d_ff", 2048),
        "dropout": hyperparams.get("dropout", 0.1),
        "embed": hyperparams.get("embed", "timeF"),
        "freq": spec.get("freq", "h"),
        "top_k": hyperparams.get("top_k", 5),
        "num_kernels": hyperparams.get("num_kernels", 6),
        "moving_avg": hyperparams.get("moving_avg", 25),
        "individual": hyperparams.get("individual", False),
        "factor": hyperparams.get("factor", 3),
        "activation": hyperparams.get("activation", "gelu"),
        "num_class": hyperparams.get("num_class", 1),
        "output_attention": hyperparams.get("output_attention", False),
        "channel_independence": hyperparams.get("channel_independence", 1),
        "decomp_method": hyperparams.get("decomp_method", "moving_avg"),
        "use_norm": hyperparams.get("use_norm", 1),
        "down_sampling_method": hyperparams.get("down_sampling_method", "avg"),
        "down_sampling_layers": hyperparams.get("down_sampling_layers", 1),
        "down_sampling_window": hyperparams.get("down_sampling_window", 2),
    }
    config_values.update(hyperparams)

    model_name = str(spec.get("model", "")).casefold()
    if model_name == "segrnn":
        seg_len = int(hyperparams.get("seg_len", math.gcd(int(spec["seq_len"]), model_pred_len)))
        if seg_len <= 0 or int(spec["seq_len"]) % seg_len != 0 or model_pred_len % seg_len != 0:
            raise ValueError("SegRNN seg_len must be positive and divide both seq_len and pred_len")
        config_values["seg_len"] = seg_len

    if model_name == "timemixer" and config_flag_enabled(
        config_values["channel_independence"]
    ):
        # Channel-independent TimeMixer reshapes B*N outputs back to c_out channels.
        # The output channel count must therefore match the multivariate input width.
        config_values["c_out"] = num_features

    return SimpleNamespace(**config_values)


def build_decoder_input(seq_y: torch.Tensor, pred_len: int) -> torch.Tensor:
    """Construct the decoder input expected by REF forecast models."""
    label_len = seq_y.shape[1] - pred_len
    zeros = torch.zeros(
        seq_y.shape[0],
        pred_len,
        seq_y.shape[2],
        dtype=seq_y.dtype,
        device=seq_y.device,
    )
    return torch.cat([seq_y[:, :label_len, :], zeros], dim=1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    pred_len: int,
    train_epochs: int,
    learning_rate: float,
    on_epoch: Callable[[int, int, float], None] | None = None,
) -> None:
    """Run a minimal supervised training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(train_epochs):
        epoch_losses: list[float] = []
        for seq_x, seq_y, seq_x_mark, seq_y_mark in train_loader:
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            seq_x_mark = seq_x_mark.to(device)
            seq_y_mark = seq_y_mark.to(device)

            decoder_input = build_decoder_input(seq_y, pred_len)
            target = seq_y[:, -pred_len:, :]

            optimizer.zero_grad(set_to_none=True)
            output = model(seq_x, seq_x_mark, decoder_input, seq_y_mark)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        if on_epoch is not None:
            on_epoch(epoch, train_epochs, epoch_loss)
        print(f"epoch={epoch + 1} loss={epoch_loss:.6f}")


def write_checkpoint_artifacts(
    *,
    model: nn.Module,
    run_dir: Path,
    model_name: str,
    eval_split: str,
    phase: str,
) -> Path:
    """Persist reusable trained state for locally trained torch models."""
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)  # checkpoint.pth
    checkpoint_sha256 = compute_file_hash(checkpoint_path)

    metadata = {
        "kind": "checkpoint",
        "path": "model/checkpoint.pth",
        "sha256": checkpoint_sha256,
        "model": model_name,
        "eval_split": eval_split,
        "phase": phase,
        "trained_on": "train",
        "format": "torch_state_dict",
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint_path


def validate_reuse_contract(
    spec: dict[str, Any],
    source_spec: dict[str, Any],
) -> None:
    """Ensure inference-only replay preserves the fitted validation model contract."""
    source = normalize_execution_contract(
        normalize_adjustment_fit_holdout_ratio(source_spec)
    )
    if source.get("eval_split", "val") != "val":
        raise ValueError("Reusable model source must be a validation run")
    if source.get("phase", "forecast") != "forecast":
        raise ValueError("Reusable model source must come from the forecast phase")

    eval_split = str(spec.get("eval_split", "val"))
    expected = dict(source)
    if eval_split == "adjustment_fit":
        expected["eval_split"] = "adjustment_fit"
        contract_description = "except for eval_split='adjustment_fit'"
    elif eval_split == "test" and spec.get("phase") == "post-forecast":
        expected["phase"] = "post-forecast"
        expected["eval_split"] = "test"
        contract_description = "except for phase='post-forecast' and eval_split='test'"
    else:
        raise ValueError(
            "--reuse-model-from is only valid for adjustment_fit replay or "
            "post-forecast test inference"
        )

    normalized_spec = normalize_execution_contract(
        normalize_adjustment_fit_holdout_ratio(spec)
    )
    expected.setdefault("window_stride", 1)
    normalized_spec.setdefault("window_stride", 1)
    expected.setdefault("forecast_gap", 0)
    normalized_spec.setdefault("forecast_gap", 0)
    if canonical_spec_json(expected) != canonical_spec_json(normalized_spec):
        raise ValueError(
            "Inference replay spec must exactly match its source validation spec "
            f"{contract_description}"
        )


def write_reuse_provenance(
    *,
    run_dir: Path,
    source_run_dir: Path,
    model_name: str,
    artifact_kind: str,
    artifact_sha256: str,
    evaluated_split: str,
    foundation_revision: str | None = None,
) -> Path:
    """Record that a fitted validation model was reused without training."""
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "reuse.json"
    path.write_text(
        json.dumps(
            {
                "kind": "model_reuse",
                "model": model_name,
                "source_run_id": source_run_dir.name,
                "source_run_path": str(source_run_dir.resolve()),
                "source_artifact_kind": artifact_kind,
                "source_artifact_sha256": artifact_sha256,
                "foundation_revision": foundation_revision,
                "training_performed": False,
                "evaluated_split": evaluated_split,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def write_predictions(
    model: nn.Module,
    dataset: ForecastWindowDataset,
    run_dir: Path,
    device: torch.device,
    pred_len: int,
    include_gap_predictions: bool = False,
) -> Path:
    """Generate prediction rows for the evaluation split and write predictions.csv."""
    rows: list[dict[str, float | str]] = []
    gap_rows: list[dict[str, float | str | int]] = []
    actual_values: list[float] = []
    predicted_values: list[float] = []
    model.eval()

    with torch.no_grad():
        for index in range(len(dataset)):
            seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[index]
            seq_x = seq_x.unsqueeze(0).to(device)
            seq_y = seq_y.unsqueeze(0).to(device)
            seq_x_mark = seq_x_mark.unsqueeze(0).to(device)
            seq_y_mark = seq_y_mark.unsqueeze(0).to(device)

            decoder_input = build_decoder_input(seq_y, pred_len)
            output = model(seq_x, seq_x_mark, decoder_input, seq_y_mark)
            all_predictions = output[0, :, dataset.target_index].detach().cpu().numpy()
            prediction_end = dataset.forecast_gap + dataset.pred_len
            if len(all_predictions) < prediction_end:
                raise ValueError("Model output is shorter than forecast_gap + pred_len")
            predictions = all_predictions[dataset.forecast_gap:prediction_end]

            if include_gap_predictions and dataset.forecast_gap > 0:
                gap_timestamps = dataset.forecast_gap_timestamps(index)
                gap_predictions = all_predictions[: dataset.forecast_gap]
                for gap_index, (timestamp, predicted_value) in enumerate(
                    zip(gap_timestamps, gap_predictions, strict=True)
                ):
                    gap_rows.append(
                        {
                            "timestamp": str(timestamp),
                            "predicted": float(predicted_value),
                            "prediction_layout": "forecast_gap",
                            "window_index": index,
                            "gap_index": gap_index,
                        }
                    )

            timestamps, actual = dataset.prediction_targets(index)
            for horizon_index, (timestamp, actual_value, predicted_value) in enumerate(
                zip(timestamps, actual, predictions, strict=True)
            ):
                actual_values.append(float(actual_value))
                predicted_values.append(float(predicted_value))
                rows.append(
                    {
                        "timestamp": str(timestamp),
                        "actual": float(actual_value),
                        "predicted": float(predicted_value),
                        "prediction_layout": "sliding_window",
                        "window_index": index,
                        "horizon_index": horizon_index,
                    }
                )

    predictions_path = run_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    np.save(run_dir / "actual.npy", np.asarray(actual_values, dtype=np.float64), allow_pickle=False)
    if include_gap_predictions and dataset.forecast_gap > 0:
        pd.DataFrame(gap_rows).to_csv(run_dir / "gap_predictions.csv", index=False)
    np.save(run_dir / "pred.npy", np.asarray(predicted_values, dtype=np.float64), allow_pickle=False)
    return predictions_path


def write_statistical_artifacts(
    fitted: Any,
    run_dir: Path,
    model_name: str,
    eval_split: str,
    phase: str,
) -> Path:
    """Persist a statsmodels fitted result for inference-only replay."""
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    is_arima = is_arima_model(model_name)
    result_name = "arima-results.pkl" if is_arima else "statistical-results.pkl"
    result_path = model_dir / result_name
    save_statistical_result(fitted, result_path)
    result_sha256 = compute_file_hash(result_path)
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "kind": "arima_results" if is_arima else "statistical_results",
                "path": f"model/{result_name}",
                "sha256": result_sha256,
                "model": model_name,
                "eval_split": eval_split,
                "phase": phase,
                "trained_on": "train",
                "format": "statsmodels_arima_results"
                if is_arima
                else "statsmodels_results_pickle",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result_path


def write_statistical_predictions(
    fitted: Any,
    eval_payload: dict[str, object],
    run_dir: Path,
    target_index: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    window_stride: int,
    forecast_gap: int = 0,
    include_gap_predictions: bool = False,
) -> Path:
    """Forecast the same strided evaluation windows used by sequence models."""
    target_start_index = int(
        eval_payload.get(
            "target_start_index",
            int(eval_payload.get("forecast_origin_index", 0)) + forecast_gap,
        )
    )
    eval_series = np.asarray(eval_payload["data"])[
        target_start_index:, target_index
    ].astype(np.float64)
    forecast_offset = int(eval_payload.get("forecast_offset", 0))
    all_predictions = np.asarray(
        forecast_statistical_result(
            fitted,
            steps=forecast_offset + len(eval_series),
        ),
        dtype=np.float64,
    )
    window_dataset = ForecastWindowDataset(
        eval_payload,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        target_index=target_index,
        input_contract_version=1,
        window_stride=window_stride,
        forecast_gap=forecast_gap,
    )
    rows: list[dict[str, float | str | int]] = []
    gap_rows: list[dict[str, float | str | int]] = []
    actual_values: list[float] = []
    predicted_values: list[float] = []
    for window_index in range(len(window_dataset)):
        prediction_start = (
            window_dataset.window_start(window_index)
            + seq_len
            + window_dataset.forecast_gap
            - target_start_index
        )
        forecast_start = forecast_offset + prediction_start
        forecast_end = forecast_start + pred_len
        predictions = all_predictions[forecast_start:forecast_end]
        timestamps, actual = window_dataset.prediction_targets(window_index)
        if len(predictions) != pred_len:
            raise ValueError(
                "Statistical forecast did not cover the requested evaluation window"
            )
        if include_gap_predictions and forecast_gap > 0:
            gap_start = forecast_start - forecast_gap
            gap_values = all_predictions[gap_start:forecast_start]
            gap_timestamps = window_dataset.forecast_gap_timestamps(window_index)
            if len(gap_values) != forecast_gap:
                raise ValueError(
                    "Statistical forecast did not cover the requested forecast gap"
                )
            for gap_index, (timestamp, predicted_value) in enumerate(
                zip(gap_timestamps, gap_values, strict=True)
            ):
                gap_rows.append(
                    {
                        "timestamp": str(timestamp),
                        "predicted": float(predicted_value),
                        "prediction_layout": "forecast_gap",
                        "window_index": window_index,
                        "gap_index": gap_index,
                    }
                )
        for horizon_index, (timestamp, actual_value, predicted_value) in enumerate(
            zip(timestamps, actual, predictions, strict=True)
        ):
            actual_values.append(float(actual_value))
            predicted_values.append(float(predicted_value))
            rows.append(
                {
                    "timestamp": str(timestamp),
                    "actual": float(actual_value),
                    "predicted": float(predicted_value),
                    "prediction_layout": "sliding_window",
                    "window_index": window_index,
                    "horizon_index": horizon_index,
                }
            )

    predictions_path = run_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    if include_gap_predictions and forecast_gap > 0:
        pd.DataFrame(gap_rows).to_csv(run_dir / "gap_predictions.csv", index=False)
    np.save(
        run_dir / "actual.npy",
        np.asarray(actual_values, dtype=np.float64),
        allow_pickle=False,
    )
    np.save(
        run_dir / "pred.npy",
        np.asarray(predicted_values, dtype=np.float64),
        allow_pickle=False,
    )
    return predictions_path


def run_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Execute one deterministic run while preventing concurrent writers."""
    spec_path = Path(args.spec)
    spec, canonical_spec, run_id = load_spec(spec_path, getattr(args, "run_id", None))
    out_dir = Path(args.out_dir)
    run_dir = out_dir / run_id
    reuse_source_dir = (
        Path(args.reuse_model_from).resolve()
        if getattr(args, "reuse_model_from", None)
        else None
    )
    with exclusive_run_lock(out_dir, run_id):
        if reuse_source_dir is not None and reuse_source_dir != run_dir.resolve():
            with exclusive_run_lock(reuse_source_dir.parent, reuse_source_dir.name):
                return _run_experiment_locked(
                    args=args,
                    spec=spec,
                    canonical_spec=canonical_spec,
                    run_id=run_id,
                    out_dir=out_dir,
                    run_dir=run_dir,
                    reuse_source_dir=reuse_source_dir,
                )
        return _run_experiment_locked(
            args=args,
            spec=spec,
            canonical_spec=canonical_spec,
            run_id=run_id,
            out_dir=out_dir,
            run_dir=run_dir,
            reuse_source_dir=reuse_source_dir,
        )


def _run_experiment_locked(
    *,
    args: argparse.Namespace,
    spec: dict[str, Any],
    canonical_spec: str,
    run_id: str,
    out_dir: Path,
    run_dir: Path,
    reuse_source_dir: Path | None,
) -> tuple[dict[str, Any], int]:
    """Execute the run after its cross-process writer lock is held."""
    evaluator_hash_store = Path(args.evaluator_hash_store)
    cached_eval_path = run_dir / "eval.json"
    if completed_run_artifacts_exist(
        run_dir,
        run_id,
        spec,
        reuse_source_dir=reuse_source_dir,
    ):
        cached_eval = json.loads(cached_eval_path.read_text(encoding="utf-8"))
        return (
            {
                "status": "cached",
                "run_id": run_id,
                "out_dir": str(run_dir),
                "metrics": cached_eval.get("metrics"),
            },
            0,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    clear_incomplete_run_artifacts(run_dir)
    (run_dir / "spec.json").write_text(canonical_spec, encoding="utf-8")
    train_log_path = run_dir / "train.log"
    project_root = resolve_project_root(out_dir)
    hf_environment = apply_hf_environment(project_root=project_root)
    _orig_stderr = sys.stderr
    _run_start = _time.monotonic()
    train_log_writer: Any | None = None

    def _emit_event(event_type: str, **kwargs: object) -> None:
        """Emit a JSON event line for TUI progress and post-failure inspection."""
        event = {
            "type": event_type,
            "elapsed_s": round(_time.monotonic() - _run_start, 2),
            **kwargs,
        }
        event_line = json.dumps(event)
        print(event_line, file=_orig_stderr, flush=True)
        if train_log_writer is not None:
            print(event_line, file=train_log_writer, flush=True)

    def _emit_cache_inspection(model_name: str) -> None:
        repo_id = FOUNDATION_MODEL_REPOS.get(model_name)
        if repo_id is None:
            return
        cache_event = inspect_repo_cache(repo_id, hf_environment.hub_cache)
        cache_event_type = cache_event.get("type", "cache_warning")
        _emit_event(
            cache_event_type,
            **{key: value for key, value in cache_event.items() if key != "type"},
        )

    def _emit_dependency_missing(exc: ModuleNotFoundError) -> None:
        module_name = _missing_module_name(exc)
        _emit_event(
            "dependency_missing",
            module=module_name,
            package=_recommended_dependency_package(module_name),
            guidance=HEAVY_MODEL_DEPENDENCY_GUIDANCE,
        )

    with train_log_path.open("w", encoding="utf-8") as train_log:
        train_log_writer = train_log
        try:
            with contextlib.redirect_stdout(train_log), contextlib.redirect_stderr(train_log):
                _emit_event(
                    "cache_config",
                    cache_root=hf_environment.cache_root,
                    hub_cache=hf_environment.hub_cache,
                    xet_cache=hf_environment.xet_cache,
                    assets_cache=hf_environment.assets_cache,
                    transformers_cache=hf_environment.transformers_cache,
                    endpoint=hf_environment.endpoint,
                    source=hf_environment.source,
                    warnings=hf_environment.warnings,
                )
                for warning in hf_environment.warnings:
                    _emit_event(
                        "cache_warning",
                        code=warning["code"],
                        message=warning["message"],
                    )
                reject_if_test_in_forecast(spec)
                eval_split = str(spec.get("eval_split", "val"))
                source_spec: dict[str, Any] | None = None
                source_artifact_identity: dict[str, str] | None = None
                evaluated_artifact_identity: dict[str, str] | None = None
                if reuse_source_dir is not None:
                    source_spec_path = reuse_source_dir / "spec.json"
                    source_eval_path = reuse_source_dir / "eval.json"
                    if not source_spec_path.exists():
                        raise ValueError(
                            f"Reusable source run is missing spec.json: {source_spec_path}"
                        )
                    if not source_eval_path.exists():
                        raise ValueError(
                            f"Reusable source run is missing eval.json: {source_eval_path}"
                        )
                    source_spec = json.loads(source_spec_path.read_text(encoding="utf-8"))
                    source_eval = json.loads(source_eval_path.read_text(encoding="utf-8"))
                    source_spec_hash = compute_spec_hash(
                        normalize_adjustment_fit_holdout_ratio(source_spec)
                    )
                    if (
                        source_eval.get("run_id") != reuse_source_dir.name
                        or source_eval.get("eval_split") != "val"
                        or source_eval.get("spec_hash")
                        not in {reuse_source_dir.name, source_spec_hash}
                    ):
                        raise ValueError(
                            "Reusable source run is not a self-consistent validation result"
                        )
                    validate_reuse_contract(spec, source_spec)
                    source_artifact_identity = read_source_artifact_identity(
                        reuse_source_dir,
                        str(source_spec.get("model", "")),
                    )
                    if not eval_artifact_binding_matches(
                        eval_result=source_eval,
                        run_dir=reuse_source_dir,
                        model_artifact=source_artifact_identity,
                        model_source_run_id=reuse_source_dir.name,
                    ):
                        raise ValueError(
                            "Reusable source validation metrics are not bound to "
                            "the current model and prediction artifacts"
                        )
                    evaluated_artifact_identity = source_artifact_identity
                elif eval_split == "adjustment_fit":
                    raise ValueError(
                        "eval_split='adjustment_fit' requires --reuse-model-from; "
                        "training on the reserved validation-front evidence split is forbidden"
                    )
                elif eval_split == "test" and spec.get("phase") == "post-forecast":
                    raise ValueError(
                        "Post-forecast test inference requires --reuse-model-from; "
                        "retraining after validation model selection is forbidden"
                    )

                set_seed(spec.get("seed", 42))
                device = detect_device()
                print(f"device_info={json.dumps(get_device_info(), sort_keys=True)}")

                if reuse_source_dir is not None and eval_split == "adjustment_fit":
                    allowed_splits = ["train", "adjustment_fit"]
                elif reuse_source_dir is not None and eval_split == "test":
                    allowed_splits = ["train", "test"]
                elif spec.get("phase") == "post-forecast":
                    allowed_splits = ["train", "val", "test"]
                else:
                    allowed_splits = ["train", "val"]

                dataset_splits = load_dataset(spec, allowed_splits=allowed_splits)
                if eval_split not in dataset_splits:
                    raise ValueError(f"Requested eval split {eval_split!r} is not available")

                target_index_value = dataset_splits["train"].get("target_index")
                target_index = (
                    int(target_index_value)
                    if target_index_value is not None
                    else get_target_index(spec)
                )
                raw_model_name = str(spec["model"])
                if is_statistical_model(raw_model_name):
                    hyperparams = extract_hyperparams(spec)
                    canonical_statistical_name = normalize_statistical_model_name(
                        raw_model_name
                    )
                    if canonical_statistical_name is None:
                        raise ValueError(
                            f"Unknown statistical model {raw_model_name!r}; "
                            f"available: {', '.join(STATISTICAL_MODEL_NAMES)}"
                        )
                    print(f"statistical_model={canonical_statistical_name}")
                    if reuse_source_dir is not None:
                        result_name = (
                            "arima-results.pkl"
                            if is_arima_model(raw_model_name)
                            else "statistical-results.pkl"
                        )
                        result_path = reuse_source_dir / "model" / result_name
                        if source_artifact_identity is None:
                            raise ValueError(
                                "Reusable statistical source identity is unavailable"
                            )
                        fitted = load_statistical_result(result_path)
                        if (
                            compute_file_hash(result_path)
                            != source_artifact_identity["sha256"]
                        ):
                            raise ValueError(
                                "Reusable statistical results changed while being loaded"
                            )
                        write_reuse_provenance(
                            run_dir=run_dir,
                            source_run_dir=reuse_source_dir,
                            model_name=raw_model_name,
                            artifact_kind=source_artifact_identity["kind"],
                            artifact_sha256=source_artifact_identity["sha256"],
                            evaluated_split=eval_split,
                        )
                    else:
                        _emit_event("train_start")
                        train_series = np.asarray(dataset_splits["train"]["data"])[
                            :, target_index
                        ].astype(np.float64)
                        canonical_statistical_name, fitted = fit_statistical_model(
                            model_name=raw_model_name,
                            train_series=train_series,
                            hyperparams=hyperparams,
                        )
                        write_statistical_artifacts(
                            fitted=fitted,
                            run_dir=run_dir,
                            model_name=canonical_statistical_name,
                            eval_split=eval_split,
                            phase=str(spec.get("phase", "")),
                        )
                        evaluated_artifact_identity = read_source_artifact_identity(
                            run_dir,
                            raw_model_name,
                        )
                    _emit_event("predict_start")
                    predictions_path = write_statistical_predictions(
                        fitted=fitted,
                        eval_payload=dataset_splits[eval_split],
                        run_dir=run_dir,
                        target_index=target_index,
                        seq_len=int(spec["seq_len"]),
                        label_len=int(spec.get("label_len", int(spec["seq_len"]) // 2)),
                        pred_len=int(spec["pred_len"]),
                        window_stride=spec.get("window_stride", 1),
                        forecast_gap=spec.get("forecast_gap", 0),
                        include_gap_predictions=eval_split == "test",
                    )
                else:
                    num_features = np.asarray(dataset_splits["train"]["data"]).shape[1]
                    model_registry = LazyModelDict(str(Path(__file__).resolve().parent / "models"))
                    resolve_key = getattr(model_registry, "resolve_key", None)
                    resolved_model_name = (
                        resolve_key(raw_model_name) if callable(resolve_key) else raw_model_name
                    )
                    model_name = canonical_foundation_model_name(resolved_model_name)
                    hyperparams = extract_hyperparams(spec)
                    config = build_model_config({**spec, "model": model_name}, num_features)
                    is_zero_shot = is_zero_shot_model(model_name)
                    if is_zero_shot:
                        _emit_cache_inspection(model_name)

                    try:
                        model_class = model_registry[model_name]
                        if is_zero_shot:
                            pinned_revision = (
                                source_artifact_identity.get("revision")
                                if source_artifact_identity is not None
                                else None
                            )
                            foundation_snapshot, resolved_revision = resolve_foundation_snapshot(
                                model_name,
                                pinned_revision,
                            )
                            config.foundation_model_repo = FOUNDATION_MODEL_REPOS[model_name]
                            config.foundation_model_revision = resolved_revision
                            config.foundation_model_path = str(
                                foundation_model_load_path(model_name, foundation_snapshot)
                            )
                        model = instantiate_model(model_class, config, hyperparams).to(device)
                    except ModuleNotFoundError as exc:
                        _emit_dependency_missing(exc)
                        raise

                    eval_dataset = ForecastWindowDataset(
                        dataset_splits[eval_split],
                        seq_len=config.seq_len,
                        label_len=config.label_len,
                        pred_len=int(spec["pred_len"]),
                        target_index=target_index,
                        freq=config.freq,
                        input_contract_version=int(spec.get("input_contract_version", 1)),
                        window_stride=spec.get("window_stride", 1),
                        forecast_gap=spec.get("forecast_gap", 0),
                    )

                    if is_zero_shot:
                        if reuse_source_dir is None:
                            write_foundation_artifact(
                                run_dir=run_dir,
                                model_name=model_name,
                                revision=config.foundation_model_revision,
                            )
                            source_artifact_identity = read_source_artifact_identity(
                                run_dir,
                                model_name,
                            )
                            evaluated_artifact_identity = source_artifact_identity
                        if reuse_source_dir is not None:
                            if source_artifact_identity is None:
                                raise ValueError("Foundation source identity is unavailable")
                            write_reuse_provenance(
                                run_dir=run_dir,
                                source_run_dir=reuse_source_dir,
                                model_name=model_name,
                                artifact_kind="foundation_model_reference",
                                artifact_sha256=source_artifact_identity["sha256"],
                                evaluated_split=eval_split,
                                foundation_revision=source_artifact_identity["revision"],
                            )
                        _emit_event("predict_start", model=model_name, zero_shot=True)
                    elif reuse_source_dir is not None:
                        checkpoint_path = reuse_source_dir / "model" / "checkpoint.pth"
                        if source_artifact_identity is None:
                            raise ValueError("Reusable checkpoint source identity is unavailable")
                        state_dict = torch.load(
                            checkpoint_path,
                            map_location=device,
                            weights_only=True,
                        )
                        if compute_file_hash(checkpoint_path) != source_artifact_identity["sha256"]:
                            raise ValueError("Reusable checkpoint changed while being loaded")
                        model.load_state_dict(state_dict)
                        write_reuse_provenance(
                            run_dir=run_dir,
                            source_run_dir=reuse_source_dir,
                            model_name=model_name,
                            artifact_kind="torch_state_dict",
                            artifact_sha256=source_artifact_identity["sha256"],
                            evaluated_split=eval_split,
                        )
                        _emit_event("predict_start", model=model_name, reused_model=True)
                    else:
                        _emit_event("train_start")
                        train_dataset = ForecastWindowDataset(
                            dataset_splits["train"],
                            seq_len=config.seq_len,
                            label_len=config.label_len,
                            pred_len=int(spec["pred_len"]),
                            target_index=target_index,
                            freq=config.freq,
                            input_contract_version=int(spec.get("input_contract_version", 1)),
                            window_stride=spec.get("window_stride", 1),
                            forecast_gap=spec.get("forecast_gap", 0),
                        )
                        batch_size = int(hyperparams.get("batch_size", 32))
                        learning_rate = float(hyperparams.get("learning_rate", 0.0001))
                        train_epochs = int(hyperparams.get("train_epochs", 5))
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                        )
                        train_model(
                            model=model,
                            train_loader=train_loader,
                            device=device,
                            pred_len=config.pred_len,
                            train_epochs=train_epochs,
                            learning_rate=learning_rate,
                            on_epoch=lambda epoch, total_epochs, loss: _emit_event("epoch", epoch=epoch + 1, total_epochs=total_epochs, loss=round(loss, 6)),
                        )
                        write_checkpoint_artifacts(
                            model=model,
                            run_dir=run_dir,
                            model_name=model_name,
                            eval_split=eval_split,
                            phase=str(spec.get("phase", "")),
                        )
                        evaluated_artifact_identity = read_source_artifact_identity(
                            run_dir,
                            model_name,
                        )
                        _emit_event("predict_start")

                    predictions_path = write_predictions(
                        model=model,
                        dataset=eval_dataset,
                        run_dir=run_dir,
                        device=device,
                        pred_len=config.pred_len,
                        include_gap_predictions=eval_split == "test",
                    )

                train_target = np.asarray(dataset_splits["train"]["data"])[
                    :, target_index
                ].astype(np.float64)
                train_scale = float(np.mean(np.abs(np.diff(train_target)))) if len(train_target) > 1 else 0.0
                (run_dir / "train_scale.json").write_text(
                    json.dumps({"train_scale": train_scale}, indent=2),
                    encoding="utf-8",
                )

                if evaluated_artifact_identity is None:
                    raise ValueError("Evaluated model artifact identity is unavailable")
                prediction_artifact_hashes = compute_prediction_artifact_hashes(run_dir)
                eval_result = evaluate(
                    run_dir=run_dir,
                    spec=spec,
                    evaluator_hash_store=evaluator_hash_store,
                    predictions_path=predictions_path,
                    train_values=train_target,
                )
                if compute_prediction_artifact_hashes(run_dir) != prediction_artifact_hashes:
                    raise ValueError("Prediction artifacts changed during evaluation")
                artifact_source_dir = reuse_source_dir or run_dir
                current_artifact_identity = read_source_artifact_identity(
                    artifact_source_dir,
                    str(spec.get("model", "")),
                )
                if current_artifact_identity != evaluated_artifact_identity:
                    raise ValueError("Model artifact changed during prediction or evaluation")
                eval_result = {
                    **eval_result,
                    "run_id": run_id,
                    "spec_hash": run_id,
                    "artifact_binding": build_eval_artifact_binding(
                        model_artifact=evaluated_artifact_identity,
                        model_source_run_id=artifact_source_dir.name,
                        prediction_artifacts=prediction_artifact_hashes,
                    ),
                }
                (run_dir / "eval.json").write_text(
                    json.dumps(eval_result, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                _emit_event("done")

            return (
                {
                    "status": "success",
                    "run_id": run_id,
                    "out_dir": str(run_dir),
                    "metrics": eval_result["metrics"],
                },
                0,
            )
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            traceback.print_exc(file=train_log)
            return (
                {
                    "status": "error",
                    "run_id": run_id,
                    "out_dir": str(run_dir),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
                1,
            )
        except BaseException as exc:  # noqa: BLE001
            traceback.print_exc(file=train_log)
            return (
                {
                    "status": "error",
                    "run_id": run_id,
                    "out_dir": str(run_dir),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
                1,
            )
        finally:
            train_log_writer = None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    result, exit_code = run_experiment(parse_args(argv))
    print(json.dumps(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
