"""Tests for Hugging Face cache_config and cache_warning environment resolution."""

from __future__ import annotations

import os
from pathlib import Path

from adacast_ml.env import apply_hf_environment, resolve_hf_environment


STANDARD_ENV_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_ASSETS_CACHE",
    "TRANSFORMERS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "PYTORCH_TRANSFORMERS_CACHE",
    "PYTORCH_PRETRAINED_BERT_CACHE",
)


def clear_standard_env(monkeypatch) -> None:
    for name in (*STANDARD_ENV_NAMES, "HF_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)


def test_resolve_hf_environment_cache_config_preserves_operator_endpoint_and_cache(
    tmp_path: Path,
) -> None:
    resolved = resolve_hf_environment(
        project_root=tmp_path,
        environ={
            "HF_ENDPOINT": "https://hf-mirror.example",
            "HF_HUB_CACHE": "/operator/hub",
        },
    )

    assert resolved.endpoint == "https://hf-mirror.example"
    assert resolved.source == "operator"
    assert resolved.hub_cache == "/operator/hub"
    assert resolved.env["HF_ENDPOINT"] == "https://hf-mirror.example"
    assert resolved.env["HF_HUB_CACHE"] == "/operator/hub"
    assert resolved.warnings == []


def test_resolve_hf_environment_cache_config_uses_project_model_cache_defaults(
    tmp_path: Path,
) -> None:
    resolved = resolve_hf_environment(project_root=tmp_path, environ={})
    cache_root = str(resolved.cache_root).replace("\\", "/")
    hub_cache = str(resolved.hub_cache).replace("\\", "/")
    xet_cache = str(resolved.xet_cache).replace("\\", "/")
    assets_cache = str(resolved.assets_cache).replace("\\", "/")

    assert resolved.source == "project_default"
    assert ".forecast/model-cache" in cache_root
    assert ".forecast/model-cache/hub" in hub_cache
    assert ".forecast/model-cache/xet" in xet_cache
    assert ".forecast/model-cache/assets" in assets_cache
    assert resolved.transformers_cache is None
    assert resolved.env == {
        "HF_HUB_CACHE": str(tmp_path / ".forecast" / "model-cache" / "hub"),
        "HF_XET_CACHE": str(tmp_path / ".forecast" / "model-cache" / "xet"),
        "HF_ASSETS_CACHE": str(tmp_path / ".forecast" / "model-cache" / "assets"),
    }
    assert "HF_HOME" not in resolved.env


def test_resolve_hf_environment_cache_config_derives_hub_from_transformers_cache(
    tmp_path: Path,
) -> None:
    resolved = resolve_hf_environment(
        project_root=tmp_path,
        environ={
            "TRANSFORMERS_CACHE": "/legacy/transformers",
        },
    )

    assert resolved.source == "operator"
    assert resolved.transformers_cache == "/legacy/transformers"
    assert resolved.hub_cache == "/legacy/transformers"
    assert resolved.env["TRANSFORMERS_CACHE"] == "/legacy/transformers"
    assert resolved.env["HF_HUB_CACHE"] == "/legacy/transformers"


def test_apply_hf_environment_cache_config_mutates_os_environ(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clear_standard_env(monkeypatch)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")

    resolved = apply_hf_environment(project_root=tmp_path)

    assert resolved.source == "project_default"
    assert resolved.endpoint == "https://hf-mirror.example"
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.example"
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / ".forecast" / "model-cache" / "hub")
    assert os.environ["HF_XET_CACHE"] == str(tmp_path / ".forecast" / "model-cache" / "xet")
    assert os.environ["HF_ASSETS_CACHE"] == str(tmp_path / ".forecast" / "model-cache" / "assets")


def test_resolve_hf_environment_cache_warning_when_project_cache_unwritable(
    tmp_path: Path,
) -> None:
    (tmp_path / ".forecast").write_text("not a directory", encoding="utf-8")

    resolved = resolve_hf_environment(project_root=tmp_path, environ={})

    assert resolved.source == "hf_default"
    assert resolved.cache_root is None
    assert "HF_HUB_CACHE" not in resolved.env
    assert resolved.warnings == [
        {
            "code": "project_cache_unwritable",
            "message": "Project model cache .forecast/model-cache is not writable; using Hugging Face defaults.",
        }
    ]
