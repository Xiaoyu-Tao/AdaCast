"""Hugging Face environment resolution for AdaCast model execution."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


STANDARD_HF_CACHE_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_ASSETS_CACHE",
    "TRANSFORMERS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "PYTORCH_TRANSFORMERS_CACHE",
    "PYTORCH_PRETRAINED_BERT_CACHE",
)

PROJECT_CACHE_WARNING = {
    "code": "project_cache_unwritable",
    "message": "Project model cache .forecast/model-cache is not writable; using Hugging Face defaults.",
}


@dataclass(frozen=True)
class HfEnvironment:
    """Resolved Hugging Face cache and mirror environment values."""

    env: dict[str, str]
    cache_root: str | None
    hub_cache: str | None
    xet_cache: str | None
    assets_cache: str | None
    transformers_cache: str | None
    endpoint: str | None
    source: Literal["operator", "project_default", "hf_default"]
    warnings: list[dict[str, str]]


def _present(environ: Mapping[str, str], key: str) -> str | None:
    value = environ.get(key)
    return value if value else None


def _has_operator_cache_env(environ: Mapping[str, str]) -> bool:
    return any(_present(environ, key) is not None for key in STANDARD_HF_CACHE_ENV_KEYS)


def _cache_root_writable(cache_root: Path) -> bool:
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        probe_path = cache_root / f".adacast-probe-{os.getpid()}"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _operator_environment(
    environ: Mapping[str, str],
    endpoint: str | None,
) -> HfEnvironment:
    resolved_env: dict[str, str] = {}
    if endpoint:
        resolved_env["HF_ENDPOINT"] = endpoint

    for key in STANDARD_HF_CACHE_ENV_KEYS:
        value = _present(environ, key)
        if value is not None:
            resolved_env[key] = value

    transformers_cache = _present(environ, "TRANSFORMERS_CACHE")
    if transformers_cache and not _present(environ, "HF_HUB_CACHE"):
        resolved_env["HF_HUB_CACHE"] = transformers_cache

    return HfEnvironment(
        env=resolved_env,
        cache_root=_present(environ, "HF_HOME"),
        hub_cache=resolved_env.get("HF_HUB_CACHE")
        or _present(environ, "HUGGINGFACE_HUB_CACHE"),
        xet_cache=resolved_env.get("HF_XET_CACHE"),
        assets_cache=resolved_env.get("HF_ASSETS_CACHE"),
        transformers_cache=transformers_cache,
        endpoint=endpoint,
        source="operator",
        warnings=[],
    )


def resolve_hf_environment(
    project_root: Path,
    environ: Mapping[str, str] | None = None,
) -> HfEnvironment:
    """Resolve standard Hugging Face env overrides for a project."""
    source_environ = os.environ if environ is None else environ
    endpoint = _present(source_environ, "HF_ENDPOINT")

    if _has_operator_cache_env(source_environ):
        return _operator_environment(source_environ, endpoint)

    resolved_env: dict[str, str] = {}
    if endpoint:
        resolved_env["HF_ENDPOINT"] = endpoint

    cache_root = project_root / ".forecast" / "model-cache"
    if not _cache_root_writable(cache_root):
        return HfEnvironment(
            env=resolved_env,
            cache_root=None,
            hub_cache=None,
            xet_cache=None,
            assets_cache=None,
            transformers_cache=None,
            endpoint=endpoint,
            source="hf_default",
            warnings=[dict(PROJECT_CACHE_WARNING)],
        )

    hub_cache = cache_root / "hub"
    xet_cache = cache_root / "xet"
    assets_cache = cache_root / "assets"
    resolved_env.update(
        {
            "HF_HUB_CACHE": str(hub_cache),
            "HF_XET_CACHE": str(xet_cache),
            "HF_ASSETS_CACHE": str(assets_cache),
        }
    )

    return HfEnvironment(
        env=resolved_env,
        cache_root=str(cache_root),
        hub_cache=str(hub_cache),
        xet_cache=str(xet_cache),
        assets_cache=str(assets_cache),
        transformers_cache=None,
        endpoint=endpoint,
        source="project_default",
        warnings=[],
    )


def apply_hf_environment(
    project_root: Path,
    environ: MutableMapping[str, str] | None = None,
) -> HfEnvironment:
    """Apply resolved Hugging Face env overrides to a mutable environment."""
    target_environ = os.environ if environ is None else environ
    resolved = resolve_hf_environment(project_root=project_root, environ=target_environ)
    target_environ.update(resolved.env)
    return resolved


__all__ = [
    "HfEnvironment",
    "STANDARD_HF_CACHE_ENV_KEYS",
    "apply_hf_environment",
    "resolve_hf_environment",
]
