from __future__ import annotations

import tomllib
from pathlib import Path


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_pytorch_sources_enable_cuda_on_windows_and_linux() -> None:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    uv_config = config["tool"]["uv"]
    sources = uv_config["sources"]

    for package in ("torch", "torchvision"):
        cuda_source = next(
            source for source in sources[package] if source["index"] == "pytorch-cu126"
        )
        cpu_source = next(
            source for source in sources[package] if source["index"] == "pytorch-cpu"
        )

        assert "sys_platform == 'linux'" in cuda_source["marker"]
        assert "sys_platform == 'win32'" in cuda_source["marker"]
        assert cpu_source["marker"] == "sys_platform == 'darwin'"

    assert set(uv_config["environments"]) == {
        "sys_platform == 'linux'",
        "sys_platform == 'win32'",
        "sys_platform == 'darwin'",
    }
