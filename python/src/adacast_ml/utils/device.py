"""Runtime device detection helpers."""

from __future__ import annotations

import torch


def detect_device() -> torch.device:
    """Detect the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict[str, object]:
    """Return lightweight device metadata for logs and artifacts."""
    device = detect_device()
    info: dict[str, object] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        )
    return info
