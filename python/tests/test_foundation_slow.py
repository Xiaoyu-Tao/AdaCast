"""Default-skipped smoke tests for real foundation model downloads."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.slow

if os.environ.get("ADACAST_RUN_FOUNDATION_SLOW") != "1":
    pytest.skip(
        "Set ADACAST_RUN_FOUNDATION_SLOW=1 to run heavyweight foundation model smoke tests.",
        allow_module_level=True,
    )


PYTHON_DIR = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SYNTHETIC_CSV = FIXTURES_DIR / "synthetic_etth1.csv"


def foundation_spec(tmp_path: Path) -> Path:
    """Write one representative zero-shot Chronos smoke spec."""
    spec = {
        "model": "Chronos",
        "dataset_path": str(SYNTHETIC_CSV),
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
        "phase": "forecast",
        "eval_split": "val",
        "seed": 42,
        "hyperparams": {},
        "metrics": ["mse", "mae", "wape", "mase"],
    }
    spec_path = tmp_path / "chronos_foundation_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


def parse_stderr_events(stderr: str) -> list[dict]:
    return [json.loads(line) for line in stderr.splitlines() if line.strip().startswith("{")]


def test_real_foundation_model_smoke_uses_cache_diagnostics(tmp_path: Path) -> None:
    """Run manually to validate HF mirror/cache behavior with heavy dependencies.

    Operators can configure HF_ENDPOINT, HF_HUB_CACHE, HF_XET_CACHE,
    HF_ASSETS_CACHE, and TRANSFORMERS_CACHE before enabling this test.
    Without explicit cache env, AdaCast should use .forecast/model-cache.
    Structured diagnostics are expected to include cache_config plus one of
    cache_hit, cache_miss, cache_warning, or dependency_missing.
    """
    spec_path = foundation_spec(tmp_path)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(PYTHON_DIR),
            "--extra",
            "heavy-models",
            "python",
            "-m",
            "adacast_ml.runner",
            "--spec",
            str(spec_path),
            "--out-dir",
            str(tmp_path / "runs"),
            "--evaluator-hash-store",
            str(tmp_path / ".forecast" / "evaluator.sha256"),
        ],
        cwd=PYTHON_DIR,
        capture_output=True,
        text=True,
        timeout=900,
    )

    events = parse_stderr_events(result.stderr)
    event_types = {event["type"] for event in events}
    stdout_payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert result.returncode in (0, 1)
    assert stdout_payload["status"] in {"success", "cached", "error"}
    assert "cache_config" in event_types
    assert event_types & {"cache_hit", "cache_miss", "cache_warning", "dependency_missing"}
    assert any(".forecast/model-cache" in json.dumps(event) for event in events) or os.environ.get(
        "HF_HUB_CACHE"
    )
