"""Modified evaluator — computes MSE, MAE, WAPE, MASE and writes eval.json.

THIS FILE HAS BEEN MODIFIED BY THE AGENT.
LLM agents MUST NOT modify this file. Integrity is enforced by:
- INFR-04: Agent permission rules (deny write python/evaluator.py) — enforced in Phase 4
- SAFE-04 / D-10: SHA-256 hash verification before each evaluation call
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adacast_lite_ml.engine.utils.hash import compute_spec_hash, verify_evaluator
from adacast_lite_ml.engine.utils.metrics import mae, mase, mse, wape


def evaluate(
    run_dir: Path,
    spec: dict,
    evaluator_hash_store: Path,
    predictions_path: Path | None = None,
    train_values: np.ndarray | None = None,
) -> dict:
    """Compute evaluation metrics, write eval.json, and return its payload."""
    evaluator_path = Path(__file__)
    evaluator_hash = verify_evaluator(evaluator_path, evaluator_hash_store)

    if predictions_path is None:
        predictions_path = run_dir / "predictions.csv"

    pred_df = pd.read_csv(predictions_path)
    pred_values = pred_df["predicted"].to_numpy(dtype=np.float64)
    true_values = pred_df["actual"].to_numpy(dtype=np.float64)

    metrics_result: dict[str, float | None] = {
        "mse": mse(pred_values, true_values),
        "mae": mae(pred_values, true_values),
        "wape": wape(pred_values, true_values),
        "mase": None,
    }
    if train_values is not None:
        metrics_result["mase"] = mase(pred_values, true_values, train_values)

    spec_hash = compute_spec_hash(spec)
    eval_result = {
        "run_id": spec_hash,
        "evaluator_hash": evaluator_hash,
        "spec_hash": spec_hash,
        "metrics": metrics_result,
        "eval_split": spec.get("eval_split", "val"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    eval_path = run_dir / "eval.json"
    eval_path.write_text(
        json.dumps(eval_result, indent=2, sort_keys=True), encoding="utf-8"
    )

    return eval_result
