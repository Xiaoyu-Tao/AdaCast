from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from adacast_ml.data_loader import load_dataset
from adacast_ml.runner import ForecastWindowDataset
from adacast_ml.utils.metrics import mae, mase, mse, wape


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a persistence baseline with the runner's exact split and sliding windows"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed run directory containing the canonical spec.json",
    )
    return parser.parse_args(argv)


def compute_persistence_baseline(run_dir: str | Path) -> dict[str, object]:
    run_path = Path(run_dir)
    spec = json.loads((run_path / "spec.json").read_text(encoding="utf-8"))
    eval_split = str(spec.get("eval_split", "val"))
    if eval_split not in {"val", "test"}:
        raise ValueError(
            "Persistence baseline is only defined for validation or test runs"
        )

    splits = load_dataset(spec, ["train", eval_split])
    target_index = int(splits["train"]["target_index"])
    dataset = ForecastWindowDataset(
        splits[eval_split],
        seq_len=int(spec["seq_len"]),
        label_len=int(spec.get("label_len", int(spec["seq_len"]) // 2)),
        pred_len=int(spec["pred_len"]),
        target_index=target_index,
        freq=str(spec.get("freq", "h")),
        input_contract_version=int(spec.get("input_contract_version", 1)),
        window_stride=spec.get("window_stride", 1),
        forecast_gap=spec.get("forecast_gap", 0),
    )

    starts = dataset.window_start_index + np.arange(len(dataset)) * dataset.window_stride
    target_data = dataset.data[:, target_index]
    last_observed = target_data[starts + dataset.seq_len - 1]
    horizon_offsets = np.arange(dataset.pred_len)
    target_indices = (
        starts[:, None]
        + dataset.seq_len
        + dataset.forecast_gap
        + horizon_offsets[None, :]
    )
    actual = target_data[target_indices].astype(np.float64, copy=False).reshape(-1)
    predicted = np.broadcast_to(
        last_observed[:, None], target_indices.shape
    ).astype(np.float64, copy=False).reshape(-1)
    train_target = np.asarray(splits["train"]["data"], dtype=np.float64)[
        :, target_index
    ]

    return {
        "baseline": "persistence_last_observed",
        "eval_split": eval_split,
        "scale": "runner_standardized_training_prefix",
        "window_count": len(dataset),
        "point_count": int(actual.size),
        "metrics": {
            "mse": mse(predicted, actual),
            "mae": mae(predicted, actual),
            "wape": wape(predicted, actual),
            "mase": mase(predicted, actual, train_target),
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = compute_persistence_baseline(args.run_dir)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        sys.exit(1)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
