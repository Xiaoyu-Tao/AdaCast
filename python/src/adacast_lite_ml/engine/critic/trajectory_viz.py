from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaCast experiment trajectory visualization")
    parser.add_argument("--history-jsonl", required=True, help="Path to .forecast/history.jsonl")
    parser.add_argument("--out", required=True, help="Output PNG file path")
    return parser.parse_args(argv)


def load_experiment_metrics(history_path: str) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    attempt_index = 0
    with open(history_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "type" in entry:
                continue
            run_id = str(entry.get("run_id", ""))
            if not run_id or run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            spec = entry.get("spec")
            if not isinstance(spec, dict) or spec.get("eval_split", "val") != "val":
                continue
            attempt_index += 1

            metrics = entry.get("metrics")
            if entry.get("status") != "success" or not isinstance(metrics, dict):
                continue

            mse = metrics.get("mse")
            mae = metrics.get("mae")
            if mse is None or mae is None:
                continue

            try:
                results.append(
                    {
                        "mae": float(mae),
                        "model": str(entry.get("model", "unknown")),
                        "mse": float(mse),
                        "round": attempt_index,
                    }
                )
            except (TypeError, ValueError):
                continue

    return results, attempt_index


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    experiments, attempt_count = load_experiment_metrics(args.history_jsonl)

    if not experiments:
        print(json.dumps({"error": "No successful experiments found in history", "generated": False}, sort_keys=True))
        sys.exit(1)

    rounds = np.asarray([entry["round"] for entry in experiments], dtype=np.int64)
    mse_values = np.asarray([entry["mse"] for entry in experiments], dtype=np.float64)
    mae_values = np.asarray([entry["mae"] for entry in experiments], dtype=np.float64)
    mse_best = np.minimum.accumulate(mse_values)
    mae_best = np.minimum.accumulate(mae_values)

    fig, (mse_ax, mae_ax) = plt.subplots(1, 2, figsize=(14, 5))

    mse_ax.plot(rounds, mse_values, "o-", color="steelblue", alpha=0.7, label="MSE per round")
    mse_ax.plot(rounds, mse_best, "-", color="tomato", linewidth=2, label="Best-so-far")
    mse_ax.set_xlabel("Experiment round")
    mse_ax.set_ylabel("MSE")
    mse_ax.set_title("MSE trajectory")
    mse_ax.legend()
    mse_ax.grid(True, alpha=0.25)

    mae_ax.plot(rounds, mae_values, "o-", color="steelblue", alpha=0.7, label="MAE per round")
    mae_ax.plot(rounds, mae_best, "-", color="tomato", linewidth=2, label="Best-so-far")
    mae_ax.set_xlabel("Experiment round")
    mae_ax.set_ylabel("MAE")
    mae_ax.set_title("MAE trajectory")
    mae_ax.legend()
    mae_ax.grid(True, alpha=0.25)

    fig.suptitle("Experiment Metric Trajectory", fontsize=13)
    fig.tight_layout()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="png", dpi=120)
    plt.close(fig)

    print(
        json.dumps(
            {
                "best_mae": float(mae_best[-1]),
                "best_mse": float(mse_best[-1]),
                "attempt_count": attempt_count,
                "experiment_count": len(experiments),
                "generated": True,
                "output_path": args.out,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
