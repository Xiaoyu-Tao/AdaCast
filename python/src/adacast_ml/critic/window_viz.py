from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaCast window-level prediction visualization")
    parser.add_argument("--best-run-dir", required=True, help="Path to the best run directory containing pred.npy")
    parser.add_argument("--actual-npy", required=True, help="Path to the actual values .npy file")
    parser.add_argument(
        "--round-run-dirs",
        nargs="*",
        default=[],
        help="Run directories for ensemble variance CI estimation",
    )
    parser.add_argument("--out", required=True, help="Output PNG file path")
    parser.add_argument(
        "--fallback-ci-pct",
        type=float,
        default=0.1,
        help="Fallback CI band as fraction of predicted value when < 2 runs available",
    )
    return parser.parse_args(argv)


def load_predictions(run_dir: str | Path) -> pd.DataFrame | None:
    predictions_path = Path(run_dir) / "predictions.csv"
    if not predictions_path.exists():
        return None
    frame = pd.read_csv(predictions_path)
    required = {"timestamp", "actual", "predicted"}
    if not required.issubset(frame.columns):
        return None
    grouped = (
        frame.loc[:, ["timestamp", "actual", "predicted"]]
        .groupby("timestamp", sort=False, as_index=False)
        .agg(actual=("actual", "mean"), predicted=("predicted", "mean"))
    )
    grouped["_parsed_timestamp"] = pd.to_datetime(grouped["timestamp"], errors="coerce")
    grouped = grouped.sort_values(
        ["_parsed_timestamp", "timestamp"], kind="stable", na_position="last"
    )
    return grouped.drop(columns=["_parsed_timestamp"]).reset_index(drop=True)


def compute_ci(
    predictions_list: list[np.ndarray],
    predicted: np.ndarray,
    fallback_pct: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    if len(predictions_list) >= 2:
        stacked = np.stack(predictions_list, axis=0)
        std = np.std(stacked, axis=0)
        ci_lower = predicted - 1.96 * std
        ci_upper = predicted + 1.96 * std
        label = "95% CI (ensemble variance)"
    else:
        band = np.maximum(np.abs(predicted), 1e-8) * fallback_pct
        ci_lower = predicted - band
        ci_upper = predicted + band
        label = f"CI (+/-{fallback_pct * 100:.0f}% band)"
    return ci_lower, ci_upper, label


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    best_frame = load_predictions(args.best_run_dir)
    if best_frame is None:
        print(json.dumps({"error": "valid predictions.csv not found in best run directory", "generated": False}, sort_keys=True))
        sys.exit(1)

    predicted = best_frame["predicted"].to_numpy(dtype=np.float64)
    actual = best_frame["actual"].to_numpy(dtype=np.float64)
    timestamps = best_frame["timestamp"].astype(str).to_numpy()
    min_len = int(len(best_frame))
    if min_len == 0:
        print(json.dumps({"error": "No overlapping values available to plot", "generated": False}, sort_keys=True))
        sys.exit(1)

    predicted = predicted[:min_len]
    actual = actual[:min_len]

    ensemble_predictions: list[np.ndarray] = [predicted]
    best_run_path = Path(args.best_run_dir).resolve()
    for run_dir in args.round_run_dirs:
        run_path = Path(run_dir).resolve()
        if run_path == best_run_path:
            continue
        round_frame = load_predictions(run_path)
        if round_frame is None:
            continue
        aligned = best_frame.loc[:, ["timestamp"]].merge(
            round_frame.loc[:, ["timestamp", "predicted"]],
            on="timestamp",
            how="left",
            validate="one_to_one",
        )
        flat = aligned["predicted"].to_numpy(dtype=np.float64)
        if flat.size != min_len or not np.isfinite(flat).all():
            continue
        ensemble_predictions.append(flat)

    ci_lower, ci_upper, ci_label = compute_ci(ensemble_predictions, predicted, args.fallback_ci_pct)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(min_len)
    ax.plot(x, actual, color="steelblue", linewidth=1.5, label="Actual")
    ax.plot(x, predicted, color="tomato", linestyle="--", linewidth=1.5, label="Predicted")
    ax.fill_between(x, ci_lower, ci_upper, alpha=0.2, color="tomato", label=ci_label)
    ax.set_xlabel("Chronological target timestamp")
    ax.set_ylabel("Value")
    ax.set_title("Prediction Window: Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="png", dpi=120)
    plt.close(fig)

    print(
        json.dumps(
            {
                "ci_label": ci_label,
                "ensemble_size": len(ensemble_predictions),
                "generated": True,
                "aggregation": "mean_prediction_by_timestamp",
                "output_path": args.out,
                "raw_points": int(
                    pd.read_csv(Path(args.best_run_dir) / "predictions.csv").shape[0]
                ),
                "start_time": str(timestamps[0]),
                "end_time": str(timestamps[-1]),
                "time_steps": min_len,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
