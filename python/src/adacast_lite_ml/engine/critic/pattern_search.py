from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import correlate

from adacast_lite_ml.engine.data_loader import resolve_split_indices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaCast historical pattern similarity search")
    parser.add_argument(
        "--task-json",
        help="Frozen .forecast/task.json; preferred because it supports both split modes",
    )
    parser.add_argument("--dataset-path", help="Path to the source CSV dataset")
    parser.add_argument("--target-col", help="Name of the target column")
    parser.add_argument("--time-col", help="Name of the time/date column")
    parser.add_argument(
        "--query-window-start",
        type=int,
        help="Explicit start index of the query window in the dataset",
    )
    parser.add_argument(
        "--query-window-end",
        type=int,
        help="Explicit end index of the query window (exclusive)",
    )
    parser.add_argument(
        "--search-window-end",
        type=int,
        help="Explicit exclusive end index of the historical search corpus",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        help="Outer training ratio used to derive the leakage-safe search boundary",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        help="Validation ratio used with train-ratio to derive the test boundary",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        help="Forecast horizon used to derive the query immediately before the test split",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of most similar historical segments to return")
    return parser.parse_args(argv)


def pearson_sliding_similarity(series: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return Pearson correlation coefficients for each valid series window."""
    n = len(query)
    if n < 2 or len(series) < n:
        return np.array([], dtype=np.float64)

    query_std = float(query.std())
    if query_std < 1e-8:
        return np.zeros(len(series) - n + 1, dtype=np.float64)

    q_norm = (query - query.mean()) / (query_std + 1e-8)
    corr = correlate(series - series.mean(), q_norm, mode="valid")

    num_windows = len(series) - n + 1
    windows_std = np.array([series[index : index + n].std() for index in range(num_windows)], dtype=np.float64)
    windows_std = np.where(windows_std < 1e-8, 1e-8, windows_std)

    similarities = corr / (n * windows_std)
    return np.clip(similarities, -1.0, 1.0)


def build_matches(
    similarities: np.ndarray,
    time_values: np.ndarray,
    query_length: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0 or similarities.size == 0:
        return []

    k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-k:][::-1]

    matches: list[dict[str, Any]] = []
    for raw_index in top_indices:
        idx = int(raw_index)
        window_end = idx + query_length
        matches.append(
            {
                "end_index": window_end,
                "end_time": str(time_values[min(window_end - 1, len(time_values) - 1)]),
                "similarity": round(float(similarities[idx]), 4),
                "start_index": idx,
                "start_time": str(time_values[idx]),
            }
        )
    return matches


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.top_k <= 0:
        print(json.dumps({"error": "top_k must be positive", "matches": []}, sort_keys=True))
        sys.exit(1)

    task: dict[str, Any] | None = None
    if args.task_json:
        task_path = Path(args.task_json).resolve()
        try:
            task = json.loads(task_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            print(
                json.dumps(
                    {"error": f"Failed to read task JSON: {exc}", "matches": []},
                    sort_keys=True,
                )
            )
            sys.exit(1)
        dataset_path = Path(str(task["dataset_path"]))
        if not dataset_path.is_absolute():
            dataset_path = (task_path.parent.parent / dataset_path).resolve()
        args.dataset_path = str(dataset_path)
        args.target_col = str(task["target_col"])
        args.time_col = str(task["time_col"])
        args.pred_len = int(task["pred_len"])

    if not args.dataset_path or not args.target_col or not args.time_col:
        print(
            json.dumps(
                {
                    "error": (
                        "Provide --task-json, or provide dataset-path, target-col, "
                        "and time-col"
                    ),
                    "matches": [],
                },
                sort_keys=True,
            )
        )
        sys.exit(1)

    try:
        df = pd.read_csv(args.dataset_path)
    except Exception as exc:
        print(json.dumps({"error": f"Failed to read dataset: {exc}", "matches": []}, sort_keys=True))
        sys.exit(1)

    if args.target_col not in df.columns:
        print(
            json.dumps(
                {"error": f"Target column '{args.target_col}' not found in dataset", "matches": []},
                sort_keys=True,
            )
        )
        sys.exit(1)

    try:
        series = df[args.target_col].astype(float).to_numpy(copy=True)
    except Exception as exc:
        print(json.dumps({"error": f"Failed to parse target column as float: {exc}", "matches": []}, sort_keys=True))
        sys.exit(1)

    if args.time_col in df.columns:
        time_values = df[args.time_col].to_numpy(copy=True)
    else:
        time_values = np.arange(len(series))

    task_mode = task is not None
    derived_values = (args.train_ratio, args.val_ratio, args.pred_len)
    derived_mode = all(value is not None for value in derived_values)
    explicit_values = (
        args.query_window_start,
        args.query_window_end,
        args.search_window_end,
    )
    explicit_mode = all(value is not None for value in explicit_values)

    if not task_mode and not derived_mode and not explicit_mode:
        print(
            json.dumps(
                {
                    "error": (
                        "Provide either train-ratio, val-ratio, and pred-len, "
                        "or all three explicit window boundaries"
                    ),
                    "matches": [],
                },
                sort_keys=True,
            )
        )
        sys.exit(1)

    if task_mode:
        try:
            search_end, test_start = resolve_split_indices(df, task)
        except Exception as exc:
            print(json.dumps({"error": str(exc), "matches": []}, sort_keys=True))
            sys.exit(1)
        pred_len = int(args.pred_len)
        if pred_len <= 0:
            print(
                json.dumps(
                    {"error": "Invalid pred_len in task JSON", "matches": []},
                    sort_keys=True,
                )
            )
            sys.exit(1)
        start_idx = test_start - pred_len
        end_idx = test_start
    elif derived_mode:
        train_ratio = float(args.train_ratio)
        val_ratio = float(args.val_ratio)
        pred_len = int(args.pred_len)
        if (
            train_ratio <= 0
            or val_ratio < 0
            or train_ratio + val_ratio >= 1
            or pred_len <= 0
        ):
            print(
                json.dumps(
                    {
                        "error": "Invalid train-ratio, val-ratio, or pred-len",
                        "matches": [],
                    },
                    sort_keys=True,
                )
            )
            sys.exit(1)
        search_end = int(len(series) * train_ratio)
        test_start = search_end + int(len(series) * val_ratio)
        start_idx = test_start - pred_len
        end_idx = test_start
    else:
        start_idx = int(args.query_window_start)
        end_idx = int(args.query_window_end)
        search_end = int(args.search_window_end)

    if start_idx < 0 or end_idx > len(series) or start_idx >= end_idx:
        print(json.dumps({"error": f"Invalid query window [{start_idx}, {end_idx})", "matches": []}, sort_keys=True))
        sys.exit(1)

    query = series[start_idx:end_idx]
    if search_end <= 0 or search_end > start_idx:
        print(
            json.dumps(
                {
                    "error": (
                        "search-window-end must be positive and no later than "
                        "query-window-start"
                    ),
                    "matches": [],
                },
                sort_keys=True,
            )
        )
        sys.exit(1)

    train_series = series[:search_end]
    if len(train_series) < len(query):
        print(json.dumps({"error": "Training series shorter than query window", "matches": []}, sort_keys=True))
        sys.exit(1)

    similarities = pearson_sliding_similarity(train_series, query)
    matches = build_matches(similarities, time_values, len(query), args.top_k)

    print(
        json.dumps(
            {
                "matches": matches,
                "query_window_end": int(end_idx),
                "query_window_start": int(start_idx),
                "query_length": int(len(query)),
                "search_length": int(len(train_series)),
                "search_window_end": int(search_end),
                "top_k": min(args.top_k, len(similarities)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
