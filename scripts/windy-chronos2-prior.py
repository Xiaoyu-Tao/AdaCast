"""Generate the fixed 2024 windy MS known-future-covariate Chronos-2 prior.

Each task is independent: 96 historical points -> 96 future points, stride 96.
Only historical real_power and historical/future weather covariates enter the model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/windy.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/windy-chronos2-ms-visible")
    parser.add_argument("--batch-tasks", type=int, default=4)
    args = parser.parse_args()
    if args.batch_tasks < 1:
        parser.error("--batch-tasks must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prior.csv").exists():
        raise ValueError("prior.csv already exists; use a new output directory")

    frame = pd.read_csv(args.dataset)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    expected = pd.date_range("2024-01-01", "2024-12-31 23:45", freq="15min")
    if not timestamps.equals(expected):
        raise ValueError("Dataset must be the exact regular 35,136-row 2024 grid")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    covariates = [c for c in frame.columns if c not in {"date", "real_power"}
                  and np.isfinite(numeric[c].to_numpy(dtype=np.float64)).all()]
    omitted = [c for c in frame.columns if c not in {"date", "real_power", *covariates}]
    target = numeric["real_power"].to_numpy(dtype=np.float32)
    weather = {c: numeric[c].to_numpy(dtype=np.float32) for c in covariates}
    if not np.isfinite(target).all() or not covariates:
        raise ValueError("A finite target and numeric weather covariates are required")
    train_end = timestamps.get_loc(pd.Timestamp("2024-09-02 18:15")) + 1
    val_start = timestamps.get_loc(pd.Timestamp("2024-09-03 00:00"))
    test_start = timestamps.get_loc(pd.Timestamp("2024-11-12 00:00"))
    assert (train_end, val_start, test_start) == (23594, 23616, 30336)
    origins = list(range(val_start, len(frame) - 95, 96))
    assert len(origins) == 120 and sum(i < test_start for i in origins) == 70

    tasks = [{"task_index": i, "split": "validation" if origin < test_start else "test",
              "input_start": str(timestamps[origin - 96]), "input_end": str(timestamps[origin - 1]),
              "label_start": str(timestamps[origin]), "label_end": str(timestamps[origin + 95]),
              "origin_row": origin} for i, origin in enumerate(origins)]
    pd.DataFrame(tasks).to_csv(output / "tasks.csv", index=False)
    print(json.dumps({"event": "data_checked", "rows": len(frame), "training_points": train_end,
                      "history_only_bridge_points": val_start - train_end, "validation_tasks": 70,
                      "test_tasks": 50, "covariates": len(covariates), "omitted": omitted}), flush=True)

    import torch
    from chronos import Chronos2Pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    start = time.monotonic()
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, revision=REVISION, device_map="cuda", torch_dtype=torch.bfloat16)
    resolved = getattr(pipeline.model.config, "_commit_hash", None)
    if resolved != REVISION:
        raise ValueError(f"Unexpected model revision: {resolved}")
    predictions = []
    for offset in range(0, len(origins), args.batch_tasks):
        batch_origins = origins[offset:offset + args.batch_tasks]
        inputs = [{"target": target[origin - 96:origin].copy(),
                   "past_covariates": {c: values[origin - 96:origin].copy() for c, values in weather.items()},
                   "future_covariates": {c: values[origin:origin + 96].copy() for c, values in weather.items()}}
                  for origin in batch_origins]
        # No future target is provided. Tasks never share their rolling histories.
        with torch.inference_mode():
            _, point = pipeline.predict_quantiles(inputs, prediction_length=96, context_length=96,
                                                   quantile_levels=[0.1, 0.5, 0.9],
                                                   batch_size=args.batch_tasks * (1 + len(covariates)),
                                                   cross_learning=False)
        for value in point:
            array = value.detach().cpu().float().numpy().reshape(-1)
            if array.shape != (96,) or not np.isfinite(array).all():
                raise ValueError("Chronos-2 returned invalid point forecasts")
            predictions.append(array)
        np.save(output / "point_forecasts.partial.npy", np.asarray(predictions), allow_pickle=False)
        print(json.dumps({"event": "predicted", "tasks_completed": len(predictions), "tasks_total": len(origins),
                          "seconds": round(time.monotonic() - start, 2)}), flush=True)

    forecasts = np.asarray(predictions)
    np.save(output / "point_forecasts.npy", forecasts, allow_pickle=False)
    (output / "point_forecasts.partial.npy").unlink()
    # Preserve original column names (including the empty column), ordering and
    # all non-target strings; replace only the declared validation/test labels.
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    records = []
    for task, predicted in zip(tasks, forecasts, strict=True):
        origin = task["origin_row"]
        for horizon, value in enumerate(predicted):
            index = origin + horizon
            rows[index]["real_power"] = repr(float(value))
            records.append({"timestamp": frame.iloc[index]["date"], "split": task["split"],
                            "task_index": task["task_index"], "horizon_index": horizon,
                            "origin": origin, "row_index": index,
                            "actual_original": float(frame.iloc[index]["real_power"]),
                            "predicted_original": float(value)})
    with (output / "prior.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(records).to_csv(output / "predictions.csv", index=False)
    actual = numeric["real_power"].to_numpy(dtype=np.float64)
    train_mean = float(actual[:train_end].mean())
    train_std = float(actual[:train_end].std())
    summaries = {}
    for split in ["validation", "test"]:
        subset = [r for r in records if r["split"] == split]
        observed = np.array([r["actual_original"] for r in subset])
        predicted = np.array([r["predicted_original"] for r in subset])
        error = predicted - observed
        summaries[split] = {"tasks": len(subset) // 96, "points": len(subset),
                            "mse_original": float(np.mean(error ** 2)), "mae_original": float(np.mean(abs(error))),
                            "wape_original": float(np.sum(abs(error)) / np.sum(abs(observed))),
                            "mse_standardized": float(np.mean(error ** 2) / train_std ** 2),
                            "mae_standardized": float(np.mean(abs(error)) / train_std)}
    manifest = {"model": MODEL, "revision": REVISION, "mode": "zero_shot_MS_known_future_covariates",
                "target": "real_power", "past_covariates": covariates, "future_covariates": covariates,
                "omitted_columns": omitted, "seq_len": 96, "pred_len": 96, "forecast_gap": 0,
                "window_stride": 96, "cross_learning": False, "point_forecast": "native median (0.5 quantile)",
                "postprocessing": "none", "dtype": "bfloat16", "device": torch.cuda.get_device_name(), "seed": 42,
                "dataset_path": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
                "prior_sha256": sha256(output / "prior.csv"), "training_rows": [0, train_end],
                "history_only_bridge_rows": [train_end, val_start], "validation_label_rows": [val_start, test_start],
                "test_label_rows": [test_start, len(frame)], "training_target_mean": train_mean,
                "training_target_std": train_std, "metrics": summaries,
                "versions": {name: importlib.metadata.version(name) for name in ["chronos-forecasting", "torch", "transformers", "numpy", "pandas"]},
                "inference_seconds": round(time.monotonic() - start, 2)}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"event": "completed", "prior": str(output / "prior.csv"), "metrics": summaries}), flush=True)


if __name__ == "__main__":
    main()
