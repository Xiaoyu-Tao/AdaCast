"""Deterministic data boundaries, training-only scales, and prediction coordinates."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path


def sha_file(file):
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_safe(v) for v in value]
    return value


def dump(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(file.name + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(file)


def stamp(text):
    import pandas as pd
    return pd.to_datetime(text,utc=True,errors="raise")


class Data:
    def __init__(self, task):
        self.task = task
        self.path = Path(task["dataset_path"])
        with self.path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self.columns = reader.fieldnames or []
            self.rows = list(reader)
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("CSV has duplicate column names")
        if task["time_col"] not in self.columns or task["target_col"] not in self.columns:
            raise ValueError("Missing time or target column")
        if task.get("exogenous_columns") is not None:
            selected = task["exogenous_columns"]
            if len(set(selected)) != len(selected) or set(selected) & {task["time_col"], task["target_col"]} or set(selected)-set(self.columns):
                raise ValueError("Invalid or missing exogenous_columns")
        if task.get("exogenous_history_steps") is not None:
            steps = task["exogenous_history_steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or not 0 < steps <= task["seq_len"]:
                raise ValueError("exogenous_history_steps must be a positive integer no larger than seq_len")
        if any(None in row or any(v is None for v in row.values()) for row in self.rows):
            raise ValueError("CSV contains malformed rows")
        self.times = [row[task["time_col"]] for row in self.rows]
        import pandas as pd
        from .frequencies import normalize_frequency
        parsed = pd.DatetimeIndex(pd.to_datetime(self.times,format="mixed",errors="raise",utc=True))
        if parsed.hasnans or not parsed.is_unique or not parsed.is_monotonic_increasing:
            raise ValueError("CSV timestamps must be unique and increasing")
        self.frequency = normalize_frequency(task["frequency"])
        expected = pd.date_range(parsed[0],periods=len(parsed),freq=self.frequency)
        self.regular = parsed.equals(expected)
        if task.get("strict_frequency",False) and not self.regular:
            raise ValueError("CSV timestamps must match the declared frequency in strict mode")
        self.values = [float(row[task["target_col"]]) for row in self.rows]
        if not self.values or not all(math.isfinite(v) for v in self.values):
            raise ValueError("Target values must be finite numbers")
        n = len(self.values)
        if task.get("split_mode", "ratio") == "timestamp":
            def index(boundary):
                target = stamp(boundary)
                found = [i for i, t in enumerate(parsed) if t == target]
                if len(found) != 1:
                    raise ValueError("Timestamp boundary must match exactly one row")
                return found[0] + 1
            self.t = index(task["train_end_timestamp"])
            self.v = index(task["validation_end_timestamp"])
        else:
            ratios = [task[k] for k in ("train_ratio", "val_ratio", "test_ratio")]
            if any(not 0 < r < 1 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
                raise ValueError("Split ratios must be positive and sum to 1")
            self.t = math.floor(n * ratios[0])
            self.v = self.t + math.floor(n * ratios[1])
        self.validation_start = self.t
        if task.get("validation_start_timestamp") is not None:
            if task.get("split_mode", "ratio") != "timestamp":
                raise ValueError("validation_start_timestamp requires timestamp split mode")
            self.validation_start = index(task["validation_start_timestamp"]) - 1
            if not self.t <= self.validation_start < self.v:
                raise ValueError("Validation must start after training and before validation ends")
        self.a = self.validation_start + (self.v - self.validation_start) // 2
        seq, horizon = task["seq_len"], task.get("forecast_gap", 0) + task["pred_len"]
        if not 0 < self.t < self.a < self.v < n or self.t < seq + horizon:
            raise ValueError("Insufficient training data or invalid split boundaries")
        if min(self.a - self.validation_start, self.v - self.a, n - self.v) < horizon:
            raise ValueError("adjustment_fit, validation and test each need a complete forecast window")
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from .engine.data_loader import numeric_feature_columns
        frame = pd.read_csv(self.path, low_memory=False)
        self.feature_columns = numeric_feature_columns(frame, task["time_col"], task["target_col"])
        self.target_index = self.feature_columns.index(task["target_col"])
        self.raw_features = frame[self.feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        scaler = StandardScaler().fit(self.raw_features[:self.t])
        self.features = scaler.transform(self.raw_features)
        self.means = scaler.mean_.tolist()
        self.scales = scaler.scale_.tolist()
        self.mean = self.means[self.target_index]
        self.scale = self.scales[self.target_index]
        self.std = float(np.sqrt(scaler.var_[self.target_index]))
        self.y = self.features[:, self.target_index].tolist()
        self.mase_scale = float(np.mean(np.abs(np.diff(self.features[:self.t, self.target_index].astype(np.float64)))))

    def boundaries(self):
        return {"train": [0, self.t], "adjustment_fit": [self.validation_start, self.a], "val": [self.a, self.v], "test": [self.v, len(self.y)]}

    def origins(self, split):
        lo, hi = self.boundaries()[split]
        lo = max(lo, self.task["seq_len"])
        end = hi - self.task.get("forecast_gap", 0) - self.task["pred_len"] + 1
        return range(lo, end, self.task.get("window_stride", 1))

    def info(self):
        train = self.values[:self.t]
        return {"rows": len(self.y), "columns": self.columns, "dataset_sha256": sha_file(self.path),
                "boundaries": self.boundaries(), "window_counts": {s: len(self.origins(s)) for s in self.boundaries()},
                "train_statistics": {"count": self.t, "mean": self.mean, "std": self.std,
                                     "min": min(train), "max": max(train), "last": train[-1]},
                "scale": {"mean": self.mean, "std": self.scale},
                "feature_columns": self.feature_columns, "target_index": self.target_index,
                "target": self.task["target_col"], "frequency": self.frequency,
                "time_grid": {"mode":"rows", "regular":self.regular, "strict":self.task.get("strict_frequency",False)}}


def metrics(points, mase_scale):
    if not points:
        raise ValueError("No prediction points")
    import numpy as np
    actual = np.asarray([p["actual"] for p in points], dtype=np.float64)
    predicted = np.asarray([p["predicted"] for p in points], dtype=np.float64)
    absolute = np.abs(actual-predicted)
    mae = float(np.mean(absolute))
    denominator = float(np.sum(np.abs(actual)))
    return {"mse": float(np.mean((actual-predicted)**2)), "mae": mae,
            "wape": float(np.sum(absolute)/denominator) if denominator else float("inf"),
            "mase": mae/mase_scale if mase_scale else float("inf")}


def save_predictions(file, points):
    with Path(file).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)


def load_predictions(file):
    with Path(file).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("actual", "predicted", "actual_original", "predicted_original"):
            row[key] = float(row[key])
        for key in ("window_index", "horizon_index", "origin", "row_index"):
            row[key] = int(row[key])
    return rows
