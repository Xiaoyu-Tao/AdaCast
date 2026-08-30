from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import adacast_ml.reflection.controller as reflection_controller

from adacast_ml.reflection.controller import (
    calibrate_soft_constraints,
    check_response,
    derive_policy,
    detect_window,
    finalize,
    nested_forbidden_keys,
    prepare,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_skill_guided_active_recalibration_covers_at_least_twenty_percent() -> None:
    selected = [
        ordinal
        for ordinal in range(1, 57)
        if reflection_controller.skill_guided_active_recalibration_priority(ordinal)
    ]
    assert selected == [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56]
    assert len(selected) / 56 >= 0.20
    with pytest.raises(ValueError, match="ordinal must be positive"):
        reflection_controller.skill_guided_active_recalibration_priority(0)


def synthetic_contract(tmp_path: Path, *, pred_len: int = 3) -> tuple[Path, Path, Path]:
    timestamps = pd.date_range("2026-01-01", periods=120, freq="15min")
    target = 100.0 + np.sin(np.arange(120) / 8.0) * 10.0
    frame = pd.DataFrame(
        {
            "date": timestamps,
            "target": target,
            "temperature": np.linspace(-5.0, 8.0, 120),
            "weather": ["clear"] * 120,
        }
    )
    dataset_path = tmp_path / "series.csv"
    frame.to_csv(dataset_path, index=False)
    task = {
        "dataset_path": str(dataset_path),
        "target_col": "target",
        "time_col": "date",
        "pred_len": pred_len,
        "forecast_gap": 2,
        "seq_len": 4,
        "label_len": 4,
        "window_stride": 3,
        "train_ratio": 0.5,
        "val_ratio": 0.3,
        "test_ratio": 0.2,
        "adjustment_fit_holdout_ratio": 0.5,
        "eval_metrics": ["mse", "mae", "wape", "mase"],
        "model_families": ["DLinear"],
        "reflection_enabled": True,
        "frozen_at": "2026-07-15T00:00:00Z",
    }
    task_path = tmp_path / ".forecast" / "task.json"
    write_json(task_path, task)

    train = target[:60]
    mean = float(np.mean(train))
    scale = float(np.std(train))
    rows = []
    for window_index, start in enumerate((80, 83)):
        for horizon_index in range(pred_len):
            value = float(target[start + horizon_index])
            rows.append(
                {
                    "timestamp": str(timestamps[start + horizon_index]),
                    "actual": (value - mean) / scale,
                    "predicted": (value - mean) / scale,
                    "prediction_layout": "sliding_window",
                    "window_index": window_index,
                    "horizon_index": horizon_index,
                }
            )
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    policy = derive_policy(task, task_path)
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    return task_path, predictions_path, policy_path


def test_controller_cli_preserves_unicode_paths_on_windows_style_workspaces(tmp_path: Path) -> None:
    unicode_root = tmp_path / "中文路径（反思测试）"
    unicode_root.mkdir()
    task_path, _, _ = synthetic_contract(unicode_root)
    policy_path = unicode_root / "反思策略.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "adacast_ml.reflection.controller",
            "configure",
            "--task-json",
            str(task_path),
            "--out",
            str(policy_path),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )

    payload = json.loads(completed.stdout)
    assert payload["policy_path"] == str(policy_path)
    assert policy_path.exists()


def test_policy_rejects_planner_limit_that_excludes_training_values(tmp_path: Path) -> None:
    task_path, _, _ = synthetic_contract(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="may not exclude"):
        derive_policy(task, task_path, {"absolute_max": 50.0})


def test_policy_calibrates_on_validation_front_without_using_later_targets(tmp_path: Path) -> None:
    task_path, _, _ = synthetic_contract(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    dataset_path = Path(task["dataset_path"])
    frame = pd.read_csv(dataset_path)
    # train=0..59, validation-front=60..77, validation-back=78..95, test=96..119
    frame.loc[60:77, "target"] = np.linspace(120.0, 150.0, 18)
    frame.loc[78:, "target"] = 1_000_000.0
    frame.to_csv(dataset_path, index=False)

    first = derive_policy(task, task_path)
    assert first["controller_version"] == "2"
    calibration = first["calibration"]
    assert calibration["method"] == "training_baseline_validation_front_calibration_v2"
    assert calibration["source_split"] == "validation_front"
    assert calibration["training_row_count"] == 60
    assert calibration["validation_front_row_count"] == 18
    assert calibration["validation_front_window_count"] == 5
    assert calibration["validation_back_used"] is False
    assert calibration["test_used"] is False
    assert calibration["validation_front_window_statistics"]["window_count"] == 5
    replay = calibration["validation_front_reference_replay"]
    assert replay["hard_violation_window_count"] == 0
    assert replay["final_trigger_rate"] <= replay["max_reference_trigger_rate"]
    assert calibration["calibrated_rules"] == [
        "absolute_range",
        "adjacent_step",
        "curvature",
        "severe_flat_window",
        "flat_window",
        "window_mean_correction",
        "point_emergency_correction",
    ]
    assert first["target_scale"]["validation_front"]["observed_max"] == pytest.approx(150.0)
    assert first["target_scale"]["calibrated"]["observed_max"] == pytest.approx(150.0)
    assert first["constraints"]["absolute_max"] >= 150.0

    # Changing held-out validation-back and test targets must not change the policy.
    frame.loc[78:, "target"] = -1_000_000.0
    frame.to_csv(dataset_path, index=False)
    second = derive_policy(task, task_path)
    assert second["constraints"] == first["constraints"]
    assert second["target_scale"] == first["target_scale"]

    with pytest.raises(ValueError, match="validation-front observations"):
        derive_policy(task, task_path, {"absolute_max": 140.0})


def test_policy_reader_never_loads_validation_back_or_test_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_path, _, _ = synthetic_contract(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(task["dataset_path"])
    task.update(
        {
            "split_mode": "timestamp",
            "train_end_timestamp": str(frame.iloc[59]["date"]),
            "validation_end_timestamp": str(frame.iloc[95]["date"]),
        }
    )
    write_json(task_path, task)
    calls: list[dict[str, object]] = []
    original_read_csv = pd.read_csv

    def traced_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(dict(kwargs))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(reflection_controller.pd, "read_csv", traced_read_csv)
    derive_policy(task, task_path)

    target_reads = [
        call
        for call in calls
        if call.get("usecols") == [task["target_col"]]
    ]
    assert len(target_reads) == 1
    assert target_reads[0]["nrows"] == 78


@pytest.mark.parametrize(
    ("constant_value", "pred_len"),
    [(0.0, 1), (100.0, 2), (-250.0, 3), (1e12, 2), (1e-12, 1)],
)
def test_policy_handles_constant_targets_and_short_prediction_windows(
    tmp_path: Path,
    constant_value: float,
    pred_len: int,
) -> None:
    task_path, _, _ = synthetic_contract(tmp_path, pred_len=pred_len)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(task["dataset_path"])
    frame["target"] = constant_value
    frame.to_csv(task["dataset_path"], index=False)

    policy = derive_policy(task, task_path)

    constraints = policy["constraints"]
    assert constraints["absolute_min"] < constraints["absolute_max"]
    assert constraints["soft_absolute_min"] < constraints["soft_absolute_max"]
    assert 0 < constraints["soft_abs_step"] <= constraints["max_abs_step"]
    assert 0 < constraints["soft_abs_curvature"] <= constraints["max_abs_curvature"]
    assert constraints["hard_min_window_range"] <= constraints["min_window_range"]
    replay = policy["calibration"]["validation_front_reference_replay"]
    assert replay["hard_violation_window_count"] == 0
    assert replay["final_trigger_count"] == 0


def test_legacy_v1_policy_keeps_its_version_through_final_artifacts(tmp_path: Path) -> None:
    task_path, predictions_path, policy_path = synthetic_contract(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["controller_version"] = "1"
    policy.pop("calibration", None)
    policy["constraints"] = {
        key: value
        for key, value in policy["constraints"].items()
        if key
        not in {
            "soft_absolute_min",
            "soft_absolute_max",
            "soft_abs_curvature",
            "max_abs_curvature",
            "hard_min_window_range",
            "min_window_range",
        }
    }
    policy["constraints"]["absolute_max"] = 50.0
    write_json(policy_path, policy)
    out_dir = tmp_path / "legacy-reflection"

    prepared = prepare(
        Namespace(
            task_json=str(task_path),
            policy_json=str(policy_path),
            predictions_csv=str(predictions_path),
            candidate_json=None,
            adjustment_candidate_id=None,
            baseline_kind="raw_anchor",
            source_run_id="run-val",
            split="val",
            out_dir=str(out_dir),
        )
    )
    assert prepared["controller_version"] == "1"
    context = json.loads(Path(prepared["windows"][0]["blind_context_path"]).read_text(encoding="utf-8"))
    assert context["controller_version"] == "1"

    finalized = finalize(
        Namespace(
            prepared_json=prepared["prepared_path"],
            corrections_dir=str(out_dir / "corrections"),
            out_dir=str(out_dir),
        )
    )
    assert finalized["controller_version"] == "1"


@pytest.mark.parametrize("seed", range(8))
def test_policy_calibration_stays_valid_across_deterministic_stress_series(
    tmp_path: Path,
    seed: int,
) -> None:
    pred_len = 1 + seed % 6
    task_path, _, _ = synthetic_contract(tmp_path, pred_len=pred_len)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(task["dataset_path"])
    rng = np.random.default_rng(seed)
    index = np.arange(len(frame), dtype=np.float64)
    scale = 10.0 ** (seed - 3)
    offset = -2.0 * scale if seed % 3 == 0 else 100.0 * scale
    frame["target"] = offset + scale * (
        0.03 * index
        + 3.0 * np.sin(index / (3.0 + seed))
        + rng.normal(0.0, 0.15 + seed * 0.02, len(frame))
    )
    if seed == 7:
        frame.loc[70, "target"] += 25.0 * scale
    frame.to_csv(task["dataset_path"], index=False)

    policy = derive_policy(task, task_path)

    constraints = policy["constraints"]
    assert all(
        math.isfinite(float(value))
        for value in constraints.values()
        if not isinstance(value, bool)
    )
    assert constraints["absolute_min"] < constraints["soft_absolute_min"]
    assert constraints["soft_absolute_min"] < constraints["soft_absolute_max"]
    assert constraints["soft_absolute_max"] < constraints["absolute_max"]
    assert constraints["soft_abs_step"] <= constraints["max_abs_step"]
    assert constraints["soft_abs_curvature"] <= constraints["max_abs_curvature"]
    assert constraints["hard_min_window_range"] <= constraints["min_window_range"]
    replay = policy["calibration"]["validation_front_reference_replay"]
    assert replay["hard_violation_window_count"] == 0
    assert replay["final_trigger_rate"] <= replay["max_reference_trigger_rate"]


def test_detector_adds_calibrated_envelope_curvature_and_flatness_checks(tmp_path: Path) -> None:
    task_path, _, policy_path = synthetic_contract(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["constraints"].update(
        {
            "absolute_min": 0.0,
            "absolute_max": 1000.0,
            "soft_absolute_min": 90.0,
            "soft_absolute_max": 110.0,
            "soft_abs_step": 100.0,
            "max_abs_step": 200.0,
            "soft_abs_curvature": 15.0,
            "max_abs_curvature": 30.0,
            "hard_min_window_range": 1.0,
            "min_window_range": 5.0,
        }
    )

    flat_outside_envelope = detect_window(np.asarray([120.0, 120.0, 122.0]), policy)
    assert flat_outside_envelope["trigger"] is True
    assert {item["id"] for item in flat_outside_envelope["soft_warnings"]} == {
        "soft_absolute_envelope",
        "flat_window",
    }

    extreme_curvature = detect_window(np.asarray([100.0, 120.0, 100.0]), policy)
    assert extreme_curvature["trigger"] is True
    assert {item["id"] for item in extreme_curvature["hard_violations"]} == {
        "extreme_curvature"
    }

    severe_flat = detect_window(np.asarray([100.0, 100.0, 100.0]), policy)
    assert severe_flat["trigger"] is True
    assert {item["id"] for item in severe_flat["hard_violations"]} == {
        "severe_flat_window"
    }


def test_validation_front_replay_relaxes_overly_sensitive_soft_thresholds() -> None:
    windows = [
        np.asarray([0.0, 10.0, 0.0]),
        np.asarray([0.0, 20.0, 0.0]),
        np.asarray([0.0, 30.0, 0.0]),
    ]
    constraints = {
        "nonnegative": True,
        "absolute_min": 0.0,
        "absolute_max": 100.0,
        "soft_absolute_min": 0.0,
        "soft_absolute_max": 100.0,
        "soft_abs_step": 5.0,
        "max_abs_step": 100.0,
        "soft_abs_curvature": 5.0,
        "max_abs_curvature": 200.0,
        "hard_min_window_range": 0.0,
        "min_window_range": 0.0,
        "window_mean_abs_delta_limit": 50.0,
        "point_emergency_delta_limit": 200.0,
    }

    calibrated, replay = calibrate_soft_constraints(windows, constraints)

    assert replay["initial_trigger_rate"] == 1.0
    assert replay["final_trigger_rate"] <= replay["max_reference_trigger_rate"]
    assert replay["soft_threshold_adjustment_rounds"] > 0
    assert calibrated["soft_abs_step"] > constraints["soft_abs_step"]
    assert calibrated["soft_abs_curvature"] > constraints["soft_abs_curvature"]


def test_detector_requires_hard_or_two_independent_soft_signals(tmp_path: Path) -> None:
    task_path, _, policy_path = synthetic_contract(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["constraints"]["soft_abs_step"] = 1.0
    policy["constraints"]["max_abs_step"] = 100.0
    policy["constraints"]["window_mean_abs_delta_limit"] = 5.0
    policy["constraints"].update(
        {
            "soft_absolute_min": 0.0,
            "soft_absolute_max": 1000.0,
            "soft_abs_curvature": 1000.0,
            "max_abs_curvature": 2000.0,
            "hard_min_window_range": 0.0,
            "min_window_range": 0.0,
        }
    )
    values = np.asarray([100.0, 110.0, 110.0])

    one_soft = detect_window(values, policy)
    assert one_soft["trigger"] is False
    assert {item["id"] for item in one_soft["soft_warnings"]} == {"abrupt_step"}

    two_soft = detect_window(
        values,
        policy,
        reference=np.asarray([90.0, 100.0, 100.0]),
        matched_rule_ids=["skill-rule"],
        require_adjustment_evidence=True,
    )
    assert two_soft["trigger"] is True
    assert {item["id"] for item in two_soft["soft_warnings"]} == {
        "abrupt_step",
        "large_supported_mean_delta",
    }

    policy["constraints"]["max_abs_step"] = 5.0
    extreme = detect_window(values, policy)
    assert extreme["trigger"] is True
    assert {item["id"] for item in extreme["hard_violations"]} == {"extreme_step"}


def test_three_day_exogenous_compaction_is_lossless_and_positional() -> None:
    timestamps = pd.date_range("2026-01-01", periods=5 * 96, freq="15min")
    frame = pd.DataFrame(
        {
            "date": timestamps,
            "target": np.arange(len(timestamps), dtype=np.float64),
            "dynamic_weather": np.arange(len(timestamps), dtype=np.float64) / 10.0,
            "special_day": np.repeat(["day-1", "day-2", "day-3", "day-4", "day-5"], 96),
        }
    )

    recent, roles = reflection_controller.recent_three_day_frame(
        frame,
        time_column="date",
        first_target=pd.Timestamp("2026-01-05 00:00:00"),
    )
    assert roles == {
        "yesterday": "2026-01-03",
        "forecast_issue_day_today": "2026-01-04",
        "target_day_tomorrow": "2026-01-05",
    }
    assert len(recent) == 3 * 96
    assert pd.Timestamp(recent.iloc[0]["date"]) == pd.Timestamp("2026-01-03 00:00:00")
    assert pd.Timestamp(recent.iloc[-1]["date"]) == pd.Timestamp("2026-01-05 23:45:00")

    table = reflection_controller.positional_table(
        recent,
        time_column="date",
        value_columns=["dynamic_weather", "special_day"],
    )
    assert table["columns"] == ["timestamp", "dynamic_weather", "special_day"]
    assert table["row_count"] == 3 * 96
    assert all(len(row) == len(table["columns"]) for row in table["rows"])
    assert "target" not in table["columns"]

    daily = reflection_controller.daily_constant_exogenous_table(
        frame,
        time_column="date",
        exogenous_columns=["dynamic_weather", "special_day"],
    )
    assert daily["columns"] == ["date", "special_day"]
    assert daily["row_count"] == 5
    assert daily["rows"] == [
        ["2026-01-01", "day-1"],
        ["2026-01-02", "day-2"],
        ["2026-01-03", "day-3"],
        ["2026-01-04", "day-4"],
        ["2026-01-05", "day-5"],
    ]


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("天气", "小雪", "weather:snow"),
        ("天气", "中雨", "weather:rain_or_storm"),
        ("天气", "多云", "weather:cloudy"),
        ("节假日", "元旦复工", "event_phase:resume_or_recovery"),
        ("环保", "解除环保管控", "event_phase:end_or_release"),
        ("政策", "启动有序用电", "event_phase:start_or_restrict"),
        ("政策", "两会召开", "event_phase:start_or_restrict"),
        ("政策", "93阅兵后第一天", "event_phase:post_event"),
    ],
)
def test_provider_safe_text_projection_preserves_chinese_semantics(
    column: str, value: str, expected: str
) -> None:
    assert reflection_controller.provider_safe_text_value(column, value) == expected


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        (
            "\u8282\u5047\u65e5",
            "\u56fd\u5e86\u8282\u590d\u5de5",
            {
                "semantic": "event_phase:resume_or_recovery",
                "domain": "calendar_or_holiday",
                "event_type": "national_day",
            },
        ),
        (
            "\u653f\u7b56",
            "\u5929\u6d25\u89e3\u9664\u9632\u6c5b\u7528\u7535\u653f\u7b56",
            {
                "semantic": "event_phase:end_or_release",
                "domain": "policy",
                "topics": ["flood_response", "electricity_policy"],
            },
        ),
        (
            "\u653f\u7b56",
            "\u4e24\u4f1a\u53ec\u5f00",
            {
                "semantic": "event_phase:start_or_restrict",
                "domain": "policy",
                "topics": ["major_conference"],
            },
        ),
        (
            "\u653f\u7b56",
            "93\u9605\u5175\u540e\u7b2c\u4e00\u5929",
            {
                "semantic": "event_phase:post_event",
                "domain": "policy",
                "topics": ["major_public_event"],
            },
        ),
        ("\u662f\u5426\u6781\u7aef\u5929\u6c14", "False", {"semantic": "none", "boolean": False}),
        ("\u65e5\u7167", "8.4\u5c0f\u65f6", {"hours": 8.4}),
        ("\u65e5\u7167", "0", {"semantic": "none", "hours": 0.0}),
        (
            "\u98ce\u5411\u98ce\u529b",
            "\u4e1c\u5317\u98ce2\u7ea7",
            {"direction": "northeast", "level": 2},
        ),
    ],
)
def test_provider_safe_text_context_preserves_joint_exogenous_meaning(
    column: str, value: str, expected: dict[str, object]
) -> None:
    actual = reflection_controller.provider_safe_text_context(column, value)
    for key, expected_value in expected.items():
        assert actual[key] == expected_value


def test_skill_guided_test_matches_drive_llm_context_and_two_level_fallback(tmp_path: Path) -> None:
    task_path, predictions_path, policy_path = synthetic_contract(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["business_context"] = "Synthetic load forecast"
    task["holiday_effects"] = "raw holiday transition detail must not reach the provider"
    task["anomaly_periods"] = "raw anomaly incident detail must not reach the provider"
    write_json(task_path, task)
    reports_dir = task_path.parent / "reports"
    reports_dir.mkdir()
    write_json(
        reports_dir / "quantitative.json",
        {
            "target_col": "target",
            "time_col": "date",
            "analysis_scope": {"target_scope": "training_split_only"},
            "analysis": {
                "trend": {"direction": "upward", "slope": 0.2},
                "seasonality": {"dominant_period": 24, "strength": 0.8},
            },
            "analyzer_version": "test",
        },
    )
    (reports_dir / "pre-forecast.md").write_text(
        "# Qualitative analysis\nStable domain structure; model rankings still require validation.",
        encoding="utf-8",
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["constraints"].update(
        {
            "absolute_min": 0.0,
            "absolute_max": 1000.0,
            "soft_absolute_min": 0.0,
            "soft_absolute_max": 1000.0,
            "soft_abs_step": 1000.0,
            "max_abs_step": 2000.0,
            "soft_abs_curvature": 1000.0,
            "max_abs_curvature": 2000.0,
            "hard_min_window_range": 0.0,
            "min_window_range": 0.0,
            "window_mean_abs_delta_limit": 100.0,
            "point_emergency_delta_limit": 5000.0,
        }
    )
    write_json(policy_path, policy)

    predictions = pd.read_csv(predictions_path)
    dataset = pd.read_csv(task["dataset_path"])
    dataset.loc[pd.to_datetime(dataset["date"]).dt.date == pd.Timestamp("2026-01-02").date(), "weather"] = "storm"
    dataset.to_csv(task["dataset_path"], index=False)
    train = dataset["target"].iloc[:60].to_numpy(dtype=np.float64)
    mean = float(np.mean(train))
    scale = float(np.std(train))
    raw_original = predictions["predicted"].to_numpy(dtype=np.float64) * scale + mean
    baseline_original = raw_original + 1.0
    baseline_original[:3] = 2000.0
    predictions["raw_predicted_original"] = raw_original
    predictions["adjusted_predicted_original"] = baseline_original
    predictions["predicted"] = (baseline_original - mean) / scale
    adjusted_path = tmp_path / "adjusted_predictions.csv"
    predictions.to_csv(adjusted_path, index=False)

    candidate = {
        "candidate_id": "locked-skill",
        "rules": [
            {
                "id": "rule-always",
                "condition": {
                    "kind": "exogenous_text_in",
                    "column": "weather",
                    "values": ["storm"],
                    "day_offset": 1,
                },
                "action": {"type": "add", "value": 1.0},
                "bounds": {"min_delta_original": 0.0, "max_delta_original": 5.0},
                "rationale": "Use a small positive correction when the sample evidence supports it.",
                "evidence_refs": ["validation-front:synthetic"],
            }
        ],
    }
    candidate_path = tmp_path / "candidate.json"
    write_json(candidate_path, candidate)

    timestamps = pd.to_datetime(dataset["date"])
    gap_rows = []
    for window_index, target_start in enumerate((80, 83)):
        for gap_index, row_index in enumerate(range(target_start - 2, target_start)):
            original_value = float(dataset.iloc[row_index]["target"])
            gap_rows.append(
                {
                    "timestamp": str(timestamps.iloc[row_index]),
                    "predicted": (original_value - mean) / scale,
                    "prediction_layout": "forecast_gap",
                    "window_index": window_index,
                    "gap_index": gap_index,
                }
            )
    gap_path = tmp_path / "gap_predictions.csv"
    pd.DataFrame(gap_rows).to_csv(gap_path, index=False)

    out_dir = tmp_path / "skill-guided-reflection"
    prepared = prepare(
        Namespace(
            task_json=str(task_path),
                    policy_json=str(policy_path),
                    predictions_csv=str(predictions_path),
                    deterministic_predictions_csv=str(adjusted_path),
            gap_predictions_csv=str(gap_path),
            candidate_json=str(candidate_path),
            adjustment_candidate_id="locked-skill",
            baseline_kind="skill_adjusted",
            source_run_id="run-test",
            split="test",
            out_dir=str(out_dir),
        )
    )

    with pytest.raises(ValueError, match="candidate id does not match"):
        prepare(
            Namespace(
                task_json=str(task_path),
                policy_json=str(policy_path),
                    predictions_csv=str(predictions_path),
                    deterministic_predictions_csv=str(adjusted_path),
                gap_predictions_csv=str(gap_path),
                candidate_json=str(candidate_path),
                adjustment_candidate_id="different-skill",
                baseline_kind="skill_adjusted",
                source_run_id="run-test-mismatched-skill",
                split="test",
                out_dir=str(tmp_path / "mismatched-skill-reflection"),
            )
        )
    assert prepared["trigger_count"] == 1
    assert prepared["skill_match_count"] == 2
    assert prepared["llm_required_count"] == 2
    assert prepared["active_recalibration_priority_count"] == 1
    assert prepared["active_recalibration_minimum_share"] == pytest.approx(0.20)
    assert all(window["llm_required"] for window in prepared["windows"])
    assert prepared["windows"][0]["active_recalibration_priority"] is True
    assert prepared["windows"][1]["active_recalibration_priority"] is False
    prepared_sha256 = hashlib.sha256(Path(prepared["prepared_path"]).read_bytes()).hexdigest()
    first_window = prepared["windows"][0]
    assert first_window["blind_context_sha256"] == hashlib.sha256(
        Path(first_window["blind_context_path"]).read_bytes()
    ).hexdigest()
    first_context = json.loads(
        Path(prepared["windows"][0]["blind_context_path"]).read_text(encoding="utf-8")
    )
    assert first_context["reasoning_mode"] == "skill_guided_test"
    assert first_context["active_recalibration"]["priority"] is True
    assert first_context["active_recalibration"]["matched_window_ordinal"] == 1
    assert first_context["active_recalibration"]["minimum_share_of_skill_matched_windows"] == pytest.approx(0.20)
    assert first_context["active_recalibration"]["minimum_meaningful_points"] == 3
    assert "no target observations" in first_context["active_recalibration"]["selection_basis"]
    assert first_context["task_and_scenario"]["business_context"] == "Synthetic load forecast"
    assert first_context["task_and_scenario"]["holiday_effects"]["semantic_role"] == "holiday_effects"
    assert first_context["task_and_scenario"]["anomaly_periods"]["semantic_role"] == "anomaly_periods"
    serialized_context = json.dumps(first_context, ensure_ascii=False)
    assert "raw holiday transition detail" not in serialized_context
    assert "raw anomaly incident detail" not in serialized_context
    assert "history_rows" not in first_context
    assert "gap_and_future_known_rows" not in first_context
    history_table = first_context["visible_history_target_table"]
    assert history_table["columns"] == ["timestamp", "observed_value"]
    assert history_table["row_count"] == 4
    assert history_table["rows"][0][0] == str(timestamps.iloc[74])
    assert history_table["rows"][-1][0] == str(timestamps.iloc[77])
    lookback_exogenous = first_context["lookback_window_exogenous_table"]
    assert lookback_exogenous["columns"] == ["timestamp", "temperature", "weather"]
    assert lookback_exogenous["row_count"] == 4
    assert [row[0] for row in lookback_exogenous["rows"]] == [
        str(timestamp) for timestamp in timestamps.iloc[74:78]
    ]
    predict_exogenous = first_context["predict_window_exogenous_table"]
    assert predict_exogenous["columns"] == ["timestamp", "temperature", "weather"]
    assert predict_exogenous["row_count"] == 3
    assert [row[0] for row in predict_exogenous["rows"]] == [
        str(timestamp) for timestamp in timestamps.iloc[80:83]
    ]
    assert "target" not in lookback_exogenous["columns"]
    assert "target" not in predict_exogenous["columns"]
    recent_table = first_context["recent_three_day_exogenous_table"]
    assert recent_table["columns"] == ["timestamp", "temperature", "weather"]
    assert recent_table["row_count"] == 9
    assert all(len(row) == len(recent_table["columns"]) for row in recent_table["rows"])
    assert "target" not in recent_table["columns"]
    assert '"storm"' not in json.dumps(first_context, ensure_ascii=False)
    assert first_context["context_representation"]["recent_exogenous_date_roles"] == {
        "yesterday": "2025-12-30",
        "forecast_issue_day_today": "2025-12-31",
        "target_day_tomorrow": "2026-01-01",
    }
    sample_context = first_context["current_sample_exogenous_context"]
    assert sample_context["exogenous_column_count"] == 2
    assert sample_context["all_exogenous_columns"] == ["temperature", "weather"]
    assert sample_context["skill_primary_columns"] == ["weather"]
    assert sample_context["secondary_context_columns"] == ["temperature"]
    assert set(sample_context["day_profiles"]) == {
        "yesterday",
        "forecast_issue_day_today",
        "target_day_tomorrow",
    }
    assert "never exclusive" in sample_context["reasoning_contract"]
    assert set(sample_context["numeric_mean_changes"]) == {
        "target_day_tomorrow_minus_forecast_issue_day_today",
        "target_day_tomorrow_minus_yesterday",
    }
    assert "temperature" in sample_context["day_profiles"]["target_day_tomorrow"]["numeric"]
    candidates = first_context["evidence_candidates"]
    assert set(candidates) == {"classification_contract", "experience", "context", "knowledge"}
    assert candidates["experience"]["candidate_type"] == "experience"
    assert candidates["context"]["candidate_type"] == "context"
    assert candidates["knowledge"]["candidate_type"] == "knowledge"
    assert candidates["context"]["evidence_candidate_id"] == "context:current-window"
    assert candidates["context"]["scope"] == "current forecast window only"
    assert candidates["context"]["payload"]["current_sample_exogenous"] == sample_context
    assert candidates["context"]["payload"]["lookback_window_exogenous"] == lookback_exogenous
    assert candidates["context"]["payload"]["predict_window_exogenous"] == predict_exogenous
    assert "summaries are auxiliary" in candidates["context"]["payload"]["table_contract"]
    assert "non_negotiable_constraints" not in candidates["experience"]["payload"]
    assert "hard_physical_constraints" in candidates["knowledge"]["payload"]
    data_analysis = candidates["knowledge"]["payload"]["task_semantics_and_structure"]["data_analysis"]
    assert data_analysis["epistemic_status"] == "empirical_data_knowledge"
    assert data_analysis["quantitative_analysis"]["analysis"]["seasonality"]["dominant_period"] == 24
    assert "Stable domain structure" in data_analysis["qualitative_analysis"]["content"]
    critic_state = first_context["critic_prediction_state"]
    assert critic_state["version"] == 1
    assert critic_state["window_index"] == 0
    assert critic_state["evidence_gap"] == 1
    assert critic_state["needed_evidence"] == ""
    assert critic_state["evidences"] == []
    assert critic_state["predictions"] == [
        {
            "horizon_index": point["horizon_index"],
            "timestamp": point["timestamp"],
            "value": point["raw_anchor"],
        }
        for point in first_context["prediction_points"]
    ]
    assert first_window["evidence_candidate_ids"] == {
        "experience": "experience:locked-skill",
        "context": "context:current-window",
        "knowledge": "knowledge:frozen-task",
    }
    assert set(first_window["critic_prediction_state_history"]) == {"initial"}
    rule_reference_artifact = first_window["experience_rule_reference_artifact"]
    rule_reference = json.loads(Path(rule_reference_artifact["path"]).read_text(encoding="utf-8"))
    assert hashlib.sha256(Path(rule_reference_artifact["path"]).read_bytes()).hexdigest() == rule_reference_artifact["sha256"]
    assert rule_reference["semantics"] == "locked_rules_applied_to_initial_prediction_only"
    assert rule_reference["experience_candidate_id"] == "experience:locked-skill"
    assert rule_reference["experience_candidate_artifact"] == first_window["evidence_candidate_artifacts"]["experience"]
    assert rule_reference["reasoning_contract"] == {
        "reference_only": True,
        "not_a_state_transition": True,
        "do_not_add_reference_delta_to_current": True,
    }
    assert rule_reference["initial_predictions"] == critic_state["predictions"]
    assert rule_reference["predictions"] == [
        {
            "horizon_index": point["horizon_index"],
            "timestamp": point["timestamp"],
            "value": point["pre_reflection_baseline"],
        }
        for point in first_context["prediction_points"]
    ]
    state_artifact = first_window["critic_prediction_state_artifact"]
    state_artifact_path = Path(state_artifact["path"])
    assert json.loads(state_artifact_path.read_text(encoding="utf-8")) == critic_state
    assert hashlib.sha256(state_artifact_path.read_bytes()).hexdigest() == state_artifact["sha256"]
    initial_state_artifact = first_window["critic_prediction_state_history"]["initial"]
    initial_state = json.loads(Path(initial_state_artifact["path"]).read_text(encoding="utf-8"))
    assert state_artifact == initial_state_artifact
    assert initial_state["evidences"] == []
    assert initial_state["predictions"] == [
        {
            "horizon_index": point["horizon_index"],
            "timestamp": point["timestamp"],
            "value": point["raw_anchor"],
        }
        for point in first_context["prediction_points"]
    ]
    for candidate_type in ("experience", "context", "knowledge"):
        artifact = first_window["evidence_candidate_artifacts"][candidate_type]
        artifact_path = Path(artifact["path"])
        assert artifact_path.exists()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]
    daily_table = first_context["full_sample_daily_constant_exogenous_table"]
    assert daily_table["columns"] == ["date", "weather"]
    assert daily_table["row_count"] == 1
    assert "known_exogenous_summary" not in first_context["deterministic_sample_features"]
    assert first_context["deterministic_sample_features"]["full_sample_exogenous_summary"]["numeric"][
        "temperature"
    ]["count"] == 9
    assert len(first_context["gap_prediction_points"]) == 2
    assert first_context["gap_prediction_points"][0]["timestamp"] == str(timestamps.iloc[78])
    assert all("observed_value" not in point for point in first_context["gap_prediction_points"])
    assert "supporting_skill_rules" not in first_context
    locked = first_context["locked_skill_experience"]
    assert "immutable initial prediction" in locked["guidance_contract"]
    assert "must never be mechanically added" in locked["guidance_contract"]
    assert "generate only the residual effect" in locked["guidance_contract"]
    assert "Preserving the current prediction unchanged is valid" in locked["guidance_contract"]
    assert "already contains the deterministic result of every matching rule" in first_context["instructions"]["baseline_semantics"]
    assert "not a privileged baseline" in first_context["instructions"]["baseline_semantics"]
    assert "do not undo or halve their combined effect" in first_context["instructions"]["multi_rule_semantics"]
    assert "soft warning alone is not enough reason" in first_context["instructions"]["detector_semantics"]
    assert "Avoid isolated one-point spikes or troughs" in first_context["instructions"]["shape_guidance"]
    assert "at least two independent evidence families" in first_context["instructions"]["decision_process"]
    assert "every secondary_context_column" in first_context["instructions"]["exogenous_joint_reasoning"]
    assert "Do not search for or nominate one closest historical day" in first_context["instructions"]["analogue_semantics"]
    assert "not historical model residuals" in first_context["instructions"]["analogue_semantics"]
    assert "Rule bounds are safety limits" in first_context["instructions"]["direction_and_magnitude"]
    assert "never use those labels to amplify a downward correction" in first_context["instructions"]["direction_and_magnitude"]
    assert "smallest coherent evidence-supported adjustment" in first_context["instructions"]["when_uncertain"]
    assert "joint target-day context" in first_context["instructions"]["rationale_contract"]
    assert "secondary supporting and opposing or neutral evidence" in first_context["instructions"]["rationale_contract"]
    assert "residual effect not already represented" in first_context["instructions"]["rationale_contract"]
    assert locked["important_guidance_rules"][0]["matched_horizon_indices"] == [0, 1, 2]
    assert locked["important_guidance_rules"][0]["condition"]["values"] == ["weather:rain_or_storm"]
    assert first_context["provider_safe_projection"]["deterministic_matching_uses_original_values"] is True
    assert "window_mean_abs_delta_limit" not in locked["non_negotiable_constraints"]["frozen_physical_safety"]
    assert locked["important_guidance_rules"][0]["bounds"]["max_delta_original"] == 5.0
    match_evidence = first_context["deterministic_sample_features"]["rule_match_evidence_rows"]
    assert len(match_evidence) == 3
    assert match_evidence[0]["condition_exogenous_values"]["weather [day_offset=+1]"] == "weather:rain_or_storm"
    assert nested_forbidden_keys(first_context) == []

    prepared_payload = json.loads(Path(prepared["prepared_path"]).read_text(encoding="utf-8"))
    first = prepared_payload["windows"][0]
    second = prepared_payload["windows"][1]
    response_path = tmp_path / "response.json"

    frozen_initial_state = json.loads(
        Path(first["critic_prediction_state_history"]["initial"]["path"]).read_text(encoding="utf-8")
    )
    assessed_initial_state = {**frozen_initial_state, "evidence_gap": 0}
    write_json(
        response_path,
        {
            "window_index": 0,
            "points": assessed_initial_state["predictions"],
            "rationale": "The prediction prior is already sufficiently supported.",
            "evidence_ids": [],
            "assessment_stage": "initial_prior",
            "critic_prediction_state": assessed_initial_state,
        },
    )
    sufficient_prior = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            prepared_sha256=prepared_sha256,
            response_json=str(response_path),
            window_index=0,
            round=0,
            out=None,
        )
    )
    assert sufficient_prior["accepted"] is True
    assert sufficient_prior["assessment_stage"] == "initial_prior"

    terminal_state = {
        **frozen_initial_state,
        "evidences": [{
            "evidence_candidate_id": "experience:locked-skill",
            "candidate_type": "experience",
            "artifact_path": first["evidence_candidate_artifacts"]["experience"]["path"],
            "artifact_sha256": first["evidence_candidate_artifacts"]["experience"]["sha256"],
        }],
        "predictions": json.loads(
            Path(first["experience_rule_reference_artifact"]["path"]).read_text(encoding="utf-8")
        )["predictions"],
        "evidence_gap": 0,
        "needed_evidence": "",
    }
    terminal_state_path = tmp_path / "deterministic-terminal-state.json"
    write_json(terminal_state_path, terminal_state)
    terminal_state_artifact = {
        "path": str(terminal_state_path),
        "sha256": hashlib.sha256(terminal_state_path.read_bytes()).hexdigest(),
    }
    write_json(
        response_path,
        {
            "window_index": 0,
            "points": terminal_state["predictions"],
            "rationale": "Experience evidence closed the gap.",
            "evidence_ids": ["experience:locked-skill"],
            "assessment_stage": "evidence_loop_terminal",
            "critic_prediction_state": terminal_state,
            "critic_prediction_state_artifact": terminal_state_artifact,
        },
    )
    terminal = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            prepared_sha256=prepared_sha256,
            response_json=str(response_path),
            window_index=0,
            round=0,
            out=None,
        )
    )
    assert terminal["accepted"] is True
    assert terminal["detector_report"]["acceptance"] == "evidence_loop_terminal"
    assert terminal["detector_report"]["semantic_safety_policy_applied"] is False

    second_initial_state = json.loads(
        Path(second["critic_prediction_state_history"]["initial"]["path"]).read_text(encoding="utf-8")
    )
    safe_terminal_state = {
        **second_initial_state,
        "evidences": [{
            "evidence_candidate_id": "experience:locked-skill",
            "candidate_type": "experience",
            "artifact_path": second["evidence_candidate_artifacts"]["experience"]["path"],
            "artifact_sha256": second["evidence_candidate_artifacts"]["experience"]["sha256"],
        }],
        "predictions": json.loads(
            Path(second["experience_rule_reference_artifact"]["path"]).read_text(encoding="utf-8")
        )["predictions"],
        "evidence_gap": 0,
        "needed_evidence": "",
    }
    safe_terminal_state_path = tmp_path / "safe-deterministic-terminal-state.json"
    write_json(safe_terminal_state_path, safe_terminal_state)
    write_json(
        response_path,
        {
            "window_index": 1,
            "points": safe_terminal_state["predictions"],
            "rationale": "Experience evidence closed the gap for the safe window.",
            "evidence_ids": ["experience:locked-skill"],
            "assessment_stage": "evidence_loop_terminal",
            "critic_prediction_state": safe_terminal_state,
            "critic_prediction_state_artifact": {
                "path": str(safe_terminal_state_path),
                "sha256": hashlib.sha256(safe_terminal_state_path.read_bytes()).hexdigest(),
            },
        },
    )
    safe_terminal = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            prepared_sha256=prepared_sha256,
            response_json=str(response_path),
            window_index=1,
            round=0,
            out=None,
        )
    )
    assert safe_terminal["accepted"] is True
    assert safe_terminal["supported_evidence_ids"] == ["experience:locked-skill"]

    tampered_initial_state = {
        **assessed_initial_state,
        "predictions": [
            {**point, "value": float(point["value"]) + 1.0}
            for point in assessed_initial_state["predictions"]
        ],
    }
    write_json(
        response_path,
        {
            "window_index": 0,
            "points": tampered_initial_state["predictions"],
            "rationale": "Attempt to alter the prior while claiming an initial short circuit.",
            "evidence_ids": [],
            "assessment_stage": "initial_prior",
            "critic_prediction_state": tampered_initial_state,
        },
    )
    tampered_prior = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            prepared_sha256=prepared_sha256,
            response_json=str(response_path),
            window_index=0,
            round=0,
            out=None,
        )
    )
    assert tampered_prior["accepted"] is False
    assert any(
        item["id"] == "missing_selected_evidence"
        for item in tampered_prior["detector_report"]["hard_violations"]
    )

    write_json(
        response_path,
        {
            "window_index": 0,
            "points": [
                {
                    "horizon_index": index,
                    "timestamp": first["timestamps"][index],
                    "value": first["baseline_original"][index],
                }
                for index in range(3)
            ],
            "rationale": "Copy the deterministic anchor.",
            "evidence_ids": ["experience:locked-skill"],
        },
    )
    copied_priority = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            prepared_sha256=prepared_sha256,
            response_json=str(response_path),
            window_index=0,
            round=1,
            out=None,
        )
    )
    assert copied_priority["accepted"] is False
    assert not any(
        item["id"] == "active_recalibration_not_meaningful"
        for item in copied_priority["detector_report"]["hard_violations"]
    )

    write_json(
        response_path,
        {
            "window_index": 0,
            "points": [
                {
                    "horizon_index": index,
                    "timestamp": first["timestamps"][index],
                    "value": first["raw_anchor_original"][index],
                }
                for index in range(3)
            ],
            "rationale": "Use the raw anchor as the independently recalibrated safe alternative.",
            "evidence_ids": ["context:current-window"],
        },
    )
    meaningful_priority = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            prepared_sha256=prepared_sha256,
            response_json=str(response_path),
            window_index=0,
            round=2,
            out=None,
        )
    )
    assert meaningful_priority["accepted"] is True

    write_json(
        response_path,
        {
            "window_index": 1,
            "points": [
                {
                    "horizon_index": index,
                    "timestamp": second["timestamps"][index],
                    "value": second["baseline_original"][index],
                }
                for index in range(3)
            ],
            "rationale": "Keep the deterministic anchor.",
            "evidence_ids": [],
        },
    )
    unsupported = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            prepared_sha256=prepared_sha256,
            response_json=str(response_path),
            window_index=1,
            round=1,
            out=None,
        )
    )
    assert unsupported["accepted"] is False
    assert unsupported["detector_report"]["hard_violations"][-1]["id"] == "missing_selected_evidence"

    write_json(
        out_dir / "corrections" / "window-000000.json",
        {
            "accepted": True,
            "window_index": 0,
            "points": [
                {
                    "horizon_index": index,
                    "timestamp": first["timestamps"][index],
                    "value": first["baseline_original"][index],
                }
                for index in range(3)
            ],
            "rationale": "Pretend an unsafe persisted correction was accepted.",
            "evidence_ids": ["rule-always"],
        },
    )
    protocol_path = out_dir / "llm-protocol.json"
    write_json(protocol_path, {"protocol_version": 1, "mode": "skill_guided_test"})
    with pytest.raises(ValueError, match="protocol changed before finalization"):
        finalize(
            Namespace(
                prepared_json=prepared["prepared_path"],
                prepared_sha256=prepared_sha256,
                corrections_dir=str(out_dir / "corrections"),
                out_dir=str(out_dir),
                llm_protocol_json=str(protocol_path),
                llm_protocol_sha256="0" * 64,
            )
        )
    finalized = finalize(
        Namespace(
            prepared_json=prepared["prepared_path"],
            prepared_sha256=prepared_sha256,
            corrections_dir=str(out_dir / "corrections"),
            out_dir=str(out_dir),
            llm_protocol_json=None,
        )
    )
    assert finalized["fallback_count"] == 2
    assert finalized["raw_fallback_windows"] == [0, 1]
    assert finalized["deterministic_fallback_windows"] == []
    output = pd.read_csv(finalized["final_predictions_path"])
    assert output.loc[:2, "predicted"].to_numpy() == pytest.approx(
        (raw_original[:3] - mean) / scale
    )
    assert output.loc[3:, "predicted"].to_numpy() == pytest.approx(
        (raw_original[3:] - mean) / scale
    )
    assert set(output.loc[:2, "reflection_fallback_kind"]) == {"raw_anchor"}
    assert set(output.loc[3:, "reflection_fallback_kind"]) == {"raw_anchor"}

    tampered_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    tampered_candidate["rules"][0]["rationale"] = "Changed after blind-context preparation."
    write_json(candidate_path, tampered_candidate)
    with pytest.raises(ValueError, match="source candidate_json changed"):
        check_response(
            Namespace(
                prepared_json=prepared["prepared_path"],
                prepared_sha256=prepared_sha256,
                response_json=str(response_path),
                window_index=1,
                round=1,
                out=None,
            )
        )
    write_json(candidate_path, candidate)

    context_artifact_path = Path(first_window["evidence_candidate_artifacts"]["context"]["path"])
    original_context_artifact = context_artifact_path.read_bytes()
    context_artifact_path.write_bytes(original_context_artifact + b"\n")
    with pytest.raises(ValueError, match="context evidence artifact changed"):
        check_response(
            Namespace(
                prepared_json=prepared["prepared_path"],
                prepared_sha256=prepared_sha256,
                response_json=str(response_path),
                window_index=1,
                round=1,
                out=None,
            )
        )
    context_artifact_path.write_bytes(original_context_artifact)

    rule_reference_path = Path(first_window["experience_rule_reference_artifact"]["path"])
    original_rule_reference = rule_reference_path.read_bytes()
    rule_reference_path.write_bytes(original_rule_reference + b"\n")
    with pytest.raises(ValueError, match="experience rule reference changed"):
        check_response(
            Namespace(
                prepared_json=prepared["prepared_path"],
                prepared_sha256=prepared_sha256,
                response_json=str(response_path),
                window_index=1,
                round=1,
                out=None,
            )
        )
    rule_reference_path.write_bytes(original_rule_reference)

    original_state_artifact = state_artifact_path.read_bytes()
    state_artifact_path.write_bytes(original_state_artifact + b"\n")
    with pytest.raises(ValueError, match="Critic prediction state changed"):
        check_response(
            Namespace(
                prepared_json=prepared["prepared_path"],
                prepared_sha256=prepared_sha256,
                response_json=str(response_path),
                window_index=1,
                round=1,
                out=None,
            )
        )
    state_artifact_path.write_bytes(original_state_artifact)

    unmatched_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    unmatched_candidate["candidate_id"] = "unmatched-skill"
    unmatched_candidate["rules"][0]["condition"]["values"] = ["never-observed"]
    unmatched_candidate_path = tmp_path / "unmatched-candidate.json"
    write_json(unmatched_candidate_path, unmatched_candidate)
    unmatched = prepare(
        Namespace(
            task_json=str(task_path),
            policy_json=str(policy_path),
            predictions_csv=str(adjusted_path),
            gap_predictions_csv=str(gap_path),
            candidate_json=str(unmatched_candidate_path),
            adjustment_candidate_id="unmatched-skill",
            baseline_kind="skill_adjusted",
            source_run_id="run-test-unmatched",
            split="test",
            out_dir=str(tmp_path / "unmatched-reflection"),
        )
    )
    assert unmatched["skill_match_count"] == 0
    assert unmatched["llm_required_count"] == unmatched["window_count"]
    assert all(window["blind_context_path"] is not None for window in unmatched["windows"])
    for window in unmatched["windows"]:
        context = json.loads(Path(window["blind_context_path"]).read_text(encoding="utf-8"))
        assert context["critic_prediction_state"]["evidences"] == []
        assert set(context["evidence_candidates"]) == {
            "classification_contract",
            "context",
            "knowledge",
        }
    raw_test = prepare(
        Namespace(
            task_json=str(task_path),
            policy_json=str(policy_path),
            predictions_csv=str(predictions_path),
            gap_predictions_csv=str(gap_path),
            candidate_json=None,
            adjustment_candidate_id=None,
            baseline_kind="raw_anchor",
            source_run_id="run-test-raw",
            split="test",
            out_dir=str(tmp_path / "raw-test-reflection"),
        )
    )
    assert raw_test["llm_required_count"] == raw_test["window_count"]
    for window in raw_test["windows"]:
        assert window["reasoning_mode"] == "evidence_loop_test"
        assert window["critic_prediction_state_history"].keys() == {"initial"}
        context = json.loads(Path(window["blind_context_path"]).read_text(encoding="utf-8"))
        assert context["critic_prediction_state"]["evidences"] == []
        assert set(context["evidence_candidates"]) == {
            "classification_contract",
            "context",
            "knowledge",
        }
    extra_gap = pd.read_csv(gap_path)
    extra_gap.loc[len(extra_gap)] = {
        "timestamp": extra_gap.iloc[0]["timestamp"],
        "predicted": extra_gap.iloc[0]["predicted"],
        "prediction_layout": "forecast_gap",
        "window_index": 99,
        "gap_index": 0,
    }
    invalid_gap_path = tmp_path / "invalid-gap-predictions.csv"
    extra_gap.to_csv(invalid_gap_path, index=False)
    with pytest.raises(ValueError, match="exactly 4 rows"):
        prepare(
            Namespace(
                task_json=str(task_path),
                policy_json=str(policy_path),
                predictions_csv=str(adjusted_path),
                gap_predictions_csv=str(invalid_gap_path),
                candidate_json=str(candidate_path),
                adjustment_candidate_id="locked-skill",
                baseline_kind="skill_adjusted",
                source_run_id="run-test",
                split="test",
                out_dir=str(tmp_path / "invalid-gap-reflection"),
            )
        )

    prepared_payload["tampered"] = True
    write_json(Path(prepared["prepared_path"]), prepared_payload)
    with pytest.raises(ValueError, match="Prepared reflection manifest changed"):
        finalize(
            Namespace(
                prepared_json=prepared["prepared_path"],
                prepared_sha256=prepared_sha256,
                corrections_dir=str(out_dir / "corrections"),
                out_dir=str(out_dir),
                llm_protocol_json=None,
            )
        )


def test_prepare_context_uses_only_exact_visible_history_and_known_future_exogenous(tmp_path: Path) -> None:
    task_path, predictions_path, policy_path = synthetic_contract(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["constraints"]["absolute_max"] = 50.0
    write_json(policy_path, policy)
    out_dir = tmp_path / "reflection"
    result = prepare(
        Namespace(
            task_json=str(task_path),
            policy_json=str(policy_path),
            predictions_csv=str(predictions_path),
            candidate_json=None,
            adjustment_candidate_id=None,
            baseline_kind="raw_anchor",
            source_run_id="run-val",
            split="val",
            out_dir=str(out_dir),
        )
    )
    assert result["trigger_count"] == 2
    context = json.loads(Path(result["windows"][0]["blind_context_path"]).read_text(encoding="utf-8"))
    assert len(context["history_rows"]) == 4
    # target starts at row 80, gap=2, so visible history is rows 74..77.
    dataset = pd.read_csv(json.loads(task_path.read_text())["dataset_path"])
    assert context["history_rows"][0]["timestamp"] == str(dataset.iloc[74]["date"])
    assert context["history_rows"][-1]["timestamp"] == str(dataset.iloc[77]["date"])
    assert len(context["gap_and_future_known_rows"]) == 5
    assert nested_forbidden_keys(context) == []
    assert all("observed_value" not in row for row in context["gap_and_future_known_rows"])
    initial_state = context["critic_prediction_state"]
    assert initial_state["evidences"] == []
    assert initial_state["evidence_gap"] == 1
    assert initial_state["needed_evidence"] == ""
    assert initial_state["predictions"] == [
        {
            "horizon_index": point["horizon_index"],
            "timestamp": point["timestamp"],
            "value": point["raw_anchor"],
        }
        for point in context["prediction_points"]
    ]
    state_artifact = result["windows"][0]["critic_prediction_state_artifact"]
    assert state_artifact is not None
    assert hashlib.sha256(Path(state_artifact["path"]).read_bytes()).hexdigest() == state_artifact["sha256"]


def test_response_repairs_two_missing_points_and_negative_validation_disables_test_llm(tmp_path: Path) -> None:
    task_path, predictions_path, policy_path = synthetic_contract(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["constraints"]["absolute_max"] = 50.0
    policy["constraints"]["window_mean_abs_delta_limit"] = 1000.0
    policy["constraints"]["point_emergency_delta_limit"] = 5000.0
    write_json(policy_path, policy)
    out_dir = tmp_path / "reflection"
    prepared = prepare(
        Namespace(
            task_json=str(task_path),
            policy_json=str(policy_path),
            predictions_csv=str(predictions_path),
            candidate_json=None,
            adjustment_candidate_id=None,
            baseline_kind="raw_anchor",
            source_run_id="run-val",
            split="val",
            out_dir=str(out_dir),
        )
    )
    prepared_payload = json.loads(Path(prepared["prepared_path"]).read_text(encoding="utf-8"))
    first = prepared_payload["windows"][0]
    response_path = tmp_path / "response.json"
    write_json(
        response_path,
        {
            "window_index": 0,
            "points": [
                {
                    "horizon_index": 0,
                    "timestamp": first["timestamps"][0],
                    "value": first["baseline_original"][0] + 10.0,
                }
            ],
            "rationale": "Synthetic test correction",
            "evidence_ids": [],
        },
    )
    corrections = out_dir / "corrections"
    accepted_path = corrections / "window-000000.json"
    checked = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            response_json=str(response_path),
            window_index=0,
            round=1,
            out=str(accepted_path),
        )
    )
    assert checked["accepted"] is False  # Physical absolute bound remains violated.
    assert checked["missing_count"] == 2

    # Widen the physical limit and re-check: two missing points are filled from baseline.
    policy["constraints"]["absolute_max"] = 1000.0
    policy["constraints"]["soft_absolute_min"] = 0.0
    policy["constraints"]["soft_absolute_max"] = 1000.0
    policy["constraints"]["soft_abs_step"] = 1000.0
    policy["constraints"]["max_abs_step"] = 2000.0
    policy["constraints"]["soft_abs_curvature"] = 1000.0
    policy["constraints"]["max_abs_curvature"] = 2000.0
    policy["constraints"]["hard_min_window_range"] = 0.0
    policy["constraints"]["min_window_range"] = 0.0
    write_json(policy_path, policy)
    checked = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            response_json=str(response_path),
            window_index=0,
            round=2,
            out=str(accepted_path),
        )
    )
    assert checked["accepted"] is True
    assert checked["missing_count"] == 2
    assert checked["points"][1]["value"] == pytest.approx(first["baseline_original"][1])

    finalized = finalize(
        Namespace(
            prepared_json=prepared["prepared_path"],
            corrections_dir=str(corrections),
            out_dir=str(out_dir),
        )
    )
    assert finalized["status"] == "disabled_negative_mse"
    assert finalized["delta_mse"] > 0
    output = pd.read_csv(finalized["final_predictions_path"])
    assert len(output) == len(pd.read_csv(predictions_path))
    assert output["reflection_applied"].sum() == 3
    assert finalized["fallback_count"] == 1
    assert finalized["final_predictions_sha256"] == hashlib.sha256(
        Path(finalized["final_predictions_path"]).read_bytes()
    ).hexdigest()
    assert finalized["leakage_audit_sha256"] == hashlib.sha256(
        (out_dir / "leakage-audit.json").read_bytes()
    ).hexdigest()


def test_response_with_more_than_two_missing_points_requires_retry(tmp_path: Path) -> None:
    task_path, predictions_path, policy_path = synthetic_contract(tmp_path, pred_len=4)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["constraints"]["absolute_max"] = 50.0
    write_json(policy_path, policy)
    out_dir = tmp_path / "reflection"
    prepared = prepare(
        Namespace(
            task_json=str(task_path),
            policy_json=str(policy_path),
            predictions_csv=str(predictions_path),
            candidate_json=None,
            adjustment_candidate_id=None,
            baseline_kind="raw_anchor",
            source_run_id="run-val",
            split="val",
            out_dir=str(out_dir),
        )
    )
    response_path = tmp_path / "empty-response.json"
    write_json(response_path, {"window_index": 0, "points": [], "rationale": "", "evidence_ids": []})
    checked = check_response(
        Namespace(
            prepared_json=prepared["prepared_path"],
            response_json=str(response_path),
            window_index=0,
            round=1,
            out=None,
        )
    )
    assert checked["accepted"] is False
    assert checked["format_retry"] is True
    assert checked["missing_count"] == 4


def test_repaired_skill_window_is_rechecked_against_raw_anchor_and_declared_evidence(tmp_path: Path) -> None:
    task_path, _, policy_path = synthetic_contract(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["constraints"].update(
        {
            "absolute_min": 0.0,
            "absolute_max": 1000.0,
            "soft_absolute_min": 0.0,
            "soft_absolute_max": 1000.0,
            "soft_abs_step": 1000.0,
            "max_abs_step": 1000.0,
            "soft_abs_curvature": 1000.0,
            "max_abs_curvature": 1000.0,
            "hard_min_window_range": 0.0,
            "min_window_range": 0.0,
            "window_mean_abs_delta_limit": 20.0,
            "point_emergency_delta_limit": 1000.0,
        }
    )
    write_json(policy_path, policy)
    prepared_path = tmp_path / "prepared-skill.json"
    prepared = {
        "policy_json": str(policy_path),
        "baseline_kind": "skill_adjusted",
        "windows": [
            {
                "window_index": 0,
                "horizon_indices": [0, 1, 2],
                "timestamps": ["t0", "t1", "t2"],
                "raw_anchor_original": [100.0, 100.0, 100.0],
                "baseline_original": [200.0, 200.0, 200.0],
                "detection": {"matched_rule_ids": ["rule-1"]},
            }
        ],
    }
    write_json(prepared_path, prepared)
    response_path = tmp_path / "skill-response.json"

    def run_check(evidence_ids: list[str]) -> dict[str, object]:
        write_json(
            response_path,
            {
                "window_index": 0,
                "points": [
                    {"horizon_index": index, "timestamp": f"t{index}", "value": 200.0}
                    for index in range(3)
                ],
                "rationale": "Keep the rule-supported level.",
                "evidence_ids": evidence_ids,
            },
        )
        return check_response(
            Namespace(
                prepared_json=str(prepared_path),
                response_json=str(response_path),
                window_index=0,
                round=1,
                out=None,
            )
        )

    unsupported = run_check([])
    assert unsupported["accepted"] is False
    assert {item["id"] for item in unsupported["detector_report"]["hard_violations"]} == {
        "mean_absolute_delta"
    }

    supported = run_check(["rule-1"])
    assert supported["accepted"] is True
    assert supported["supported_evidence_ids"] == ["rule-1"]
