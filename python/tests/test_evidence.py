from __future__ import annotations

import json
from pathlib import Path

from adacast_ml.evidence import EVIDENCE_CANDIDATE_BOUNDARIES, load_data_analysis_knowledge


def test_data_analysis_is_scoped_knowledge_without_absorbing_model_experience(tmp_path: Path) -> None:
    task = {"target_col": "load", "time_col": "date"}
    task_path = tmp_path / ".forecast" / "task.json"
    reports = task_path.parent / "reports"
    reports.mkdir(parents=True)
    (reports / "quantitative.json").write_text(
        json.dumps(
            {
                "target_col": "load",
                "time_col": "date",
                "analysis_scope": {"target_scope": "training_split_only"},
                "analysis": {
                    "trend": {"direction": "upward", "slope": 0.5},
                    "stationarity": {"overall": "stationary"},
                },
                "analyzer_version": "1.3.0",
            }
        ),
        encoding="utf-8",
    )
    (reports / "pre-forecast.md").write_text(
        "Dataset has a daily cycle. Model A may perform best, but that requires validation.",
        encoding="utf-8",
    )

    knowledge = load_data_analysis_knowledge(task, task_path)

    assert knowledge["epistemic_status"] == "empirical_data_knowledge"
    assert knowledge["quantitative_analysis"]["analysis"]["trend"]["slope"] == 0.5
    assert "model ranking" in knowledge["qualitative_analysis"]["usage_constraint"]
    assert "model residual/performance" in EVIDENCE_CANDIDATE_BOUNDARIES["knowledge"]


def test_quantitative_analysis_for_another_task_is_not_loaded_as_knowledge(tmp_path: Path) -> None:
    task_path = tmp_path / ".forecast" / "task.json"
    reports = task_path.parent / "reports"
    reports.mkdir(parents=True)
    (reports / "quantitative.json").write_text(
        json.dumps({"target_col": "other", "time_col": "date", "analysis": {"trend": "up"}}),
        encoding="utf-8",
    )

    knowledge = load_data_analysis_knowledge(
        {"target_col": "load", "time_col": "date"},
        task_path,
    )

    assert knowledge["quantitative_analysis"] is None
