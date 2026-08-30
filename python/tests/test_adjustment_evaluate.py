"""Tests for aggregate validation acceptance of adjustment candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adacast_ml.adjustment.evaluate import acceptance_decision, main


def test_acceptance_uses_overall_mse_with_aggregate_guardrails() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": 1.8, "mae": 1.004, "wape": 0.49},
        policy={
            "min_relative_mse_gain": -1.0,
            "max_mae_regression_ratio": 0.005,
            "max_wape_regression_ratio": 0.005,
        },
    )

    assert decision is not None
    assert decision["accepted"] is True
    assert decision["checks"]["relative_mse_gain"] == pytest.approx(0.1)


def test_acceptance_rejects_overall_mse_gain_with_excessive_mae_regression() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": 1.8, "mae": 1.02, "wape": 0.49},
        policy={
            "min_relative_mse_gain": -1.0,
            "max_mae_regression_ratio": 0.005,
            "max_wape_regression_ratio": 0.005,
        },
    )

    assert decision is not None
    assert decision["accepted"] is False
    assert "mae regression above threshold" in decision["reason"]


def test_acceptance_does_not_limit_candidate_coverage() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": 1.8, "mae": 0.99, "wape": 0.49},
        policy={"max_candidate_coverage_ratio": 0.3},
        application={
            "matched_rows": 31,
            "changed_rows": 31,
            "triggered_rule_count": 1,
            "zero_trigger": False,
            "row_count": 100,
        },
    )

    assert decision is not None
    assert decision["accepted"] is True
    assert decision["checks"]["coverage_ratio"] == pytest.approx(0.31)
    assert "max_candidate_coverage_ratio" not in decision["checks"]


def test_expert_prior_is_retained_despite_metric_regression() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": 20.0, "mae": 5.0, "wape": 2.5},
        policy={
            "min_relative_mse_gain": -1.0,
            "max_mae_regression_ratio": 0.005,
            "max_wape_regression_ratio": 0.005,
        },
        application={
            "matched_rows": 10,
            "changed_rows": 10,
            "triggered_rule_count": 1,
            "zero_trigger": False,
        },
        force_accept_expert_prior=True,
    )

    assert decision is not None
    assert decision["accepted"] is True
    assert decision["checks"]["metric_guardrails_bypassed_for_expert_prior"] is True
    assert "expert-prior mirror retained" in decision["reason"]


def test_default_acceptance_threshold_accepts_any_nonnegative_mse_gain() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": 1.999, "mae": 1.0, "wape": 0.5},
        policy=None,
    )

    assert decision is not None
    assert decision["accepted"] is True
    assert decision["checks"]["min_relative_mse_gain"] == -1.0
    assert decision["checks"]["relative_mse_gain"] > 0


def test_default_acceptance_threshold_accepts_zero_mse_gain() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        policy=None,
    )

    assert decision is not None
    assert decision["accepted"] is True
    assert decision["checks"]["min_relative_mse_gain"] == -1.0
    assert decision["checks"]["relative_mse_gain"] == 0.0


def test_legacy_positive_acceptance_threshold_is_normalized_to_minus_one() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        policy={"min_relative_mse_gain": 0.05},
        application={
            "matched_rows": 0,
            "changed_rows": 0,
            "triggered_rule_count": 0,
            "zero_trigger": True,
        },
    )

    assert decision is not None
    assert decision["accepted"] is True
    assert decision["checks"]["min_relative_mse_gain"] == -1.0
    assert decision["checks"]["legacy_configured_min_relative_mse_gain_ignored"] == 0.05


def test_default_acceptance_accepts_unchanged_perfect_baseline() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 0.0, "mae": 0.0, "wape": 0.0},
        adjusted={"mse": 0.0, "mae": 0.0, "wape": 0.0},
        policy=None,
        application={
            "matched_rows": 0,
            "changed_rows": 0,
            "triggered_rule_count": 0,
            "zero_trigger": True,
        },
    )

    assert decision is not None
    assert decision["accepted"] is True
    assert decision["checks"]["relative_mse_gain"] == 0.0
    assert decision["checks"]["zero_trigger"] is True


def test_default_acceptance_threshold_rejects_mse_gain_below_minus_one() -> None:
    decision = acceptance_decision(
        split="val",
        raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": 4.1, "mae": 1.0, "wape": 0.5},
        policy=None,
    )

    assert decision is not None
    assert decision["accepted"] is False
    assert decision["checks"]["min_relative_mse_gain"] == -1.0
    assert decision["checks"]["relative_mse_gain"] < -1.0
    assert "relative MSE gain below threshold" in decision["reason"]


def test_default_acceptance_treats_only_machine_precision_mse_drift_as_equal() -> None:
    raw_mse = 2.0
    next_float = float(np.nextafter(raw_mse, np.inf))
    decision = acceptance_decision(
        split="val",
        raw={"mse": raw_mse, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": next_float, "mae": 1.0, "wape": 0.5},
        policy=None,
        application={
            "matched_rows": 1,
            "changed_rows": 1,
            "triggered_rule_count": 1,
            "zero_trigger": False,
        },
    )

    assert decision is not None
    assert decision["accepted"] is True
    assert decision["checks"]["raw_minus_adjusted_mse"] < 0
    assert decision["checks"]["relative_mse_gain"] == 0.0

    genuinely_worse = acceptance_decision(
        split="val",
        raw={"mse": raw_mse, "mae": 1.0, "wape": 0.5},
        adjusted={"mse": raw_mse + 1e-10, "mae": 1.0, "wape": 0.5},
        policy=None,
        application={
            "matched_rows": 1,
            "changed_rows": 1,
            "triggered_rule_count": 1,
            "zero_trigger": False,
        },
    )
    assert genuinely_worse is not None
    assert genuinely_worse["accepted"] is True
    assert genuinely_worse["checks"]["relative_mse_gain"] < 0

    tiny_but_real_regression = acceptance_decision(
        split="val",
        raw={"mse": 1e-10, "mae": 1e-5, "wape": 1e-5},
        adjusted={"mse": 1e-10 + 1e-15, "mae": 1e-5, "wape": 1e-5},
        policy=None,
        application={
            "matched_rows": 1,
            "changed_rows": 1,
            "triggered_rule_count": 1,
            "zero_trigger": False,
        },
    )
    assert tiny_but_real_regression is not None
    assert tiny_but_real_regression["accepted"] is True
    assert tiny_but_real_regression["checks"]["relative_mse_gain"] < 0


def test_test_split_never_creates_a_new_acceptance_decision() -> None:
    assert (
        acceptance_decision(
            split="test",
            raw={"mse": 2.0, "mae": 1.0, "wape": 0.5},
            adjusted={"mse": 1.0, "mae": 0.5, "wape": 0.25},
            policy={},
        )
        is None
    )


def test_acceptance_rejects_an_unimplemented_primary_metric() -> None:
    with pytest.raises(ValueError, match="only 'mse' is implemented"):
        acceptance_decision(
            split="val",
            raw={"mse": 2.0, "mae": 1.0},
            adjusted={"mse": 1.0, "mae": 0.5},
            policy={"primary_metric": "mae"},
        )


def test_adjusted_mase_uses_the_frozen_training_scale(tmp_path: Path) -> None:
    predictions_path = tmp_path / "adjusted_predictions.csv"
    output_path = tmp_path / "adjusted_eval.json"
    scale_path = tmp_path / "train_scale.json"
    pd.DataFrame(
        {
            "actual": [1.0, 3.0],
            "predicted": [2.0, 1.0],
        }
    ).to_csv(predictions_path, index=False)
    scale_path.write_text(json.dumps({"train_scale": 2.0}), encoding="utf-8")

    main(
        [
            "--adjusted-predictions-csv",
            str(predictions_path),
            "--out",
            str(output_path),
            "--run-id",
            "run",
            "--candidate-id",
            "candidate",
            "--split",
            "fit",
            "--train-scale-json",
            str(scale_path),
        ]
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["adjusted_metrics"]["mae"] == pytest.approx(1.5)
    assert result["adjusted_metrics"]["mase"] == pytest.approx(0.75)


def test_zero_trigger_evaluation_reuses_exact_raw_metrics_and_is_accepted(tmp_path: Path) -> None:
    predictions_path = tmp_path / "adjusted_predictions.csv"
    output_path = tmp_path / "adjusted_eval.json"
    raw_eval_path = tmp_path / "eval.json"
    apply_results_path = tmp_path / "apply-results.json"
    pd.DataFrame(
        {
            "actual": [1.0, 3.0],
            "raw_predicted": [2.0, 1.0],
            "predicted": [2.0, 1.0],
        }
    ).to_csv(predictions_path, index=False)
    raw_metrics = {"mse": 2.500000000000001, "mae": 1.5, "wape": 0.375, "mase": None}
    raw_eval_path.write_text(json.dumps({"metrics": raw_metrics}), encoding="utf-8")
    apply_results_path.write_text(
        json.dumps(
            {
                "source_run_id": "run",
                "candidate_id": "candidate",
                "split": "val",
                "row_count": 2,
                "applied_rules": [
                    {"rule_id": "rare-event", "matched_rows": 0, "clipped_rows": 0}
                ],
                "matched_rows": 0,
                "changed_rows": 0,
                "triggered_rule_count": 0,
                "zero_trigger": True,
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "--adjusted-predictions-csv",
            str(predictions_path),
            "--out",
            str(output_path),
            "--run-id",
            "run",
            "--candidate-id",
            "candidate",
            "--split",
            "val",
            "--raw-eval-json",
            str(raw_eval_path),
            "--apply-results-json",
            str(apply_results_path),
        ]
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["application_summary"] == {
        "matched_rows": 0,
        "changed_rows": 0,
        "triggered_rule_count": 0,
        "zero_trigger": True,
        "row_count": 2,
        "metrics_reused_from_raw": True,
    }
    assert result["adjusted_metrics"] == raw_metrics
    assert result["delta_vs_raw"] == {"mse": 0.0, "mae": 0.0, "wape": 0.0, "mase": None}
    assert result["acceptance_decision"]["accepted"] is True
    assert "did not trigger" in result["acceptance_decision"]["reason"]


def test_zero_trigger_evaluation_rejects_changed_predictions(tmp_path: Path) -> None:
    predictions_path = tmp_path / "adjusted_predictions.csv"
    output_path = tmp_path / "adjusted_eval.json"
    apply_results_path = tmp_path / "apply-results.json"
    pd.DataFrame(
        {
            "actual": [1.0],
            "raw_predicted": [2.0],
            "predicted": [2.1],
        }
    ).to_csv(predictions_path, index=False)
    apply_results_path.write_text(
        json.dumps(
            {
                "source_run_id": "run",
                "candidate_id": "candidate",
                "split": "val",
                "row_count": 1,
                "applied_rules": [{"rule_id": "r1", "matched_rows": 0}],
                "matched_rows": 0,
                "changed_rows": 0,
                "triggered_rule_count": 0,
                "zero_trigger": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed_rows does not match"):
        main(
            [
                "--adjusted-predictions-csv",
                str(predictions_path),
                "--out",
                str(output_path),
                "--run-id",
                "run",
                "--candidate-id",
                "candidate",
                "--split",
                "val",
                "--apply-results-json",
                str(apply_results_path),
            ]
        )
