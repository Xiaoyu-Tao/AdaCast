"""Evaluate adjusted predictions separately from raw run eval.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adacast_ml.utils.metrics import mae, mse, wape


MSE_EQUALITY_ULPS = 32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adjusted AdaCast predictions")
    parser.add_argument("--adjusted-predictions-csv", required=True)
    parser.add_argument("--out", required=True, help="Path to adjusted_eval.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--split", choices=["fit", "val", "test"], required=True)
    parser.add_argument("--candidate-json", help="Candidate JSON used to identify a frozen expert-prior mirror")
    parser.add_argument("--raw-eval-json", help="Optional raw eval.json baseline")
    parser.add_argument("--train-scale-json", help="Optional train_scale.json path for MASE")
    parser.add_argument("--policy-json", help="Optional policy.json for validation acceptance")
    parser.add_argument(
        "--apply-results-json",
        help="Optional apply-results.json used to bind trigger metadata to this evaluation",
    )
    return parser.parse_args(argv)


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None


def finite_metric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def metric_delta(adjusted: dict[str, float | None], raw: dict[str, Any] | None) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    raw_metrics = raw or {}
    for key in ("mse", "mae", "wape", "mase"):
        raw_value = finite_metric(raw_metrics.get(key))
        adjusted_value = finite_metric(adjusted.get(key))
        result[key] = None if raw_value is None or adjusted_value is None else adjusted_value - raw_value
    return result


def application_summary(
    payload: dict[str, Any] | None,
    *,
    run_id: str,
    candidate_id: str,
    split: str,
    predictions: pd.DataFrame,
) -> dict[str, Any] | None:
    if payload is None:
        return None

    if (
        payload.get("source_run_id") != run_id
        or payload.get("candidate_id") != candidate_id
        or payload.get("split") != split
    ):
        raise ValueError("apply-results.json identity does not match this adjusted evaluation")

    row_count = int(payload.get("row_count", -1))
    if row_count != len(predictions):
        raise ValueError("apply-results.json row_count does not match adjusted predictions")

    applied_rules = payload.get("applied_rules", [])
    if not isinstance(applied_rules, list):
        raise ValueError("apply-results.json applied_rules must be a list")
    rule_matches = [int(rule.get("matched_rows", 0)) for rule in applied_rules]
    if any(value < 0 for value in rule_matches):
        raise ValueError("apply-results.json matched_rows cannot be negative")

    matched_rows = int(payload.get("matched_rows", sum(rule_matches)))
    triggered_rule_count = int(
        payload.get("triggered_rule_count", sum(value > 0 for value in rule_matches))
    )
    if matched_rows < 0 or matched_rows > row_count:
        raise ValueError("apply-results.json matched_rows is outside the evaluated row range")
    if triggered_rule_count != sum(value > 0 for value in rule_matches):
        raise ValueError("apply-results.json triggered_rule_count is inconsistent with applied_rules")
    if triggered_rule_count > 0 and matched_rows == 0:
        raise ValueError("apply-results.json matched_rows is inconsistent with triggered rules")
    derived_zero_trigger = triggered_rule_count == 0 and all(value == 0 for value in rule_matches)
    zero_trigger = bool(payload.get("zero_trigger", derived_zero_trigger))
    if zero_trigger != derived_zero_trigger or (zero_trigger and matched_rows != 0):
        raise ValueError("apply-results.json zero_trigger is inconsistent with applied rule matches")

    if "raw_predicted" not in predictions.columns:
        raise ValueError("adjusted_predictions.csv missing raw_predicted audit column")
    predicted = predictions["predicted"].to_numpy(dtype=np.float64)
    raw_predicted = predictions["raw_predicted"].to_numpy(dtype=np.float64)
    observed_changed_rows = int(np.count_nonzero(predicted != raw_predicted))
    changed_rows = int(payload.get("changed_rows", observed_changed_rows))
    if changed_rows < 0 or changed_rows > row_count:
        raise ValueError("apply-results.json changed_rows is outside the evaluated row range")
    if changed_rows != observed_changed_rows:
        raise ValueError("apply-results.json changed_rows does not match adjusted predictions")
    if zero_trigger and changed_rows != 0:
        raise ValueError("A zero-trigger candidate must preserve every raw prediction exactly")

    return {
        "matched_rows": matched_rows,
        "changed_rows": changed_rows,
        "triggered_rule_count": triggered_rule_count,
        "zero_trigger": zero_trigger,
        "row_count": row_count,
        "metrics_reused_from_raw": False,
    }


def acceptance_decision(
    *,
    split: str,
    adjusted: dict[str, float | None],
    raw: dict[str, Any] | None,
    policy: dict[str, Any] | None,
    application: dict[str, Any] | None = None,
    force_accept_expert_prior: bool = False,
) -> dict[str, Any] | None:
    if split != "val" or raw is None:
        return None

    policy = policy or {}
    primary_metric = str(policy.get("primary_metric", "mse")).strip().lower()
    if primary_metric != "mse":
        raise ValueError(
            f"Unsupported adjustment primary_metric {primary_metric!r}; only 'mse' is implemented"
        )

    raw_mse = finite_metric(raw.get("mse"))
    adjusted_mse = finite_metric(adjusted.get("mse"))
    if raw_mse is None or adjusted_mse is None or raw_mse < 0:
        return {
            "accepted": False,
            "reason": "Raw MSE baseline is unavailable or negative",
            "checks": {},
        }

    configured_min_gain = float(policy.get("min_relative_mse_gain", -1.0))
    min_gain = -1.0
    max_mae_regression = float(policy.get("max_mae_regression_ratio", 0.005))
    max_wape_regression = float(policy.get("max_wape_regression_ratio", 0.005))
    checks: dict[str, Any] = {
        "min_relative_mse_gain": min_gain,
        "force_accept_expert_prior": force_accept_expert_prior,
    }
    if configured_min_gain != min_gain:
        checks["legacy_configured_min_relative_mse_gain_ignored"] = configured_min_gain
    if application is not None:
        row_count = application.get("row_count")
        coverage_ratio = (
            application["matched_rows"] / row_count
            if isinstance(row_count, int) and row_count > 0
            else None
        )
        checks.update(
            {
                "matched_rows": application["matched_rows"],
                "changed_rows": application["changed_rows"],
                "triggered_rule_count": application["triggered_rule_count"],
                "zero_trigger": application["zero_trigger"],
                **(
                    {
                        "coverage_ratio": coverage_ratio,
                    }
                    if coverage_ratio is not None
                    else {}
                ),
            }
        )

    if raw_mse == 0:
        relative_mse_gain = 0.0 if adjusted_mse == 0 else None
        checks["mse_equality_tolerance"] = 0.0
        checks["relative_mse_gain"] = relative_mse_gain
        accepted = force_accept_expert_prior or adjusted_mse == 0
    else:
        mse_delta = raw_mse - adjusted_mse
        mse_equality_tolerance = float(
            MSE_EQUALITY_ULPS
            * np.spacing(max(abs(raw_mse), abs(adjusted_mse)))
        )
        checks["mse_equality_tolerance"] = mse_equality_tolerance
        checks["raw_minus_adjusted_mse"] = mse_delta
        relative_mse_gain = (
            0.0
            if abs(mse_delta) <= mse_equality_tolerance
            else mse_delta / raw_mse
        )
        checks["relative_mse_gain"] = relative_mse_gain
        accepted = relative_mse_gain >= min_gain
    reasons = []
    if not accepted:
        reasons.append("relative MSE gain below threshold")

    for metric_name, limit in (("mae", max_mae_regression), ("wape", max_wape_regression)):
        raw_value = finite_metric(raw.get(metric_name))
        adjusted_value = finite_metric(adjusted.get(metric_name))
        if raw_value is None or raw_value <= 0 or adjusted_value is None:
            continue
        regression = (adjusted_value - raw_value) / raw_value
        checks[f"{metric_name}_regression_ratio"] = regression
        checks[f"max_{metric_name}_regression_ratio"] = limit
        if regression > limit and not force_accept_expert_prior:
            accepted = False
            reasons.append(f"{metric_name} regression above threshold")

    if force_accept_expert_prior:
        accepted = True
        reasons = []
        checks["metric_guardrails_bypassed_for_expert_prior"] = True

    accepted_reason = "accepted"
    if force_accept_expert_prior:
        accepted_reason = "accepted: expert-prior mirror retained under the frozen -100% validation policy"
    elif accepted and application is not None and application["zero_trigger"]:
        accepted_reason = "accepted: candidate did not trigger and preserved the baseline exactly"

    return {
        "accepted": accepted,
        "reason": accepted_reason if accepted else "; ".join(reasons),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    predictions = pd.read_csv(args.adjusted_predictions_csv)
    required = {"actual", "predicted"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"adjusted_predictions.csv missing required columns: {sorted(missing)}")

    pred_values = predictions["predicted"].to_numpy(dtype=np.float64)
    true_values = predictions["actual"].to_numpy(dtype=np.float64)
    train_scale_payload = load_json(args.train_scale_json)
    train_scale = (
        finite_metric(train_scale_payload.get("train_scale"))
        if train_scale_payload
        else None
    )
    adjusted_mae = mae(pred_values, true_values)

    adjusted_metrics: dict[str, float | None] = {
        "mse": mse(pred_values, true_values),
        "mae": adjusted_mae,
        "wape": wape(pred_values, true_values),
        "mase": adjusted_mae / train_scale
        if train_scale is not None and train_scale > 0
        else None,
    }
    raw_eval = load_json(args.raw_eval_json)
    raw_metrics = raw_eval.get("metrics") if raw_eval else None
    apply_payload = load_json(args.apply_results_json)
    apply_summary = application_summary(
        apply_payload,
        run_id=args.run_id,
        candidate_id=args.candidate_id,
        split=args.split,
        predictions=predictions,
    )
    if apply_summary is not None and apply_summary["changed_rows"] == 0 and raw_metrics is not None:
        # The apply artifact proves that every prediction is byte-for-byte the
        # raw baseline, whether the rules were dormant or matched with a no-op.
        # Reuse frozen raw metrics so serialization or metric recomputation
        # cannot manufacture a false negative gain.
        for key in ("mse", "mae", "wape", "mase"):
            raw_value = finite_metric(raw_metrics.get(key))
            if raw_value is not None:
                adjusted_metrics[key] = raw_value
        apply_summary["metrics_reused_from_raw"] = True
    policy = load_json(args.policy_json)
    candidate = load_json(args.candidate_json)
    if candidate is not None and not isinstance(candidate, dict):
        raise ValueError("candidate.json must contain a JSON object")
    if candidate is not None and str(candidate.get("candidate_id", "")) != args.candidate_id:
        raise ValueError("candidate.json identity does not match this adjusted evaluation")
    candidate_provenance = (
        candidate.get("llm_provenance") if isinstance(candidate, dict) else None
    )
    force_accept_expert_prior = bool(
        candidate is not None
        and (
            candidate.get("origin") == "user_prior_mirror"
            or (
                isinstance(candidate_provenance, dict)
                and candidate_provenance.get("user_prior_mirror") is True
            )
        )
    )
    decision = acceptance_decision(
        split=args.split,
        adjusted=adjusted_metrics,
        raw=raw_metrics,
        policy=policy,
        application=apply_summary,
        force_accept_expert_prior=force_accept_expert_prior,
    )

    output = {
        "run_id": args.run_id,
        "candidate_id": args.candidate_id,
        "split": args.split,
        "raw_eval_path": args.raw_eval_json,
        "adjusted_predictions_path": args.adjusted_predictions_csv,
        "raw_metrics": raw_metrics,
        "adjusted_metrics": adjusted_metrics,
        "delta_vs_raw": metric_delta(adjusted_metrics, raw_metrics),
        **({"application_summary": apply_summary} if apply_summary is not None else {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **({"acceptance_decision": decision} if decision is not None else {}),
        **({"candidate_origin": candidate.get("origin")} if candidate is not None else {}),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    json.dump(output, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
