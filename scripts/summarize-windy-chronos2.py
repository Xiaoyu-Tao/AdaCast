"""Verify and summarize the completed windy Chronos-2 -> AdaCast experiment."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "workspaces/windy-chronos2-ms-visible-20261103")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/windy-chronos2-ms-visible")
    args = parser.parse_args()
    bundle = args.workspace / ".forecast/best-model"
    manifest = load_json(bundle / "manifest.json")
    prior = load_json(args.output / "manifest.json")
    assert load_json(bundle / "task.json")["dataset_sha256"] == prior["dataset_sha256"]
    assert manifest["files"]["model/prior.csv"] == prior["prior_sha256"]
    task = load_json(args.workspace / ".forecast/task.json")
    source = pd.read_csv(args.workspace / ".forecast/inputs/dataset.csv")
    train_values = source.real_power.iloc[:23594].to_numpy(dtype=np.float64)
    mase_scale = float(np.mean(np.abs(np.diff(train_values))))
    scale = load_json(bundle / "model/model.json")["scale"]
    raw = pd.read_csv(bundle / "predictions/raw.csv")
    final = pd.read_csv(bundle / "predictions/final.csv")
    expected = pd.date_range("2024-11-12", "2024-12-31 23:45", freq="15min")
    for frame in [raw, final]:
        assert pd.DatetimeIndex(pd.to_datetime(frame.timestamp)).equals(expected)
        assert len(frame) == 4800 and frame.window_index.nunique() == 50
        assert frame.groupby("window_index").size().eq(96).all()
        assert frame.groupby("window_index").horizon_index.apply(list).apply(lambda x: x == list(range(96))).all()
    np.testing.assert_allclose(raw.actual_original, final.actual_original, rtol=0, atol=1e-10)
    original_actual = source.real_power.iloc[30336:].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(raw.actual_original, original_actual, rtol=0, atol=1e-10)
    baseline = pd.read_csv(args.output / "predictions.csv")
    baseline = baseline.loc[baseline.split == "test"]
    np.testing.assert_allclose(raw.predicted_original, baseline.predicted_original, rtol=1e-6, atol=1e-5)
    comparison = pd.DataFrame({"timestamp": expected, "window_index": raw.window_index,
                               "horizon_index": raw.horizon_index, "actual": original_actual,
                               "chronos2": raw.predicted_original.to_numpy(dtype=np.float64)})
    adjusted_file = bundle / "predictions/adjusted.csv"
    if adjusted_file.exists():
        adjusted = pd.read_csv(adjusted_file)
        assert pd.DatetimeIndex(pd.to_datetime(adjusted.timestamp)).equals(expected)
        comparison["adjusted"] = adjusted.adjusted_predicted_original
    final_column = "final_predicted_original" if "final_predicted_original" in final else "adjusted_predicted_original" if "adjusted_predicted_original" in final else "predicted_original"
    comparison["adacast_final"] = final[final_column].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(comparison.adacast_final, final.predicted.to_numpy()*scale["std"]+scale["mean"], rtol=1e-7, atol=1e-5)
    models = [c for c in ["chronos2", "adjusted", "adacast_final"] if c in comparison]

    def metrics(frame, column):
        actual = frame.actual.to_numpy(dtype=np.float64)
        pred = frame[column].to_numpy(dtype=np.float64)
        error = pred-actual
        absolute_actual_sum = float(np.sum(abs(actual)))
        return {"mse": float(np.mean(error**2)), "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(abs(error))), "wape": float(np.sum(abs(error))/absolute_actual_sum) if absolute_actual_sum > 0 else None,
                "mase": float(np.mean(abs(error))/mase_scale)}

    scores = {model: metrics(comparison, model) for model in models}
    daily = []
    for window_index, day in comparison.groupby("window_index", sort=True):
        for model in models:
            daily.append({"window_index": int(window_index), "date": str(day.timestamp.iloc[0].date()),
                          "model": model, "points": len(day), **metrics(day, model)})
    daily = pd.DataFrame(daily)
    comparison.to_csv(args.output / "test-predictions.csv", index=False)
    daily.to_csv(args.output / "daily-test-metrics.csv", index=False)
    daily_mse = daily.pivot(index="date", columns="model", values="mse")
    gain = daily_mse.chronos2-daily_mse.adacast_final
    largest_gain_day = gain.idxmax()
    excluding_largest_gain = daily_mse.drop(index=largest_gain_day)
    final_gain = (scores["chronos2"]["mse"]-scores["adacast_final"]["mse"])/scores["chronos2"]["mse"]
    audit = load_json(bundle / "audit/decisions.json")
    standardized_scores = {"chronos2": manifest["test"]["metrics"], "adacast_final": manifest["final_metrics"]}
    if "adjusted" in models:
        application = next(record for record in audit["domain"]["adjustment"]["test_application_records"]
                           if record["test_run_id"] == manifest["test"]["run_id"] and record["decision"] == "apply")
        relative = application["adjusted_eval_path"]
        assert relative.startswith(".forecast/")
        adjusted_evaluation = load_json(bundle / "audit" / relative.removeprefix(".forecast/"))
        standardized_scores["adjusted"] = adjusted_evaluation["adjusted_metrics"]
    overrides = load_json(bundle / "audit/llm-calls/policy-overrides.json")
    with sqlite3.connect(f"file:{args.workspace / '.forecast/runtime/state.sqlite'}?mode=ro", uri=True) as database:
        usage_by_status = [{"status": status, "calls": count, "provider_reported_tokens": reported,
                            "reserved_tokens_for_unknown_usage": reserved}
                           for status, count, reported, reserved in database.execute(
                               "SELECT status, COUNT(*), COALESCE(SUM(used),0), "
                               "COALESCE(SUM(CASE WHEN used IS NULL THEN reserved ELSE 0 END),0) "
                               "FROM calls GROUP BY status")]
    request_failures = {"provider_failed", "schema_failed", "response_empty", "output_truncated"}
    terminals = [load_json(file) for file in (bundle / "audit/reflection/test" / manifest["test"]["run_id"]).glob("*/corrections/window-*.json")]
    assert sorted(record["window_index"] for record in terminals) == list(range(50))
    for record in terminals:
        loop = record["evidence_loop"]
        state = record["critic_prediction_state"]
        assert record["accepted"] and loop["termination_reason"] not in request_failures
        if state["evidence_gap"] == 1 and loop["termination_reason"] in {"evidence_exhausted", "no_valid_transition"}:
            committed = {e["evidence_candidate_id"] for e in state["evidences"]}
            assert not any(r.get("reason") in request_failures and r.get("candidate_id") not in committed
                           for r in loop["rejected_transitions"])
    summary = {"protocol": {"training_points": 23594, "history_only_bridge_points": 22,
                             "validation_tasks": 70, "validation_fit_tasks": 35, "validation_holdout_tasks": 35,
                             "test_tasks": 50, "test_points": 4800, "input_len": 96, "prediction_len": 96,
                             "forecast_gap": 0, "window_stride": 96, "future_covariates": len(prior["future_covariates"]),
                             "training_end": task["train_end_timestamp"], "validation_start": task["validation_start_timestamp"],
                             "test_start": str(expected[0]), "test_end": str(expected[-1])},
               "chronos2": {"model": prior["model"], "revision": prior["revision"], "mode": prior["mode"],
                            "full_validation_metrics": prior["metrics"]["validation"]},
               "full_test_original_scale": scores,
               "full_test_standardized_scale": standardized_scores,
               "final_mse_improvement_ratio": final_gain,
               "daily_mse_comparison": {"improved": int((gain>1e-9).sum()), "worse": int((gain < -1e-9).sum()), "unchanged": int((abs(gain)<=1e-9).sum())},
               "gain_concentration": {"largest_gain_day": str(largest_gain_day),
                                      "largest_day_share_of_net_mse_reduction": float(gain.loc[largest_gain_day]/gain.sum()) if gain.sum() != 0 else None,
                                      "excluding_largest_gain_day": {
                                          "tasks": len(excluding_largest_gain),
                                          "chronos2_mse": float(excluding_largest_gain.chronos2.mean()),
                                          "adacast_final_mse": float(excluding_largest_gain.adacast_final.mean()),
                                          "mse_improvement_ratio": float(1-excluding_largest_gain.adacast_final.mean()/excluding_largest_gain.chronos2.mean())},
                                      "scope": "Post-hoc sensitivity analysis only; the primary test retains all 50 days"},
               "final_kind": manifest["final_kind"], "llm_usage": audit["usage"],
               "llm_usage_by_status": usage_by_status,
               "llm_usage_accounting": "Provider total_tokens for returned calls; reserved token estimate for interrupted or otherwise unknown calls",
               "policy_overrides": overrides["overrides"],
               "window_reflection": {"complete_windows": len(terminals),
                                     "termination_reasons": dict(Counter(r["evidence_loop"]["termination_reason"] for r in terminals)),
                                     "unresolved_request_failures": 0},
               "bundle": str(bundle.resolve()), "checks": {"dataset_and_prior_provenance_match": True, "complete_daily_windows": True, "exact_test_timestamps": True,
                "raw_prior_reused": True, "actuals_match_source": True, "final_original_and_standardized_consistent": True,
                "all_50_windows_have_valid_terminal_records": True}}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    axes[0].plot(comparison.timestamp, comparison.actual, color="#475569", linewidth=0.8, alpha=0.8, label="Actual")
    axes[0].plot(comparison.timestamp, comparison.chronos2, color="#2563eb", linewidth=0.8, alpha=0.8, label="Chronos-2 prior")
    axes[0].plot(comparison.timestamp, comparison.adacast_final, color="#c2410c", linewidth=0.8, alpha=0.9, label="AdaCast final")
    axes[0].set(ylabel="Power (source units)", title="Windy | Full test, 50 independent daily 96 -> 96 tasks | Future weather visible")
    axes[0].legend(loc="upper right", ncols=3, frameon=False)
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].bar(pd.to_datetime(gain.index), gain.values, width=0.8,
                color=["#0f766e" if x>=0 else "#b91c1c" for x in gain.values])
    axes[1].axhline(0, color="#475569", linewidth=0.7)
    axes[1].set(ylabel="Prior MSE - final MSE", xlabel="Prediction day", title=f"Daily original-scale MSE change (positive = improvement); aggregate improvement {100*final_gain:.2f}%")
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    figure.savefig(args.output / "test-comparison.png", dpi=180)
    figure.savefig(args.output / "test-comparison.svg")
    plt.close(figure)
    print(json.dumps({"metrics": scores, "mse_improvement_ratio": final_gain, "summary": str(args.output / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
