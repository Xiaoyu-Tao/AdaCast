"""Versioned computation operations. Test inference never fits a model."""
from __future__ import annotations
import csv
import json
import math
import shutil
import statistics
from pathlib import Path
from . import __version__
from .data import Data, dump, sha_file, metrics, save_predictions, load_predictions, stamp
from .models import fit, predictor, preflight, MODELS, statistical_curve, canonical_spec


def fingerprint(data):
    task = {k: v for k, v in data.task.items() if k != "dataset_path"}
    text = json.dumps({"task": task, "dataset_sha256": sha_file(data.path)}, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def analyze(data):
    from .engine.analyzer import analyze as canonical_analyze
    keys = ("dataset_path", "target_col", "time_col", "train_ratio", "val_ratio", "test_ratio", "split_mode", "train_end_timestamp", "validation_start_timestamp", "validation_end_timestamp")
    kwargs = {k: data.task[k] for k in keys if k in data.task}
    if data.task.get("split_mode") == "timestamp":
        kwargs.update(train_ratio=data.t/len(data.y), val_ratio=(data.v-data.t)/len(data.y), test_ratio=(len(data.y)-data.v)/len(data.y))
    result = canonical_analyze(**kwargs)
    return {**data.info(), **result, "canonical_analysis": result,
            "mase_scale_standardized": data.mase_scale}


def bind_model(data, spec, directory, prior_path=None):
    directory.mkdir(parents=True, exist_ok=True)
    if prior_path:
        import pandas as pd
        from .engine.prediction_prior import validate_prior_frame
        original_frame = pd.read_csv(data.path,low_memory=False)
        prior_frame = pd.read_csv(prior_path,low_memory=False)
        validate_prior_frame(task=data.task,original=original_frame,prior=prior_frame,train_end=data.t)
        shutil.copyfile(prior_path, directory / "prior.csv")
        model = {"name": "UserPrior", "params": {}, "prior": "prior.csv",
                 "seq_len": data.task["seq_len"], "horizon": data.task.get("forecast_gap", 0)+data.task["pred_len"]}
    else:
        model = fit(data, spec["model"], spec.get("params", spec.get("hyperparams", {})), directory)
    import hashlib
    if model.get("family") == "statistical":
        is_arima = model["name"] == "ARIMA"
        filename = "arima-results.pkl" if is_arima else "statistical-results.pkl"
        if filename != model["weights"]:
            (directory/model["weights"]).rename(directory/filename)
            model["weights"] = filename
        dump(directory/"metadata.json", {"kind":"arima_results" if is_arima else "statistical_results", "format":"statsmodels_arima_results" if is_arima else "statsmodels_results_pickle", "path":"model/"+filename, "sha256":sha_file(directory/filename), "model":model["name"], "phase":"forecast", "eval_split":"val", "trained_on":"train"})
    elif model.get("family") == "deep_learning":
        dump(directory/"metadata.json", {"kind":"checkpoint", "format":"torch_state_dict", "path":"model/checkpoint.pth", "sha256":sha_file(directory/"checkpoint.pth"), "model":model["name"], "phase":"forecast", "eval_split":"val", "trained_on":"train"})
    elif model.get("family") == "foundation":
        from .engine.utils.hash import compute_spec_hash
        cfg = model["engine_config"]
        dump(directory/"foundation.json", {"kind":"foundation_model_reference", "format":"huggingface_snapshot", "model":model["name"], "repo_id":cfg["foundation_model_repo"], "revision":cfg["foundation_model_revision"], "sha256":compute_spec_hash({"repo_id":cfg["foundation_model_repo"], "revision":cfg["foundation_model_revision"]})})
    elif model["name"] == "UserPrior":
        dump(directory/"prior-source.json", {"kind":"external_prediction_prior", "format":"adacast_user_prediction_prior", "model":"UserPrior", "source_dataset_sha256":sha_file(directory/"prior.csv")})
    files = {p.name: sha_file(p) for p in directory.iterdir() if p.is_file() and p.name != "model.json"}
    model.update(version=__version__, task_fingerprint=fingerprint(data), scale={"mean": data.mean, "std": data.scale}, files=files)
    dump(directory / "model.json", model)
    return model


def load_model(data, artifact, spec):
    artifact = Path(artifact)
    model = json.loads(artifact.read_text())
    if model["version"] != __version__ or model["task_fingerprint"] != fingerprint(data):
        raise ValueError("Source model version or frozen task mismatch")
    if model["name"] != spec["model"] or model["params"] != spec.get("params", spec.get("hyperparams", {})):
        raise ValueError("Test inference spec must exactly match fitted model parameters")
    for name, expected in model["files"].items():
        if Path(name).name != name or sha_file(artifact.parent / name) != expected:
            raise ValueError("Source model artifact hash mismatch")
    return model


def experiment(data, request):
    directory = Path(request["output_dir"])
    from .catalog import canonical_model
    spec = dict(request["spec"])
    spec["model"] = "UserPrior" if str(spec["model"]).casefold()=="userprior" else canonical_model(spec["model"])
    split = request.get("split", "val")
    if split not in ("val", "adjustment_fit", "test"):
        raise ValueError("Invalid evaluation split")
    directory.mkdir(parents=True, exist_ok=True)
    source = request.get("source_artifact")
    if source:
        if request.get("source_sha256") != sha_file(source):
            raise ValueError("Model source hash does not match expected provenance")
        model = load_model(data, source, spec)
        artifact = Path(source)
    else:
        if split != "val":
            raise ValueError("Fitting is only allowed for a validation experiment")
        model = bind_model(data, spec, directory / "model", request.get("prior_path"))
        artifact = directory / "model/model.json"
    statistical = model.get("family") == "statistical"
    origins = list(data.origins(split))
    prior_anchor = None
    if model["name"] == "UserPrior":
        import pandas as pd
        from .engine.prediction_prior import scaled_matrices, infer_prediction_target_anchor, prediction_window_starts
        original_frame = pd.read_csv(data.path,low_memory=False)
        prior_frame = pd.read_csv(artifact.parent/model["prior"],low_memory=False)
        _, prior_scaled, _, target_pos = scaled_matrices(original_frame,prior_frame,data.task,data.t)
        values = prior_scaled[:,target_pos].tolist()
        prior_anchor = infer_prediction_target_anchor(original=original_frame,prior=prior_frame,target_col=data.task["target_col"],train_end=data.t,search_start=data.v if split == "test" else data.t,target_start_timestamp=data.task.get("prediction_target_start_timestamp"),time_col=data.task["time_col"])
        starts = prediction_window_starts(split,train_end=data.t,adjustment_fit_end=data.a,val_end=data.v,row_count=len(data.y),seq_len=data.task["seq_len"],forecast_gap=data.task.get("forecast_gap",0),pred_len=data.task["pred_len"],window_stride=data.task.get("window_stride",1),prediction_target_anchor=prior_anchor)
        origins = [start+data.task["seq_len"] for start in starts]
        predict = None
    elif statistical:
        values = statistical_curve(model, artifact.parent, data.boundaries()[split][1]-data.t)
        predict = None
    else:
        predict = predictor(model, artifact.parent, data.y[:data.t])
        values = None
    time_marks = None
    if model.get("engine_config"):
        import numpy as np
        import pandas as pd
        from .engine.utils.timefeatures import time_features
        time_marks = time_features(pd.DatetimeIndex(pd.to_datetime(data.times)), freq=model["engine_config"]["freq"]).transpose().astype(np.float32)
    points, gap_points = [], []
    gap, horizon = data.task.get("forecast_gap", 0), data.task["pred_len"]
    for window, origin in enumerate(origins):
        if values is not None:
            start = origin-data.t if statistical else origin
            predicted = values[start:start+gap+horizon]
        else:
            if time_marks is not None:
                predict.marks = (time_marks[origin-data.task["seq_len"]:origin], time_marks[origin-data.task.get("label_len", 0):origin+gap+horizon])
                history = data.features[origin-data.task["seq_len"]:origin]
            else:
                history = data.y[origin-data.task["seq_len"]:origin]
            predicted = predict(history)
        if len(predicted) != gap+horizon or not all(math.isfinite(v) for v in predicted):
            raise ValueError("Model returned invalid prediction length or values")
        if split == "test" and gap:
            gap_points.extend({"timestamp":data.times[origin+h], "predicted":float(predicted[h]), "prediction_layout":"forecast_gap", "window_index":window, "gap_index":h} for h in range(gap))
        for h in range(horizon):
            index = origin+gap+h
            value = float(predicted[gap+h])
            points.append({"timestamp": data.times[index], "window_index": window, "horizon_index": h,
                           "origin": origin, "row_index": index, "actual": data.y[index], "predicted": value,
                           "prediction_layout": "sliding_window", "actual_original": data.values[index], "predicted_original": value*data.scale+data.mean})
    pred_file = directory / "predictions.csv"
    save_predictions(pred_file, points)
    result = {"run_id": request["run_id"], "model": model["name"], "spec": spec, "split": split,
              "metrics": metrics(points, data.mase_scale), "artifact": str(artifact.resolve()),
              "artifact_sha256": sha_file(artifact), "predictions": str(pred_file.resolve()),
              "status": "success", "windows": len(origins), "points": len(points)}
    if request.get("source_run_id"):
        result["source_run_id"] = request["source_run_id"]
    # Standard domain artifacts retain the original validation/reuse/evaluator contracts.
    import numpy as np
    from .engine.evaluator import evaluate
    persisted_spec = {**canonical_spec(data, model["name"], spec.get("params", spec.get("hyperparams", {}))), **spec,
                      "phase":"post-forecast" if split == "test" else "forecast", "eval_split":split,
                      "adjustment_fit_holdout_ratio":.5}
    if prior_anchor is not None:
        persisted_spec.update(input_contract_version=5,source_type="user_prediction_prior",prediction_grid_target_anchor_index=prior_anchor)
    dump(directory/"spec.json", persisted_spec)
    np.save(directory/"actual.npy", np.asarray([p["actual"] for p in points],dtype=np.float64),allow_pickle=False)
    np.save(directory/"pred.npy", np.asarray([p["predicted"] for p in points],dtype=np.float64),allow_pickle=False)
    dump(directory/"train_scale.json", {"train_scale":data.mase_scale})
    if gap_points:
        save_predictions(directory/"gap_predictions.csv",gap_points)
    family = model.get("family")
    if family == "statistical":
        identity = {"kind":"statsmodels_arima_results" if model["name"] == "ARIMA" else "statsmodels_statistical_results", "sha256":sha_file(artifact.parent/model["weights"])}
    elif family == "foundation":
        meta = json.loads((artifact.parent/"foundation.json").read_text())
        identity = {"kind":"foundation_model_reference", "sha256":meta["sha256"], "foundation_revision":meta["revision"]}
    elif family == "deep_learning":
        identity = {"kind":"torch_state_dict", "sha256":sha_file(artifact.parent/model["weights"])}
    elif model["name"] == "UserPrior":
        identity = {"kind":"external_prediction_prior", "sha256":sha_file(artifact.parent/"prior-source.json")}
    else:
        identity = {"kind":"baseline_model", "sha256":sha_file(artifact)}
    identity.setdefault("foundation_revision",None)
    source_run_id = request.get("source_run_id",request["run_id"])
    evaluated = evaluate(directory,persisted_spec,Path(request.get("evaluator_hash_store",directory.parent.parent/"evaluator.sha256")),pred_file,data.features[:data.t,data.target_index].astype(np.float64))
    evaluated.update(run_id=request["run_id"],spec_hash=request["run_id"],artifact_binding={"model_artifact":{**identity,"source_run_id":source_run_id}, "prediction_artifacts":{key+"_sha256":sha_file(directory/file) for key,file in (("predictions","predictions.csv"),("actual","actual.npy"),("pred","pred.npy"),("train_scale","train_scale.json"))}})
    if source:
        dump(directory/"model/reuse.json", {"source_run_id":source_run_id, "source_artifact_kind":identity["kind"], "source_artifact_sha256":identity["sha256"], "training_performed":False, "evaluated_split":split})
    dump(directory/"eval.json",evaluated)
    result["metrics"] = evaluated["metrics"]
    result["spec"] = persisted_spec
    dump(directory / "result.json", result)
    return result


def diagnostics(data, file):
    points = load_predictions(file)
    if any(not data.t <= p["row_index"] < data.a for p in points):
        raise ValueError("Adjustment diagnostics can only use validation-front predictions")
    residuals = [(p["actual"]-p["predicted"])*data.scale for p in points]
    by_horizon = {}
    for p, residual in zip(points, residuals):
        by_horizon.setdefault(str(p["horizon_index"]), []).append(residual)
    return {"scope": "adjustment_fit_only", "mean_residual_original": statistics.mean(residuals),
            "mae_original": statistics.mean(abs(v) for v in residuals),
            "residual_by_horizon": {k: statistics.mean(v) for k,v in by_horizon.items()},
            "suggested_offset": statistics.mean(residuals), "points": len(points)}


def validate_rule(rule):
    allowed = {"offset", "scale", "slope", "min", "max", "reason"}
    if set(rule) - allowed:
        raise ValueError("Unknown adjustment rule field")
    for key, value in rule.items():
        if key != "reason" and (not isinstance(value, (int,float)) or not math.isfinite(value)):
            raise ValueError("Adjustment coefficients must be finite")
    if abs(rule.get("scale", 1)) > 10 or ("min" in rule and "max" in rule and rule["min"] > rule["max"]):
        raise ValueError("Invalid adjustment scale or bounds")


def adjust(data, request):
    rule = request["rule"]
    validate_rule(rule)
    points = load_predictions(request["predictions"])
    for p in points:
        original = p["predicted_original"]*rule.get("scale", 1)+rule.get("offset", 0)+rule.get("slope", 0)*p["horizon_index"]
        original = min(rule.get("max", math.inf), max(rule.get("min", -math.inf), original))
        if not math.isfinite(original):
            raise ValueError("Non-finite adjusted prediction")
        p["predicted_original"] = original
        p["predicted"] = (original-data.mean)/data.scale
    out = Path(request["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_predictions(out / "predictions.csv", points)
    result = {"predictions": str((out/"predictions.csv").resolve()), "metrics": metrics(points, data.mase_scale), "rule": rule}
    dump(out/"result.json", result)
    return result


def policy(data):
    # Calibration can access validation-front targets, never held-out validation/test targets.
    train = data.values[:data.t]
    calibration = data.values[data.t:data.a]
    observed = train+calibration
    return {"version": 1, "scope": "train_and_adjustment_fit", "min": min(observed)-4*data.scale,
            "max": max(observed)+4*data.scale,
            "max_abs_step": max(abs(b-a) for a,b in zip(observed, observed[1:]))+4*data.scale}


def blind_windows(data, file):
    points = load_predictions(file)
    groups = {}
    for p in points:
        groups.setdefault(p["window_index"], []).append(p)
    windows = []
    for index, group in groups.items():
        origin = group[0]["origin"]
        windows.append({"window_index": index, "history": [{"timestamp": data.times[i], "value": data.values[i]}
                        for i in range(origin-data.task["seq_len"], origin)],
                        "points": [{"timestamp": p["timestamp"], "horizon_index": p["horizon_index"],
                                    "value": p["predicted_original"]} for p in group],
                        "forecast_gap": data.task.get("forecast_gap", 0)})
    return windows


def finalize_windows(data, request):
    points = load_predictions(request["predictions"])
    updates = request["windows"]
    for p in points:
        values = updates[str(p["window_index"])]
        value = values[p["horizon_index"]]
        if not isinstance(value, (int,float)) or not math.isfinite(value):
            raise ValueError("Invalid finalized window point")
        p["predicted_original"] = value
        p["predicted"] = (value-data.mean)/data.scale
    out = Path(request["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_predictions(out/"predictions.csv", points)
    result = {"predictions": str((out/"predictions.csv").resolve()), "metrics": metrics(points, data.mase_scale), "windows": len(updates)}
    dump(out/"result.json", result)
    return result


def visualize(file, output):
    points = load_predictions(file)
    stride = max(1, len(points)//1000)
    sample = points[::stride]
    values = [p[k] for p in sample for k in ("actual_original", "predicted_original")]
    lo, hi = min(values), max(values)
    width, height = 1000, 320
    def poly(key):
        return " ".join(f"{30+i*940/max(1,len(sample)-1):.2f},{290-(p[key]-lo)*250/max(hi-lo,1e-12):.2f}" for i,p in enumerate(sample))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="1000" height="320" fill="white"/><text x="30" y="20">Actual (blue) / forecast (orange); overlapping forecast points</text><polyline fill="none" stroke="#2459a6" points="{poly("actual_original")}"/><polyline fill="none" stroke="#e97624" points="{poly("predicted_original")}"/></svg>'
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(svg, encoding="utf-8")
    return {"file": str(Path(output).resolve())}


def dispatch(request):
    if request.get("protocol_version") != 1:
        raise ValueError("Unsupported IPC protocol version")
    op = request["operation"]
    if op == "engine_module":
        import contextlib
        import importlib
        import io
        import sys
        module = request["module"]
        if module not in {"adacast_lite_ml.engine.runner", "adacast_lite_ml.engine.prediction_prior", "adacast_lite_ml.engine.reflection.controller", "adacast_lite_ml.engine.adjustment.apply", "adacast_lite_ml.engine.adjustment.diagnostics", "adacast_lite_ml.engine.adjustment.evaluate"}:
            raise ValueError("Unsupported engine module")
        if module == "adacast_lite_ml.engine.runner":
            import argparse
            parser = argparse.ArgumentParser()
            for flag in ("spec","run-id","out-dir","evaluator-hash-store","reuse-model-from"):
                parser.add_argument("--"+flag,required=True)
            args = parser.parse_args(request["args"])
            spec = json.loads(Path(args.spec).read_text())
            source_artifact = Path(args.reuse_model_from)/"model/model.json"
            data = Data(request["task"])
            result = experiment(data,{"spec":spec,"split":spec.get("eval_split","val"),"run_id":args.run_id,"output_dir":str(Path(args.out_dir)/args.run_id),"source_artifact":str(source_artifact),"source_sha256":sha_file(source_artifact),"source_run_id":Path(args.reuse_model_from).name,"evaluator_hash_store":args.evaluator_hash_store})
            return {"status":"success","run_id":result["run_id"],"out_dir":str(Path(args.out_dir)/args.run_id),"metrics":result["metrics"]}
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = importlib.import_module(module).main(request["args"])
        if errors.getvalue():
            print(errors.getvalue(),file=sys.stderr)
        text = output.getvalue().strip()
        if code not in (0, None):
            raise ValueError(errors.getvalue().strip() or text or f"Engine module exited with {code}")
        return json.loads(text)
    if op == "models":
        from .catalog import catalog
        return catalog()
    if op == "predict":
        artifact = Path(request["model_artifact"])
        model = json.loads(artifact.read_text())
        for name, expected in model["files"].items():
            if Path(name).name != name or sha_file(artifact.parent / name) != expected:
                raise ValueError("Model file hash mismatch")
        if model["name"] == "UserPrior":
            raise ValueError("UserPrior is a supplied forecast file, not a fitted model for new history")
        import numpy as np
        history = np.asarray(request["history"],dtype=np.float32)
        if len(history) < model["seq_len"] or not np.isfinite(history).all():
            raise ValueError("Provide enough finite original-unit history values")
        history = history[-model["seq_len"]:]
        mean, scale = model["scale"]["mean"], model["scale"]["std"]
        if model.get("engine_config"):
            count = len(model["feature_columns"])
            if history.ndim == 1 and count == 1:
                history = history[:,None]
            if history.ndim != 2 or history.shape[1] != count:
                raise ValueError("Multivariate models require every feature in the saved feature_columns order")
            standardized = history.copy()
            # StandardScaler performs these operations separately in float32.
            standardized -= np.asarray(model["scaler_mean"])
            standardized /= np.asarray(model["scaler_scale"])
        else:
            if history.ndim != 1:
                raise ValueError("Univariate model history must be a numeric array")
            standardized = history.copy()
            standardized -= mean
            standardized /= scale
        predict = predictor(model, artifact.parent, standardized)
        if model.get("engine_config"):
            import pandas as pd
            from .engine.utils.timefeatures import time_features
            dates = request.get("history_timestamps")
            if dates:
                if len(dates) != len(request["history"]):
                    raise ValueError("history_timestamps must match history rows")
                historical = pd.DatetimeIndex(pd.to_datetime(dates[-model["seq_len"]:]))
                future = pd.date_range(historical[-1],periods=model["horizon"]+1,freq=model["engine_config"]["freq"])[1:]
                marks = time_features(historical.append(future),freq=model["engine_config"]["freq"]).transpose().astype(np.float32)
                label = model["engine_config"]["label_len"]
                predict.marks = (marks[:model["seq_len"]],marks[model["seq_len"]-label:])
            elif model["name"] != "DLinear":
                raise ValueError("Provide history_timestamps for calendar-aware model inference")
        values = [float(v)*scale+mean for v in predict(standardized)]
        gap = model.get("canonical_spec",{}).get("forecast_gap",0)
        return {"values":values,"forecast_values":values[gap:],"forecast_gap":gap,"model":model["name"],"units":"original"}
    data = Data(request["task"])
    if op == "inspect":
        if not request.get("skip_model_check"):
            preflight(data.task["model_families"],deterministic=request.get("execution_mode")=="deterministic")
        return data.info()
    if op == "analyze":
        return analyze(data)
    if op == "experiment":
        return experiment(data, request)
    if op == "diagnostics":
        return diagnostics(data, request["predictions"])
    if op == "adjust":
        return adjust(data, request)
    if op == "policy":
        return policy(data)
    if op == "blind_windows":
        return blind_windows(data, request["predictions"])
    if op == "finalize_windows":
        return finalize_windows(data, request)
    if op == "visualize":
        return visualize(request["predictions"], request["output"])
    raise ValueError(f"Unknown operation {op}")
