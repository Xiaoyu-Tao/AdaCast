"""Model behavior retained in the project's independently owned numerical engine."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

from .engine.statistical import (
    STATISTICAL_MODEL_NAMES, fit_statistical_model, save_statistical_result,
    load_statistical_result, forecast_statistical_result,
)
from .engine.models._foundation import FOUNDATION_MODEL_REPOS

from .catalog import BASELINES as _BASELINES, NAMES, preflight, canonical_model, model_family
MODELS = {name: None for name in NAMES}
_LIVE = {}


def family(name):
    return model_family(name)


def normalize_params(name, params):
    result = dict(params)
    aliases = {"epochs": "train_epochs", "kernel": "moving_avg"} if name not in _BASELINES else {}
    for alias, canonical in aliases.items():
        if alias in result:
            if canonical in result and result[alias] != result[canonical]:
                raise ValueError(f"Conflicting {alias} and {canonical}")
            result[canonical] = result.pop(alias)
    if name == "ARIMA" and any(k in result for k in ("p", "d", "q")):
        order = [int(result.pop(k, default)) for k, default in (("p", 1), ("d", 1), ("q", 1))]
        if "order" in result and list(result["order"]) != order:
            raise ValueError("Conflicting ARIMA order parameters")
        result["order"] = order
    if name == "Holt" and any(k in result for k in ("alpha", "beta")):
        raise ValueError("Holt uses optimized statsmodels fitting; fixed alpha/beta parameters are unsupported")
    return result


def canonical_spec(data, name, params):
    from .frequencies import normalize_frequency
    normalized = normalize_params(name, params)
    if "freq" in normalized: normalized["freq"] = normalize_frequency(normalized["freq"])
    controls = {k: normalized[k] for k in ("seed", "freq") if k in normalized}
    return {**data.task, "model": name, "seed": 42, "freq": data.frequency,
            "input_contract_version": 3, **controls,
            "hyperparams": {k: v for k, v in normalized.items() if k not in ("seed", "freq", "device", "threads")}}


def fit(data, name, params, directory):
    name = canonical_model(name)
    preflight([name])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if name in _BASELINES:
        from .baselines import fit as baseline_fit
        model = baseline_fit(data, name, params, directory)
        model["family"] = "baseline"
        return model
    spec = canonical_spec(data, name, params)
    if name in STATISTICAL_MODEL_NAMES:
        canonical, fitted = fit_statistical_model(name, data.features[:data.t, data.target_index], spec["hyperparams"])
        save_statistical_result(fitted, directory/"statistical-results.pkl")
        return {"name": name, "params": params, "family": "statistical", "weights": "statistical-results.pkl",
                "seq_len": data.task["seq_len"], "horizon": data.task["forecast_gap"]+data.task["pred_len"],
                "train_end": data.t, "canonical_spec": spec}
    import torch
    from torch.utils.data import DataLoader
    from .engine.runner import (LazyModelDict, build_model_config, instantiate_model, ForecastWindowDataset,
                                train_model, resolve_foundation_snapshot, foundation_model_load_path)
    from .engine.data_loader import load_dataset
    from .engine.utils.seed import set_seed
    from .engine.utils.device import detect_device
    set_seed(int(spec["seed"]))
    torch.set_num_threads(int(params.get("threads", 2)))
    device = torch.device(params["device"]) if "device" in params else detect_device()
    config = build_model_config(spec, len(data.feature_columns))
    registry = LazyModelDict(str(Path(__file__).parent/"engine/models"))
    if name in FOUNDATION_MODEL_REPOS:
        snapshot, revision = resolve_foundation_snapshot(name, None)
        config.foundation_model_repo = FOUNDATION_MODEL_REPOS[name]
        config.foundation_model_revision = revision
        config.foundation_model_path = str(foundation_model_load_path(name, snapshot))
    model = instantiate_model(registry[name], config, spec["hyperparams"]).to(device)
    if name not in FOUNDATION_MODEL_REPOS:
        splits = load_dataset(spec, ["train"])
        dataset = ForecastWindowDataset(splits["train"], seq_len=config.seq_len, label_len=config.label_len,
            pred_len=data.task["pred_len"], target_index=data.target_index, freq=config.freq, input_contract_version=3,
            window_stride=data.task.get("window_stride", 1), forecast_gap=data.task.get("forecast_gap", 0))
        loader = DataLoader(dataset, batch_size=int(spec["hyperparams"].get("batch_size", 32)), shuffle=True, drop_last=False)
        train_model(model, loader, device, config.pred_len, int(spec["hyperparams"].get("train_epochs", 5)),
                    float(spec["hyperparams"].get("learning_rate", .0001)))
        torch.save(model.state_dict(), directory/"checkpoint.pth")
    metadata = {"name": name, "params": params, "family": family(name), "weights": "checkpoint.pth" if name not in FOUNDATION_MODEL_REPOS else None,
                "seq_len": config.seq_len, "horizon": config.pred_len, "canonical_spec": spec,
                "engine_config": vars(config), "feature_columns": data.feature_columns, "target_index": data.target_index,
                "scaler_mean": data.means, "scaler_scale": data.scales}
    _LIVE[str(directory.resolve())] = (model, device)
    return metadata


def predictor(model, directory, training=None):
    directory = Path(directory)
    if model.get("family") == "statistical":
        fitted = load_statistical_result(directory/model["weights"])
        return lambda history: forecast_statistical_result(fitted, model["horizon"]).tolist()
    if model["name"] in _BASELINES:
        from .baselines import predictor as baseline_predictor
        return baseline_predictor(model, directory, training)
    import torch
    from types import SimpleNamespace
    from .engine.runner import LazyModelDict, instantiate_model, build_decoder_input
    from .engine.utils.seed import set_seed
    from .engine.utils.device import detect_device
    config = SimpleNamespace(**model["engine_config"])
    live = _LIVE.pop(str(directory.resolve()), None)
    if live:
        network, device = live
    else:
        set_seed(int(model["canonical_spec"]["seed"]))
        device = torch.device(model["params"]["device"]) if "device" in model["params"] else detect_device()
        registry = LazyModelDict(str(Path(__file__).parent/"engine/models"))
        if model["name"] in FOUNDATION_MODEL_REPOS:
            from .engine.runner import resolve_foundation_snapshot, foundation_model_load_path
            snapshot, revision = resolve_foundation_snapshot(model["name"], config.foundation_model_revision)
            config.foundation_model_path = str(foundation_model_load_path(model["name"], snapshot))
        network = instantiate_model(registry[model["name"]], config, model["canonical_spec"]["hyperparams"]).to(device)
        if model["weights"]:
            network.load_state_dict(torch.load(directory/model["weights"], map_location=device, weights_only=True))
    network.eval()
    def predict(history):
        x = torch.as_tensor(history, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        x = x.unsqueeze(0).to(device)
        label = x[:, -config.label_len:, :] if config.label_len else x[:, :0, :]
        decoder = torch.cat([label, torch.zeros((1, config.pred_len, x.shape[-1]), device=device)], dim=1)
        # Actual calendar marks are supplied by experiment() for models that consume them.
        marks = getattr(predict, "marks", None)
        if marks is None:
            marks = (torch.zeros((1, x.shape[1], 4), device=device), torch.zeros((1, config.label_len+config.pred_len, 4), device=device))
        else:
            marks = tuple(torch.as_tensor(m, dtype=torch.float32).unsqueeze(0).to(device) for m in marks)
        with torch.no_grad():
            values = network(x, marks[0], decoder, marks[1])[0, :, model["target_index"]]
        return values.detach().cpu().tolist()
    return predict


def statistical_curve(model, directory, steps):
    return forecast_statistical_result(load_statistical_result(Path(directory)/model["weights"]), steps).tolist()


def torch_model(seq, horizon, kernel):
    """Convenience constructor with the canonical DLinear initialization."""
    from types import SimpleNamespace
    from .engine.models.DLinear import Model
    return Model(SimpleNamespace(task_name="long_term_forecast", seq_len=seq, pred_len=horizon, moving_avg=kernel, enc_in=1))
