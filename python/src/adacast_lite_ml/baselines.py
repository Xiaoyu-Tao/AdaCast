"""Small explicit Lite-only baselines; original models live in the numerical engine."""
import math
import statistics

def solve(matrix, rhs):
    # Partial-pivot Gaussian elimination; ridge term is applied by the caller.
    n = len(rhs)
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda k: abs(a[k][i]))
        a[i], a[pivot] = a[pivot], a[i]
        if abs(a[i][i]) < 1e-14:
            raise ValueError("Singular regression system")
        factor = a[i][i]
        a[i] = [v / factor for v in a[i]]
        for j in range(n):
            if j != i:
                factor = a[j][i]
                a[j] = [v - factor * p for v, p in zip(a[j], a[i])]
    return [row[-1] for row in a]


def fit(data, name, params, directory):
    train = data.y[:data.t]
    seq = data.task["seq_len"]
    horizon = data.task.get("forecast_gap", 0) + data.task["pred_len"]
    model = {"name": name, "params": params, "seq_len": seq, "horizon": horizon}
    if name == "Mean":
        model["mean"] = statistics.mean(train)
    elif name == "SeasonalNaive":
        period = int(params.get("period", min(24, seq)))
        if not 1 <= period <= seq:
            raise ValueError("Seasonal period must be between 1 and seq_len")
        model["period"] = period
    elif name == "Linear":
        lags = int(params.get("lags", min(seq, 24)))
        ridge = float(params.get("ridge", 0.01))
        if not 1 <= lags <= min(seq, 128) or not 0 < ridge <= 1e6:
            raise ValueError("Linear requires lags <= min(seq_len,128) and positive ridge <= 1e6")
        size = lags + 1
        matrix = [[ridge * int(i == j) for j in range(size)] for i in range(size)]
        rhs = [0.0] * size
        for t in range(lags, len(train)):
            x = [1.0] + train[t-lags:t]
            for i in range(size):
                rhs[i] += x[i] * train[t]
                for j in range(size):
                    matrix[i][j] += x[i] * x[j]
        model.update(lags=lags, coefficients=solve(matrix, rhs))
    return model


def predictor(model,directory,training):
    name,horizon=model["name"],model["horizon"]
    def predict(history):
        if name == "Mean":
            return [model["mean"]] * horizon
        if name == "SeasonalNaive":
            cycle = history[-model["period"]:]
            return [cycle[i % len(cycle)] for i in range(horizon)]
        if name == "Linear":
            values = list(history)
            for _ in range(horizon):
                x = [1.0]+values[-model["lags"]:]
                value = sum(a*b for a, b in zip(x, model["coefficients"]))
                if not math.isfinite(value):
                    raise ValueError("Non-finite model prediction")
                values.append(value)
            return values[-horizon:]
        if name == "Naive":
            return [history[-1]] * horizon
        raise ValueError(f"Unknown model artifact {name}")
    return predict
