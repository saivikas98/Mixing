import io, torch, numpy as np, pandas as pd
from pinn_train import PINN


def load_model(file):
    c = torch.load(file, map_location="cpu")
    m = PINN(
        nin=len(c["inputs"]), nout=len(c["outputs"]), dropout=c.get("dropout", 0.05)
    )
    m.load_state_dict(c["state"])
    return m, c


def infer(file, df, n_mc=100):
    m, c = load_model(file)
    X = torch.tensor(df[c["inputs"]].values, dtype=torch.float32)
    xn = (X - c["xm"]) / c["xs"]
    m.train()
    S = []
    with torch.no_grad():
        for _ in range(n_mc):
            S.append((m(xn) * c["ys"] + c["ym"]).numpy())
    S = np.stack(S)
    train_min = c.get("train_min")
    train_max = c.get("train_max")
    outside = np.zeros(len(df), dtype=bool)
    if train_min is not None and train_max is not None:
        outside = ((X < train_min) | (X > train_max)).any(1).numpy()
    mean = S.mean(0)
    sd = S.std(0, ddof=1)
    lo = np.quantile(S, 0.025, axis=0)
    hi = np.quantile(S, 0.975, axis=0)
    o = df.copy()
    for j, n in enumerate(c["outputs"]):
        o[n + "_mean"] = mean[:, j]
        o[n + "_sd"] = sd[:, j]
        o[n + "_lo95"] = lo[:, j]
        o[n + "_hi95"] = hi[:, j]
    o["outside_training_range"] = outside
    for n in c["outputs"]:
        o[n + "_cv_pct"] = (
            100 * o[n + "_sd"] / np.maximum(np.abs(o[n + "_mean"]), 1e-12)
        )
    o["uq_warning"] = o[[n + "_cv_pct" for n in c["outputs"]]].max(axis=1) > 15
    return o
