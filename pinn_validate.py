import numpy as np, pandas as pd


def validate(pred, truth, near_threshold=0.02):
    out = []
    for v in ["Ux", "Uy", "Uz"]:
        y = truth[v].to_numpy()
        yh = pred[v + "_mean"].to_numpy()
        n = 100 * np.sqrt(np.mean((yh - y) ** 2)) / max(y.max() - y.min(), 1e-12)
        out.append([v + " NRMSE", n, "<=10%", n <= 10])
    for v, lim in [("power_W", 10), ("torque_Nm", 10), ("t95_s", 15)]:
        if v in truth and v + "_mean" in pred:
            e = 100 * np.mean(
                np.abs(pred[v + "_mean"] - truth[v])
                / np.maximum(np.abs(truth[v]), 1e-12)
            )
            out.append([v + " error", e, f"<={lim}%", e <= lim])
    cov = []
    for v in ["Ux", "Uy", "Uz", "p", "k", "omega"]:
        if v in truth and v + "_lo95" in pred:
            cov.append(
                100
                * np.mean(
                    (truth[v] >= pred[v + "_lo95"]) & (truth[v] <= pred[v + "_hi95"])
                )
            )
    c = float(np.mean(cov)) if cov else np.nan
    out.append(["95% interval coverage", c, "90-95%", np.isfinite(c) and 90 <= c <= 95])
    cv = float(pred.filter(regex="_cv_pct$").max(axis=1).max())
    out.append(["Maximum predictive CV", cv, "<=15%", cv <= 15])
    nout = int(
        pred.get("outside_training_range", pd.Series(False, index=pred.index)).sum()
    )
    out.append(["Outside training range points", nout, "0", nout == 0])
    for col, label in [("wall_distance", "wall"), ("impeller_distance", "impeller")]:
        if col in truth and (truth[col] <= near_threshold).any():
            q = truth[col] <= near_threshold
            vt = np.sqrt(sum(truth[v][q] ** 2 for v in ["Ux", "Uy", "Uz"]))
            vp = np.sqrt(sum(pred[v + "_mean"][q] ** 2 for v in ["Ux", "Uy", "Uz"]))
            b = 100 * np.mean(vp - vt) / max(np.mean(np.abs(vt)), 1e-12)
            out.append([label + " velocity bias", b, ">=-5%", b >= -5])
    return pd.DataFrame(out, columns=["Criterion", "Observed", "Acceptance", "Pass"])
