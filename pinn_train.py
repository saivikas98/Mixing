import argparse, json, pathlib
import numpy as np, pandas as pd, torch
from torch import nn


class PINN(nn.Module):
    def __init__(self, nin=6, nout=6, width=128, depth=6, dropout=0.05):
        super().__init__()
        L = [nn.Linear(nin, width), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            L += [nn.Linear(width, width), nn.Tanh(), nn.Dropout(dropout)]
        L += [nn.Linear(width, nout)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def grad(y, x):
    return torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def train(
    csv,
    out,
    epochs=5000,
    rho=1000.0,
    nu=1e-6,
    beta_star=0.09,
    alpha=0.52,
    beta=0.075,
    sigma_k=0.5,
    sigma_w=0.5,
):
    df = pd.read_csv(csv)
    ins = ["x", "y", "z", "rpm", "fill_height", "viscosity"]
    outs = ["Ux", "Uy", "Uz", "p", "k", "omega"]
    miss = [c for c in ins + outs if c not in df]
    if miss:
        raise ValueError("Missing columns: " + ",".join(miss))
    X = torch.tensor(df[ins].values, dtype=torch.float32)
    Y = torch.tensor(df[outs].values, dtype=torch.float32)
    xm, xs = X.mean(0), X.std(0).clamp_min(1e-8)
    ym, ys = Y.mean(0), Y.std(0).clamp_min(1e-8)
    xn = ((X - xm) / xs).requires_grad_(True)
    yn = (Y - ym) / ys
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    hist = []
    for e in range(epochs):
        opt.zero_grad()
        qn = model(xn)
        q = qn * ys + ym
        u, v, w, pf, k, om = [q[:, i : i + 1] for i in range(6)]
        k = torch.nn.functional.softplus(k)
        om = torch.nn.functional.softplus(om) + 1e-7
        gu, gv, gw, gp, gk, go = [grad(a, xn) / xs for a in (u, v, w, pf, k, om)]
        cont = gu[:, 0:1] + gv[:, 1:2] + gw[:, 2:3]
        # strain production Pk=2 nut S:S, nut=k/omega; derivatives in physical coordinates
        sxx, syy, szz = gu[:, 0:1], gv[:, 1:2], gw[:, 2:3]
        sxy = 0.5 * (gu[:, 1:2] + gv[:, 0:1])
        sxz = 0.5 * (gu[:, 2:3] + gw[:, 0:1])
        syz = 0.5 * (gv[:, 2:3] + gw[:, 1:2])
        S2 = sxx * sxx + syy * syy + szz * szz + 2 * (sxy * sxy + sxz * sxz + syz * syz)
        nut = k / om
        Pk = 2 * nut * S2

        def lap(a):
            ga = grad(a, xn) / xs
            return sum(
                (grad(ga[:, j : j + 1], xn) / xs)[:, j : j + 1] for j in range(3)
            )

        lu, lv, lw, lk, lo = [lap(a) for a in (u, v, w, k, om)]
        conv = lambda ga: u * ga[:, 0:1] + v * ga[:, 1:2] + w * ga[:, 2:3]
        rx = conv(gu) + gp[:, 0:1] / rho - (nu + nut) * lu
        ry = conv(gv) + gp[:, 1:2] / rho - (nu + nut) * lv
        rz = conv(gw) + gp[:, 2:3] / rho - (nu + nut) * lw
        rk = conv(gk) - Pk + beta_star * k * om - (nu + sigma_k * nut) * lk
        rw = (
            conv(go)
            - alpha * (om / k.clamp_min(1e-7)) * Pk
            + beta * om * om
            - (nu + sigma_w * nut) * lo
        )
        data = ((qn - yn) ** 2).mean()
        phys = sum(a.square().mean() for a in (cont, rx, ry, rz, rk, rw))
        loss = data + 1e-4 * phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()
        if e % 100 == 0:
            hist.append([e, float(loss), float(data), float(phys)])
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "xm": xm,
            "xs": xs,
            "ym": ym,
            "ys": ys,
            "inputs": ins,
            "outputs": outs,
            "rho": rho,
            "nu": nu,
            "dropout": 0.05,
            "train_min": X.min(0).values,
            "train_max": X.max(0).values,
        },
        out / "pinn.pt",
    )
    pd.DataFrame(hist, columns=["epoch", "loss", "data_loss", "rans_loss"]).to_csv(
        out / "history.csv", index=False
    )
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "model": "steady incompressible RANS k-omega hybrid PINN",
                "epochs": epochs,
                "UQ": "MC dropout epistemic uncertainty",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True)
    a.add_argument("--out", default="pinn_model")
    a.add_argument("--epochs", type=int, default=5000)
    z = a.parse_args()
    train(z.csv, z.out, z.epochs)
