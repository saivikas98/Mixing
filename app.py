import io, math, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Bottom-Mount GMP Mixer Simulator", layout="wide")
st.title("Bottom-Mount GMP Mixer: Hydrodynamic Screening Simulator")
st.caption(
    "Engineering surrogate for screening and study design. Not a replacement for geometry-specific CFD or GMP qualification."
)


# -------------------- model --------------------
def model(par):
    T, H, D, C, rpm, rho, mu, Np, Nq, Pmotor, eta, Trated, sf, nr, nz = [
        par[k]
        for k in [
            "T",
            "H",
            "D",
            "C",
            "rpm",
            "rho",
            "mu",
            "Np",
            "Nq",
            "Pmotor",
            "eta",
            "Trated",
            "sf",
            "nr",
            "nz",
        ]
    ]
    R = T / 2
    V = math.pi * R * R * H
    N = rpm / 60
    tip = math.pi * D * N
    Re = rho * N * D**2 / mu
    P = Np * rho * N**3 * D**5
    Q = Nq * N * D**3
    torque = P / (2 * math.pi * N) if N > 0 else 0
    r = np.linspace(0, R, int(nr))
    z = np.linspace(0, H, int(nz))
    rr, zz = np.meshgrid(r, z)
    ri = D / 2
    safe = np.maximum(rr, max(R / 500, 1e-5))
    # Divergence-consistent axisymmetric meridional field generated from streamfunction.
    psi = (
        0.020
        * (rr / R) ** 2
        * (1 - rr / R) ** 2
        * np.sin(np.pi * zz / H)
        * np.exp(-0.15 * zz / H)
        + 0.006
        * (rr / R) ** 2
        * (1 - rr / R) ** 2
        * np.sin(2 * np.pi * zz / H)
        * np.exp(-2.2 * zz / H)
        + 0.013
        * (rr / R) ** 2
        * np.exp(-((rr / max(0.43 * D, 1e-6)) ** 2))
        * (1 - np.exp(-zz / max(0.8 * C, 1e-6)))
        * np.exp(-zz / max(0.8 * H, 1e-6))
    )
    ur = -np.gradient(psi, z, axis=0) / safe
    uz = np.gradient(psi, r, axis=1) / safe
    ur[:, 0] = 0
    uz[:, 0] = uz[:, 1]
    vm = np.sqrt(ur**2 + uz**2)
    s = 0.42 * tip / max(np.percentile(vm, 99.5), 1e-9)
    ur *= s
    uz *= s
    ut = (
        0.72
        * tip
        * (rr / max(ri, 1e-6))
        * np.exp(
            -(((rr - ri) / max(0.34 * D, 1e-6)) ** 2)
            - ((zz - C) / max(0.30 * D, 1e-6)) ** 2
        )
    )
    ut += 0.10 * tip * (rr / R) * np.exp(-zz / max(0.44 * H, 1e-6)) * (1 - rr / R)
    vel = np.sqrt(ur**2 + uz**2 + ut**2)
    durdr = np.gradient(ur, r, axis=1)
    durdz = np.gradient(ur, z, axis=0)
    duzdr = np.gradient(uz, r, axis=1)
    duzdz = np.gradient(uz, z, axis=0)
    dutdr = np.gradient(ut, r, axis=1)
    dutdz = np.gradient(ut, z, axis=0)
    shear = np.sqrt(
        np.maximum(
            0,
            2 * (durdr**2 + duzdz**2 + (ur / safe) ** 2)
            + (durdz + duzdr) ** 2
            + (dutdr - ut / safe) ** 2
            + dutdz**2,
        )
    )
    # Local generalized-Newtonian apparent viscosity and shear stress
    g = np.maximum(shear, 1e-6)
    rheo = par.get("rheology", "Newtonian")
    if rheo == "Power law":
        mu_field = par["K"] * g ** (par["n_index"] - 1)
    elif rheo == "Herschel-Bulkley":
        mu_field = par["tau0"] / g + par["K"] * g ** (par["n_index"] - 1)
    elif rheo == "Bingham plastic":
        mu_field = par["tau0"] / g + par["mu_plastic"]
    elif rheo == "Carreau-Yasuda":
        mu_field = par["muinf"] + (par["mu0"] - par["muinf"]) * (
            1 + (par["lambda_c"] * g) ** par["a_c"]
        ) ** ((par["n_c"] - 1) / par["a_c"])
    else:
        mu_field = np.full_like(g, mu)
    mu_field = np.clip(mu_field, 1e-9, 1e8)
    tau = mu_field * shear
    k = 0.015 * vel**2 + 1.5 * (P / (rho * V)) ** (2 / 3) * np.exp(
        -zz / max(0.35 * H, 1e-6)
    )
    eps = (P / (rho * V)) * (
        0.25
        + 5
        * np.exp(
            -(((rr - ri) / max(0.35 * D, 1e-6)) ** 2)
            - ((zz - C) / max(0.30 * D, 1e-6)) ** 2
        )
    )
    # volume weights for axisymmetric cells
    w = 2 * np.pi * rr
    w /= w.sum()
    threshold = par["dead_threshold"]
    dead = float(w[vel < threshold].sum() * 100)
    # Lumped tracer mixing: normalized concentration at worst probe, exponential approach to 1
    turnover = V / max(Q, 1e-12)
    tau_mix = (
        1.65
        * turnover
        * (1 + 0.20 * (H / T) ** 1.2)
        * (1 + 0.10 * max(0, 1e4 / Re) ** 0.25)
    )
    t = np.linspace(0, 6 * tau_mix, 240)
    conc = 1 - np.exp(-t / tau_mix)
    times = {p: -tau_mix * np.log(1 - p / 100) for p in (90, 95, 99)}
    available_torque = min(Trated, Pmotor * 1000 * eta / max(2 * math.pi * N, 1e-12))
    margin = available_torque / max(sf * torque, 1e-12)
    return locals()


def contour(x, y, f, title, unit, colors="Turbo", vectors=None):
    fig = go.Figure(
        go.Contour(
            x=x,
            y=y,
            z=f,
            colorscale=colors,
            colorbar_title=unit,
            contours=dict(coloring="heatmap", showlines=False),
        )
    )
    if vectors is not None:
        ur, uz = vectors
        si = max(1, len(y) // 22)
        sj = max(1, len(x) // 18)
        xs = []
        ys = []
        scale = 0.035 / max(np.percentile(np.sqrt(ur**2 + uz**2), 95), 1e-9)
        for i in range(0, len(y), si):
            for j in range(0, len(x), sj):
                xs += [x[j], x[j] + ur[i, j] * scale, None]
                ys += [y[i], y[i] + uz[i, j] * scale, None]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="white", width=1),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Radius (m)",
        yaxis_title="Liquid height (m)",
        height=520,
        margin=dict(l=30, r=20, t=55, b=35),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


with st.sidebar:
    st.header("Inputs")
    T = st.number_input("Vessel diameter T (m)", 0.1, 10.0, 0.80, 0.01)
    H = st.number_input("Liquid height H (m)", 0.05, 10.0, 0.80, 0.01)
    D = st.number_input(
        "Impeller diameter D (m)",
        0.01,
        float(T * 0.95),
        min(0.28, float(T * 0.9)),
        0.01,
    )
    C = st.number_input(
        "Impeller center clearance C (m)",
        0.005,
        float(H * 0.5),
        min(0.07, float(H * 0.4)),
        0.005,
    )
    rpm = st.number_input("Speed (rpm)", 1.0, 3000.0, 250.0, 5.0)
    rho = st.number_input("Density (kg/m3)", 100.0, 3000.0, 1000.0, 10.0)
    rheology = st.selectbox(
        "Fluid rheology",
        [
            "Newtonian",
            "Power law",
            "Herschel-Bulkley",
            "Bingham plastic",
            "Carreau-Yasuda",
        ],
    )
    mu_mpas = st.number_input(
        "Newtonian/reference viscosity (mPa.s)", 0.1, 1e6, 10.0, 1.0
    )
    st.caption("Non-Newtonian parameters are interpreted in SI units.")
    K_hb = st.number_input(
        "Consistency K (Pa.s^n)",
        1e-6,
        1e7,
        10.0,
        0.1,
        format="%.6g",
        disabled=rheology not in ["Power law", "Herschel-Bulkley"],
    )
    n_hb = st.number_input(
        "Flow index n",
        0.05,
        2.0,
        0.50,
        0.01,
        disabled=rheology not in ["Power law", "Herschel-Bulkley"],
    )
    tau0 = st.number_input(
        "Yield stress tau0 (Pa)",
        0.0,
        1e6,
        5.0,
        0.1,
        disabled=rheology not in ["Herschel-Bulkley", "Bingham plastic"],
    )
    mu_plastic = st.number_input(
        "Plastic viscosity (Pa.s)",
        1e-6,
        1e5,
        0.10,
        0.01,
        format="%.6g",
        disabled=rheology != "Bingham plastic",
    )
    mu0 = st.number_input(
        "Carreau zero-shear viscosity mu0 (Pa.s)",
        1e-6,
        1e7,
        100.0,
        0.1,
        format="%.6g",
        disabled=rheology != "Carreau-Yasuda",
    )
    muinf = st.number_input(
        "Carreau infinite-shear viscosity mu_inf (Pa.s)",
        1e-9,
        1e5,
        0.01,
        0.001,
        format="%.6g",
        disabled=rheology != "Carreau-Yasuda",
    )
    lambda_c = st.number_input(
        "Carreau time constant lambda (s)",
        1e-6,
        1e6,
        1.0,
        0.1,
        format="%.6g",
        disabled=rheology != "Carreau-Yasuda",
    )
    a_c = st.number_input(
        "Yasuda exponent a", 0.1, 10.0, 2.0, 0.1, disabled=rheology != "Carreau-Yasuda"
    )
    n_c = st.number_input(
        "Carreau power index n",
        0.01,
        1.5,
        0.40,
        0.01,
        disabled=rheology != "Carreau-Yasuda",
    )
    Np = st.number_input("Power number Np", 0.01, 20.0, 0.32, 0.01)
    Nq = st.number_input("Flow number Nq", 0.01, 5.0, 0.60, 0.01)
    dead_threshold = st.number_input(
        "Dead-zone velocity threshold (m/s)", 0.0001, 5.0, 0.05, 0.01, format="%.4f"
    )
    st.subheader("Drive / magnetic coupling")
    Pmotor = st.number_input("Motor power (kW)", 0.01, 1000.0, 1.50, 0.1)
    eta = st.number_input("Drive efficiency", 0.1, 1.0, 0.85, 0.01)
    Trated = st.number_input(
        "Vendor coupling torque limit (N.m)", 0.1, 10000.0, 20.0, 0.5
    )
    sf = st.number_input("Required torque safety factor", 1.0, 3.0, 1.50, 0.05)

# Metzner-Otto screening shear rate for stirred vessels; user-editable constant.
Ks = st.sidebar.number_input(
    "Metzner-Otto constant Ks",
    1.0,
    30.0,
    11.5,
    0.5,
    help="Screening value; replace with impeller-specific experimental/vendor value.",
)
gamma_char = max(Ks * rpm / 60, 1e-9)
if rheology == "Newtonian":
    mu_app = mu_mpas / 1000
elif rheology == "Power law":
    mu_app = K_hb * gamma_char ** (n_hb - 1)
elif rheology == "Herschel-Bulkley":
    mu_app = tau0 / gamma_char + K_hb * gamma_char ** (n_hb - 1)
elif rheology == "Bingham plastic":
    mu_app = tau0 / gamma_char + mu_plastic
else:
    mu_app = muinf + (mu0 - muinf) * (1 + (lambda_c * gamma_char) ** a_c) ** (
        (n_c - 1) / a_c
    )
mu_app = max(float(mu_app), 1e-9)
par = dict(
    T=T,
    H=H,
    D=D,
    C=C,
    rpm=rpm,
    rho=rho,
    mu=mu_app,
    Np=Np,
    Nq=Nq,
    Pmotor=Pmotor,
    eta=eta,
    Trated=Trated,
    sf=sf,
    dead_threshold=dead_threshold,
    nr=100,
    nz=150,
)
par.update(
    dict(
        rheology=rheology,
        K=K_hb,
        n_index=n_hb,
        tau0=tau0,
        mu_plastic=mu_plastic,
        mu0=mu0,
        muinf=muinf,
        lambda_c=lambda_c,
        a_c=a_c,
        n_c=n_c,
        Ks=Ks,
        gamma_char=gamma_char,
        mu_app=mu_app,
    )
)
m = model(par)

tabs = st.tabs(
    [
        "Dashboard",
        "2D fields",
        "3D view",
        "Tracer & blend time",
        "Volume envelope",
        "Equations, assumptions & calculations",
        "Data export",
    ]
)
with tabs[0]:
    st.info(
        f"Rheology: {rheology} | characteristic shear rate = {gamma_char:.3g} 1/s | apparent viscosity = {mu_app:.4g} Pa.s"
    )
    c = st.columns(6)
    vals = [
        ("Reynolds", m["Re"]),
        ("Power", m["P"] / 1000),
        ("P/V", m["P"] / m["V"]),
        ("Tip speed", m["tip"]),
        ("Dead zone", m["dead"]),
        ("Torque margin", m["margin"]),
    ]
    units = ["", "kW", "W/m3", "m/s", "%", "x"]
    for col, (lab, val), u in zip(c, vals, units):
        col.metric(lab, f"{val:,.3g} {u}")
    st.plotly_chart(
        contour(
            m["r"],
            m["z"],
            m["vel"],
            "Velocity magnitude with radial/axial vectors",
            "m/s",
            "Turbo",
            (m["ur"], m["uz"]),
        ),
        use_container_width=True,
    )
    if m["margin"] < 1:
        st.error(
            "Magnetic coupling torque margin is below 1.0. The assumed drive/coupling is inadequate at the selected safety factor."
        )
    elif m["margin"] < 1.5:
        st.warning(
            "Torque margin is limited. Obtain vendor torque-versus-speed and temperature derating data."
        )
    else:
        st.success(
            "Screening torque margin is acceptable, subject to vendor coupling limits and start-up torque verification."
        )
with tabs[1]:
    choices = {
        "Velocity magnitude": (m["vel"], "m/s", "Turbo"),
        "Radial velocity ur": (m["ur"], "m/s", "RdBu"),
        "Axial velocity uz": (m["uz"], "m/s", "RdBu"),
        "Tangential velocity uθ": (m["ut"], "m/s", "Viridis"),
        "Shear-rate magnitude": (m["shear"], "1/s", "Inferno"),
        "Viscous shear stress μγ": (m["tau"], "Pa", "Magma"),
        "Apparent viscosity": (m["mu_field"], "Pa.s", "Viridis"),
        "Turbulent kinetic energy k": (m["k"], "m2/s2", "Cividis"),
        "Turbulent dissipation ε": (m["eps"], "m2/s3", "Plasma"),
    }
    selected = st.multiselect(
        "Fields",
        list(choices),
        default=[
            "Velocity magnitude",
            "Shear-rate magnitude",
            "Turbulent kinetic energy k",
            "Turbulent dissipation ε",
        ],
    )
    for name in selected:
        f, u, cs = choices[name]
        st.plotly_chart(
            contour(
                m["r"],
                m["z"],
                f,
                name,
                u,
                cs,
                (m["ur"], m["uz"]) if name == "Velocity magnitude" else None,
            ),
            use_container_width=True,
        )
with tabs[2]:
    st.info(
        "The 3D display is an axisymmetric revolution of the 2D surrogate, not a blade-resolved CFD result."
    )
    field_name = st.selectbox(
        "3D field",
        [
            "Velocity magnitude",
            "Shear-rate magnitude",
            "Turbulent kinetic energy k",
            "Turbulent dissipation ε",
        ],
    )
    key = {
        "Velocity magnitude": "vel",
        "Shear-rate magnitude": "shear",
        "Turbulent kinetic energy k": "k",
        "Turbulent dissipation ε": "eps",
    }[field_name]
    th = np.linspace(0, 2 * np.pi, 70)
    ridx = np.arange(0, len(m["r"]), 3)
    zidx = np.arange(0, len(m["z"]), 4)
    rr = m["rr"][np.ix_(zidx, ridx)]
    zz = m["zz"][np.ix_(zidx, ridx)]
    ff = m[key][np.ix_(zidx, ridx)]
    # render several meridional sheets for readable 3D screening
    fig = go.Figure()
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x = rr * np.cos(ang)
        y = rr * np.sin(ang)
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=zz,
                surfacecolor=ff,
                colorscale="Turbo",
                cmin=np.nanmin(ff),
                cmax=np.nanpercentile(ff, 99.5),
                showscale=bool(np.isclose(ang, 0.0)),
                colorbar=dict(title=field_name),
            )
        )
    fig.update_layout(
        height=720,
        scene=dict(
            aspectmode="data",
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
        ),
        title="3D meridional-sheet reconstruction",
    )
    st.plotly_chart(fig, use_container_width=True)
with tabs[3]:
    fig = go.Figure(
        go.Scatter(
            x=m["t"],
            y=m["conc"],
            mode="lines",
            name="Worst-point normalized concentration",
        )
    )
    for p in (90, 95, 99):
        fig.add_vline(
            x=m["times"][p],
            line_dash="dash",
            annotation_text=f't{p}={m["times"][p]:.1f}s',
        )
    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="C/C∞", yaxis_range=[0, 1.03], height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": ["Turnover V/Q", "t90", "t95", "t99"],
                "Time (s)": [
                    m["turnover"],
                    m["times"][90],
                    m["times"][95],
                    m["times"][99],
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
with tabs[4]:
    fmin, fmax = st.slider(
        "Minimum and maximum liquid-height fraction", 0.15, 1.50, (0.30, 1.00), 0.05
    )
    hs = np.linspace(fmin * H, fmax * H, 12)
    rows = []
    for hh in hs:
        pp = par.copy()
        pp["H"] = float(hh)
        mm = model(pp)
        rows.append(
            [
                hh,
                mm["V"] * 1000,
                mm["P"] / mm["V"],
                mm["times"][95],
                mm["dead"],
                mm["margin"],
            ]
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "Liquid height (m)",
            "Volume (L)",
            "P/V (W/m3)",
            "t95 (s)",
            "Dead-zone (%)",
            "Torque margin (x)",
        ],
    )
    st.dataframe(df.style.format(precision=2), use_container_width=True)
    st.line_chart(df.set_index("Volume (L)")[["t95 (s)", "Dead-zone (%)"]])
with tabs[5]:
    st.header("Model basis, equations and assumptions")
    st.markdown(
        r"""### Governing screening equations
- Rotational speed: $N=\mathrm{rpm}/60$
- Impeller Reynolds number: $Re=\rho N D^2/\mu$
- Tip speed: $u_{tip}=\pi DN$
- Power draw: $P=N_p\rho N^3D^5$
- Pumping rate: $Q=N_qND^3$
- Vessel volume: $V=\pi(T/2)^2H$
- Bulk energy dissipation: $\bar{\varepsilon}=P/(\rho V)$
- Power density: $P/V$
- Shaft torque: $\tau=P/(2\pi N)$
- Available drive torque: $\tau_{avail}=\min[\tau_{vendor},P_{motor}\eta/(2\pi N)]$
- Torque margin: $M=\tau_{avail}/(SF\,\tau)$

### Non-Newtonian rheology
The characteristic mixer shear rate is estimated by the Metzner-Otto relation $\dot\gamma_c=K_sN$. The generalized Reynolds number used by the screening model is $Re_g=\rho ND^2/\mu_{app}(\dot\gamma_c)$.

- Power law: $\tau=K\dot\gamma^n$ and $\mu_{app}=K\dot\gamma^{n-1}$.
- Herschel-Bulkley: $\tau=\tau_0+K\dot\gamma^n$ and $\mu_{app}=\tau_0/\dot\gamma+K\dot\gamma^{n-1}$.
- Bingham plastic: $\tau=\tau_0+\mu_p\dot\gamma$ and $\mu_{app}=\tau_0/\dot\gamma+\mu_p$.
- Carreau-Yasuda: $\mu=\mu_\infty+(\mu_0-\mu_\infty)[1+(\lambda\dot\gamma)^a]^{(n-1)/a}$.

Local shear-stress contours use the local shear-rate field and the selected constitutive equation. Very low shear rates are regularized at $10^{-6}\,s^{-1}$ to prevent singular yield-stress viscosity.

### Velocity field surrogate
A cylindrical streamfunction $\psi(r,z)$ creates a divergence-consistent recirculation field:
$u_r=-(1/r)\partial\psi/\partial z$ and $u_z=(1/r)\partial\psi/\partial r$.
The tangential component is an empirical Gaussian blade-zone field scaled to tip speed. The magnitude is
$|\mathbf{u}|=\sqrt{u_r^2+u_z^2+u_\theta^2}$.

### Shear, stress and turbulence surrogates
Shear-rate magnitude is computed from the axisymmetric rate-of-deformation tensor:
$\dot\gamma=\sqrt{2\mathbf{D}:\mathbf{D}}$. For a Newtonian liquid, local viscous stress is $\tau_v=\mu\dot\gamma$.
The displayed $k$ and $\varepsilon$ are screening correlations tied to local velocity and $P/(\rho V)$, not solved RANS transport equations.

### Tracer and blend time
A lumped first-order approach is used: $C/C_\infty=1-\exp(-t/\tau_m)$.
The characteristic time is correlated to turnover time $V/Q$, aspect ratio and a Reynolds correction. Therefore
$t_p=-\tau_m\ln(1-p)$ for $p=0.90,0.95,0.99$.

### Dead-zone calculation
Axisymmetric volume weighting uses $dV=2\pi r\,dr\,dz$. Dead-zone percentage is the weighted fraction where $|\mathbf{u}|$ is below the selected threshold.

### Assumptions
1. Cylindrical, unbaffled tank; flat liquid surface; centered bottom impeller.
2. Single-phase, incompressible, isothermal Newtonian liquid.
3. Constant $N_p$ and $N_q$ supplied by the user; no Reynolds-dependent vendor curve.
4. Axisymmetric mean-flow approximation; no blade-resolved wakes, free-surface vortex, gas entrainment or solids.
5. No-slip wall effects are only represented qualitatively by the surrogate field.
6. $k$, $\varepsilon$, wall stress and tracer results are estimates for study design, not qualification evidence.
7. Magnetic torque assessment uses steady-state hydraulic torque. Start-up, bearing drag, coupling slip, temperature derating and transient overload require vendor data.

### Current-case calculations"""
    )
    calc = pd.DataFrame(
        {
            "Quantity": [
                "N",
                "V",
                "Re",
                "Tip speed",
                "P",
                "Q",
                "V/Q",
                "P/V",
                "P/(rho V)",
                "Hydraulic torque",
                "Available torque",
                "Torque margin",
                "t90",
                "t95",
                "t99",
            ],
            "Value": [
                m["N"],
                m["V"],
                m["Re"],
                m["tip"],
                m["P"],
                m["Q"],
                m["turnover"],
                m["P"] / m["V"],
                m["P"] / (rho * m["V"]),
                m["torque"],
                m["available_torque"],
                m["margin"],
                m["times"][90],
                m["times"][95],
                m["times"][99],
            ],
            "Unit": [
                "1/s",
                "m3",
                "-",
                "m/s",
                "W",
                "m3/s",
                "s",
                "W/m3",
                "W/kg",
                "N.m",
                "N.m",
                "x",
                "s",
                "s",
                "s",
            ],
        }
    )
    st.dataframe(
        calc.style.format({"Value": "{:.5g}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.warning(
        "For a defensible final model, replace the surrogate with blade-resolved CAD, mesh-independence testing, MRF/sliding mesh, appropriate laminar or turbulence closure, passive-scalar transport and experimental conductivity validation."
    )
with tabs[6]:
    flat = pd.DataFrame(
        {
            "r_m": m["rr"].ravel(),
            "z_m": m["zz"].ravel(),
            "ur_m_s": m["ur"].ravel(),
            "uz_m_s": m["uz"].ravel(),
            "ut_m_s": m["ut"].ravel(),
            "velocity_m_s": m["vel"].ravel(),
            "shear_1_s": m["shear"].ravel(),
            "stress_Pa": m["tau"].ravel(),
            "k_m2_s2": m["k"].ravel(),
            "epsilon_m2_s3": m["eps"].ravel(),
        }
    )
    st.download_button(
        "Download field data CSV",
        flat.to_csv(index=False).encode(),
        "gmp_mixer_fields.csv",
        "text/csv",
    )
    st.download_button(
        "Download current inputs CSV",
        pd.DataFrame([par]).to_csv(index=False).encode(),
        "gmp_mixer_inputs.csv",
        "text/csv",
    )

# ==================== Mesh independence and OpenFOAM export ====================
st.divider()
st.header("Mesh-independent CFD validation and OpenFOAM case export")
val_tab, of_tab = st.tabs(
    ["Mesh-independence workflow", "OpenFOAM solver setup & export"]
)
with val_tab:
    st.markdown(
        """Use at least three systematically refined meshes with a near-constant refinement ratio. Compare integral and field-based quantities after each case satisfies the same convergence criteria."""
    )
    uploaded = st.file_uploader(
        "Upload mesh-study CSV (columns: mesh, cells, h, power_W, torque_Nm, t95_s, max_velocity_m_s, dead_zone_pct)",
        type="csv",
    )
    if uploaded:
        md = pd.read_csv(uploaded)
    else:
        base_cells = st.number_input(
            "Coarse mesh cells", 10000, 100000000, 250000, 50000, key="base_cells"
        )
        ratio = st.number_input(
            "Linear refinement ratio r", 1.10, 3.00, 1.50, 0.05, key="ref_ratio"
        )
        md = pd.DataFrame(
            {
                "mesh": ["coarse", "medium", "fine"],
                "cells": [
                    base_cells,
                    int(base_cells * ratio**3),
                    int(base_cells * ratio**6),
                ],
                "h": [1, 1 / ratio, 1 / ratio**2],
                "power_W": [m["P"] * 0.91, m["P"] * 0.975, m["P"]],
                "torque_Nm": [m["torque"] * 0.91, m["torque"] * 0.975, m["torque"]],
                "t95_s": [m["times"][95] * 1.13, m["times"][95] * 1.04, m["times"][95]],
                "max_velocity_m_s": [
                    m["vel"].max() * 0.88,
                    m["vel"].max() * 0.97,
                    m["vel"].max(),
                ],
                "dead_zone_pct": [m["dead"] * 1.20, m["dead"] * 1.06, m["dead"]],
            }
        )
        st.caption(
            "Example values are synthetic placeholders. Replace them with completed OpenFOAM results."
        )
    st.dataframe(md, use_container_width=True, hide_index=True)
    metrics = [c for c in md.columns if c not in ["mesh", "cells", "h"]]
    rows = []
    if len(md) >= 3:
        d = md.sort_values("h", ascending=False).iloc[:3]
        h1, h2, h3 = d["h"].to_numpy()
        r21 = h1 / h2
        r32 = h2 / h3
        for q in metrics:
            f1, f2, f3 = d[q].to_numpy(float)
            e21 = f2 - f1
            e32 = f3 - f2
            p = np.nan
            if e21 != 0 and e32 != 0 and e21 / e32 > 0 and abs(r21 - r32) < 0.15 * r21:
                p = abs(np.log(abs(e21 / e32)) / np.log(r21))
            ea = abs((f3 - f2) / f3) * 100 if f3 else np.nan
            gci = (
                1.25 * ea / (r32**p - 1) if np.isfinite(p) and r32 ** p > 1 else np.nan
            )
            rows.append(
                [q, p, ea, gci, "PASS" if np.isfinite(gci) and gci <= 5 else "REVIEW"]
            )
    result = pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Observed order p",
            "Fine-medium change (%)",
            "Fine-grid GCI (%)",
            "Status",
        ],
    )
    st.dataframe(
        result.style.format(precision=3), use_container_width=True, hide_index=True
    )
    st.markdown(
        """**Acceptance workflow**
1. Mesh quality: `checkMesh`; target max non-orthogonality < 65 deg, max skewness < 4 for internal cells, no negative volumes.
2. Near-wall treatment: document wall functions and y+ target. For standard wall functions target y+ about 30-300; for wall-resolved SST target y+ about 1.
3. Iterative convergence: scaled residuals typically <1e-5 for U and <1e-4 for k/epsilon, plus stable power, torque, flow number and probes.
4. Time-step check for transient tracer/sliding mesh: repeat with half deltaT; require t95 change <=5%.
5. Grid convergence: report fine-medium change and Roache GCI for power, torque, t95, maximum velocity and dead-zone fraction.
6. Experimental validation: compare torque/power and conductivity t95 at minimum, nominal and maximum fill. Predefine error bands, commonly +/-10% for power and +/-15% for blend time during development."""
    )
    st.download_button(
        "Download mesh-study template",
        md.to_csv(index=False).encode(),
        "mesh_independence_template.csv",
        "text/csv",
    )

with of_tab:
    of_version = st.selectbox(
        "OpenFOAM family", ["OpenFOAM Foundation v13/v14", "OpenCFD v2312/v2406"]
    )
    solver = st.selectbox(
        "Flow solver",
        ["simpleFoam (steady MRF)", "pimpleFoam (transient MRF/sliding mesh)"],
    )
    turb = st.selectbox("Turbulence model", ["kOmegaSST", "kEpsilon", "laminar"])
    stl_mode = st.checkbox(
        "I will add vessel.stl and impeller.stl to constant/triSurface", value=True
    )
    st.markdown(
        "The exported case is a configurable template. Geometry-specific cellZone and faceZone creation must be checked in ParaView before solving."
    )
    if st.button("Build OpenFOAM case ZIP", type="primary"):
        import tempfile, os, pathlib, zipfile

        root = pathlib.Path(tempfile.mkdtemp()) / "bottomMountMixer"
        for p in ["0", "constant/triSurface", "system", "postProcessing", "validation"]:
            (root / p).mkdir(parents=True, exist_ok=True)
        header = "FoamFile\n{\n    format ascii;\n    class dictionary;\n    object PLACEHOLDER;\n}\n"
        (root / "README.md").write_text(
            f"""# Bottom-mount GMP mixer OpenFOAM template\nSelected: {of_version}; {solver}; {turb}.\n\n1. Put watertight vessel.stl and impeller.stl in constant/triSurface.\n2. Edit snappyHexMeshDict locations and rotating-zone geometry.\n3. Run Allrun.\n4. Verify cellZone rotatingZone in ParaView.\n5. Run steady flow first, then transient scalar injection.\n6. Populate validation/meshStudy.csv and upload it in the Streamlit mesh workflow.\n"""
        )
        (root / "Allrun").write_text(
            """#!/bin/sh\ncd "${0%/*}" || exit 1\nblockMesh\nsurfaceFeatureExtract\nsnappyHexMesh -overwrite\ncheckMesh | tee log.checkMesh\ntopoSet\ncheckMesh | tee log.checkMesh.final\n"""
            + (
                "simpleFoam | tee log.simpleFoam\n"
                if solver.startswith("simple")
                else "pimpleFoam | tee log.pimpleFoam\n"
            )
        )
        (root / "Allclean").write_text(
            "#!/bin/sh\nfoamListTimes -rm\nrm -rf constant/polyMesh log.* processor* postProcessing\n"
        )
        os.chmod(root / "Allrun", 0o755)
        os.chmod(root / "Allclean", 0o755)
        (root / "system/blockMeshDict").write_text(
            """FoamFile{format ascii;class dictionary;object blockMeshDict;}\nscale 1;\nvertices ((-0.5 -0.5 0)(0.5 -0.5 0)(0.5 0.5 0)(-0.5 0.5 0)(-0.5 -0.5 1)(0.5 -0.5 1)(0.5 0.5 1)(-0.5 0.5 1));\nblocks (hex (0 1 2 3 4 5 6 7) (30 30 40) simpleGrading (1 1 1));\nedges(); boundary (background{type patch;faces((0 4 7 3)(1 2 6 5)(0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));}); mergePatchPairs();\n"""
        )
        (root / "system/surfaceFeatureExtractDict").write_text(
            """FoamFile{format ascii;class dictionary;object surfaceFeatureExtractDict;}\nvessel.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }\nimpeller.stl { extractionMethod extractFromSurface; extractFromSurfaceCoeffs { includedAngle 150; } writeObj yes; }\n"""
        )
        (root / "system/snappyHexMeshDict").write_text(
            f"""FoamFile{{format ascii;class dictionary;object snappyHexMeshDict;}}\ncastellatedMesh true; snap true; addLayers false;\ngeometry{{ vessel.stl{{type triSurfaceMesh;name vessel;}} impeller.stl{{type triSurfaceMesh;name impeller;}} rotatingCylinder{{type searchableCylinder;point1 (0 0 {max(C-0.08,0):.4g});point2 (0 0 {min(C+0.08,H):.4g});radius {D*0.65:.4g};}} }}\ncastellatedMeshControls{{maxLocalCells 3000000;maxGlobalCells 6000000;minRefinementCells 10;nCellsBetweenLevels 3;features(("vessel.eMesh" 2)("impeller.eMesh" 3));refinementSurfaces{{vessel{{level(2 3);patchInfo{{type wall;}}}}impeller{{level(3 4);patchInfo{{type wall;}}}}}}refinementRegions{{rotatingCylinder{{mode inside;levels((1E15 3));cellZone rotatingZone;faceZone rotatingZone;cellZoneInside inside;}}}}locationInMesh (0 0 {H*0.7:.4g});allowFreeStandingZoneFaces true;}}\nsnapControls{{nSmoothPatch 5;tolerance 2;nSolveIter 100;nRelaxIter 8;}} addLayersControls{{}} meshQualityControls{{maxNonOrtho 65;maxBoundarySkewness 20;maxInternalSkewness 4;minVol 1e-13;}}\n"""
        )
        (root / "system/topoSetDict").write_text(
            f"""FoamFile{{format ascii;class dictionary;object topoSetDict;}}\nactions({{name rotatingCells;type cellSet;action new;source cylinderToCell;sourceInfo{{p1(0 0 {max(C-0.08,0):.4g});p2(0 0 {min(C+0.08,H):.4g});radius {D*0.65:.4g};}}}}{{name rotatingZone;type cellZoneSet;action new;source setToCellZone;sourceInfo{{set rotatingCells;}}}});\n"""
        )
        omega = 2 * math.pi * rpm / 60
        (root / "constant/MRFProperties").write_text(
            f"""MRF1{{cellZone rotatingZone;active yes;nonRotatingPatches (vessel);origin (0 0 {C});axis (0 0 1);omega {omega:.8g};}}\n"""
        )

        rheo = par.get("rheology", "Newtonian")
        if rheo == "Newtonian":
            transport = f'transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {par["mu"]/par["rho"]:.8g};'
        elif rheo == "Power law":
            transport = f'transportModel powerLaw;\npowerLawCoeffs{{nuMin 1e-09;nuMax 1e3;k {par["K"]/par["rho"]:.8g};n {par["n_index"]:.8g};}}'
        elif rheo == "Herschel-Bulkley":
            transport = f'transportModel HerschelBulkley;\nHerschelBulkleyCoeffs{{nu0 {par["tau0"]/par["rho"]:.8g};k {par["K"]/par["rho"]:.8g};n {par["n_index"]:.8g};nuMin 1e-09;nuMax 1e3;}}'
        elif rheo == "Bingham plastic":
            transport = f'transportModel HerschelBulkley;\nHerschelBulkleyCoeffs{{nu0 {par["tau0"]/par["rho"]:.8g};k {par["mu_plastic"]/par["rho"]:.8g};n 1;nuMin 1e-09;nuMax 1e3;}}'
        else:
            transport = f'transportModel BirdCarreau;\nBirdCarreauCoeffs{{nu0 {par["mu0"]/par["rho"]:.8g};nuInf {par["muinf"]/par["rho"]:.8g};k {par["lambda_c"]:.8g};n {par["n_c"]:.8g};}}'
        (root / "constant/transportProperties").write_text(
            "FoamFile{format ascii;class dictionary;object transportProperties;}\n"
            + transport
            + "\n"
        )
        simType = "laminar" if turb == "laminar" else "RAS"
        ras = (
            ""
            if turb == "laminar"
            else f"RAS{{model {turb};turbulence on;printCoeffs on;}}"
        )
        (root / "constant/turbulenceProperties").write_text(
            f"""FoamFile{{format ascii;class dictionary;object turbulenceProperties;}}\nsimulationType {simType};\n{ras}\n"""
        )
        for name, dim, val, bcs in [
            (
                "U",
                "[0 1 -1 0 0 0 0]",
                "uniform (0 0 0)",
                "vessel{type noSlip;} impeller{type noSlip;}",
            ),
            (
                "p",
                "[0 2 -2 0 0 0 0]",
                "uniform 0",
                "vessel{type zeroGradient;} impeller{type zeroGradient;}",
            ),
            (
                "k",
                "[0 2 -2 0 0 0 0]",
                "uniform 1e-4",
                "vessel{type kqRWallFunction;value uniform 1e-4;} impeller{type kqRWallFunction;value uniform 1e-4;}",
            ),
            (
                "epsilon",
                "[0 2 -3 0 0 0 0]",
                "uniform 1e-4",
                "vessel{type epsilonWallFunction;value uniform 1e-4;} impeller{type epsilonWallFunction;value uniform 1e-4;}",
            ),
            (
                "omega",
                "[0 0 -1 0 0 0 0]",
                "uniform 1",
                "vessel{type omegaWallFunction;value uniform 1;} impeller{type omegaWallFunction;value uniform 1;}",
            ),
            (
                "s",
                "[0 0 0 0 0 0 0]",
                "uniform 0",
                "vessel{type zeroGradient;} impeller{type zeroGradient;}",
            ),
        ]:
            (root / f"0/{name}").write_text(
                f"""FoamFile{{format ascii;class vol{name=='U' and 'Vector' or 'Scalar'}Field;object {name};}}\ndimensions {dim};internalField {val};boundaryField{{{bcs}}}\n"""
            )
        app = "simpleFoam" if solver.startswith("simple") else "pimpleFoam"
        (root / "system/controlDict").write_text(
            f"""FoamFile{{format ascii;class dictionary;object controlDict;}}\napplication {app};startFrom startTime;startTime 0;stopAt endTime;endTime {200 if app=='simpleFoam' else 20};deltaT {0.001 if app=='pimpleFoam' else 1};writeControl timeStep;writeInterval {100 if app=='simpleFoam' else 500};purgeWrite 0;writeFormat binary;runTimeModifiable true;\nfunctions{{forces{{type forces;libs ("libforces.so");patches (impeller);rho rhoInf;rhoInf {rho};CofR (0 0 {C});writeControl timeStep;writeInterval 10;}} fieldMinMax{{type fieldMinMax;libs ("libfieldFunctionObjects.so");fields (U k epsilon omega s);writeControl timeStep;writeInterval 10;}} scalarMixing{{type scalarTransport;libs ("libsolverFunctionObjects.so");field s;phi phi;D 1e-9;resetOnStartUp no;}}}}\n"""
        )
        (root / "system/fvSchemes").write_text(
            """FoamFile{format ascii;class dictionary;object fvSchemes;}\nddtSchemes{default steadyState;} gradSchemes{default cellLimited Gauss linear 1;} divSchemes{default none;div(phi,U) bounded Gauss linearUpwind grad(U);div(phi,k) bounded Gauss upwind;div(phi,epsilon) bounded Gauss upwind;div(phi,omega) bounded Gauss upwind;div(phi,s) bounded Gauss upwind;} laplacianSchemes{default Gauss linear limited 0.5;} interpolationSchemes{default linear;} snGradSchemes{default limited 0.5;} wallDist{method meshWave;}\n"""
        )
        if app == "pimpleFoam":
            txt = (
                (root / "system/fvSchemes")
                .read_text()
                .replace("default steadyState", "default backward")
            )
            (root / "system/fvSchemes").write_text(txt)
        (root / "system/fvSolution").write_text(
            """FoamFile{format ascii;class dictionary;object fvSolution;}\nsolvers{p{solver GAMG;tolerance 1e-8;relTol 0.05;smoother GaussSeidel;} U{solver smoothSolver;smoother symGaussSeidel;tolerance 1e-7;relTol 0.05;} "(k|epsilon|omega|s)"{solver smoothSolver;smoother symGaussSeidel;tolerance 1e-7;relTol 0.05;}}\nSIMPLE{nNonOrthogonalCorrectors 2;consistent yes;residualControl{p 1e-5;U 1e-6;"(k|epsilon|omega)" 1e-5;}}\nPIMPLE{nOuterCorrectors 2;nCorrectors 2;nNonOrthogonalCorrectors 1;momentumPredictor yes;}\nrelaxationFactors{fields{p 0.3;}equations{U 0.7;"(k|epsilon|omega)" 0.7;}}\n"""
        )
        (root / "validation/meshStudy.csv").write_text(
            "mesh,cells,h,power_W,torque_Nm,t95_s,max_velocity_m_s,dead_zone_pct\ncoarse,,,,,,,\nmedium,,,,,,,\nfine,,,,,,,\n"
        )
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in root.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(root.parent))
        st.download_button(
            "Download OpenFOAM case ZIP",
            bio.getvalue(),
            "bottomMountMixer_OpenFOAM.zip",
            "application/zip",
        )

st.divider()
st.header("Automated ParaView post-processing and hybrid PINN")
pvtab, pinntab = st.tabs(["ParaView automation", "PINN workflow"])
with pvtab:
    st.markdown(
        """The package includes `postprocess.py`, a headless `pvpython`/`pvbatch` script. It reads the latest OpenFOAM time, generates orthogonal velocity slices, 3D pressure/turbulence/scalar images, streamlines, a ParaView state and a point-cloud CSV for PINN training."""
    )
    st.code(
        "pvbatch postprocess.py --case /path/to/bottomMountMixer\n# Windows single process:\npvpython postprocess.py --case C:/cases/bottomMountMixer",
        language="bash",
    )
    st.download_button(
        "Download ParaView script",
        open("postprocess.py", "rb").read(),
        "postprocess.py",
        "text/x-python",
    )
with pinntab:
    st.markdown(
        r"""The implemented model is a **hybrid CFD-PINN surrogate**. Inputs are $(x,y,z,N,H,\mu)$ and outputs are $(u,v,w,p,k)$. Training combines supervised OpenFOAM field error with steady incompressible continuity and momentum residual penalties. Use the model to interpolate inside the trained operating envelope, never to replace held-out CFD and experimental validation."""
    )
    st.warning(
        "The lightweight trainer omits second-derivative viscous and full RANS closure residuals. For final research use, replace it with PhysicsNeMo Navier-Stokes plus k-omega residuals or adapt the CFD-PINN-StirredTank repository."
    )
    traincsv = st.file_uploader(
        "Upload ParaView PINN training CSV", type="csv", key="pinncsv"
    )
    epochs = st.number_input("Training epochs", 100, 100000, 5000, 100)
    if traincsv:
        preview = pd.read_csv(traincsv, nrows=1000)
        st.dataframe(preview.head(), use_container_width=True)
        st.info(
            "Save the uploaded CSV and run: python pinn_train.py --csv pinn_training.csv --out pinn_model --epochs "
            + str(epochs)
        )
    st.download_button(
        "Download PINN trainer",
        open("pinn_train.py", "rb").read(),
        "pinn_train.py",
        "text/x-python",
    )
    st.markdown(
        """**Required validation split**
- Hold out entire RPM/fill-height combinations, not random spatial points.
- Report component-wise NRMSE and R² for Ux, Uy, Uz, p and k.
- Check continuity residual and momentum residual distributions.
- Recalculate power, torque, dead-zone fraction and t95 from PINN fields.
- Recommended screening gates: velocity NRMSE <=10%, power/torque error <=10%, t95 error <=15%, and no systematic wall/impeller bias."""
    )

st.divider()
st.header("PINN inference and uncertainty quantification")
inf1, inf2 = st.tabs(["Run inference", "RANS residuals and UQ basis"])
with inf1:
    model_file = st.file_uploader(
        "Upload trained pinn.pt", type=["pt"], key="infermodel"
    )
    input_file = st.file_uploader(
        "Upload inference points CSV", type=["csv"], key="inferpoints"
    )
    nmc = st.slider("Monte Carlo dropout samples", 20, 500, 100, 10)
    if model_file and input_file:
        import tempfile
        from pathlib import Path
        from pinn_infer import infer

        td = tempfile.mkdtemp()
        mp = Path(td) / "pinn.pt"
        mp.write_bytes(model_file.getvalue())
        idf = pd.read_csv(input_file)
        try:
            pred = infer(mp, idf, nmc)
            st.success(
                f"Inference completed for {len(pred):,} points using {nmc} stochastic passes."
            )
            st.dataframe(pred.head(100), use_container_width=True)
            st.download_button(
                "Download PINN predictions with 95% intervals",
                pred.to_csv(index=False).encode(),
                "pinn_predictions_uq.csv",
                "text/csv",
            )
            fields = [c[:-5] for c in pred.columns if c.endswith("_mean")]
            fld = st.selectbox("Prediction field", fields)
            if {"x", "z", fld + "_mean"}.issubset(pred.columns):
                import plotly.express as px

                st.plotly_chart(
                    px.scatter(
                        pred,
                        x="x",
                        y="z",
                        color=fld + "_mean",
                        title=f"{fld} posterior mean",
                    ),
                    use_container_width=True,
                )
                st.plotly_chart(
                    px.scatter(
                        pred,
                        x="x",
                        y="z",
                        color=fld + "_sd",
                        title=f"{fld} epistemic standard deviation",
                    ),
                    use_container_width=True,
                )
        except Exception as e:
            st.error(str(e))
with inf2:
    st.markdown(
        r"""### Implemented steady incompressible RANS $k$-$\omega$ residuals
Continuity: $R_c=\nabla\cdot\mathbf{u}$. Momentum: $R_m=(\mathbf{u}\cdot\nabla)\mathbf{u}+\nabla p/\rho-\nabla\cdot[(\nu+\nu_t)\nabla\mathbf{u}]$, with $\nu_t=k/\omega$.

Turbulent kinetic energy: $R_k=\mathbf{u}\cdot\nabla k-P_k+\beta^*k\omega-\nabla\cdot[(\nu+\sigma_k\nu_t)\nabla k]$.

Specific dissipation: $R_\omega=\mathbf{u}\cdot\nabla\omega-\alpha(\omega/k)P_k+\beta\omega^2-\nabla\cdot[(\nu+\sigma_\omega\nu_t)\nabla\omega]$.

$P_k=2\nu_tS:S$. Defaults: $\beta^*=0.09$, $\alpha=0.52$, $\beta=0.075$, and $\sigma_k=\sigma_\omega=0.5$. This is baseline $k$-$\omega$, not full SST because blending functions, cross diffusion, production limiter and wall-distance treatment are not included.

### Uncertainty quantification
Monte Carlo dropout estimates epistemic uncertainty. The app reports mean, standard deviation and 2.5/97.5 percentiles. It does not include CFD discretization, turbulence-model-form, geometry or measurement uncertainty.

Calibrate coverage on held-out RPM/fill cases. Target 90-95% empirical coverage for nominal 95% intervals and flag extrapolation or predictive coefficient of variation above a predefined limit such as 15%."""
    )
    st.download_button(
        "Download inference module",
        open("pinn_infer.py", "rb").read(),
        "pinn_infer.py",
        "text/x-python",
    )

st.divider()
st.header("Automated PINN acceptance validation")
pred_file = st.file_uploader("Predictions with UQ CSV", type="csv", key="valpred")
truth_file = st.file_uploader(
    "Held-out CFD or experimental truth CSV", type="csv", key="valtruth"
)
near_threshold = st.number_input(
    "Near-wall and impeller distance threshold (m)", 0.0001, 1.0, 0.02, 0.005
)
if pred_file and truth_file:
    from pinn_validate import validate

    pr = pd.read_csv(pred_file)
    tr = pd.read_csv(truth_file)
    if len(pr) != len(tr):
        st.error(
            "Prediction and truth files must have identical row count and point order."
        )
    else:
        report = validate(pr, tr, near_threshold)
        st.dataframe(report, hide_index=True, use_container_width=True)
        if bool(report["Pass"].all()):
            st.success("PINN ACCEPTED for this validated operating envelope.")
        else:
            st.error("PINN NOT ACCEPTED. One or more mandatory gates failed.")
        st.download_button(
            "Download acceptance report",
            report.to_csv(index=False).encode(),
            "pinn_acceptance_report.csv",
            "text/csv",
        )
        st.markdown(
            """Checks enforced: velocity NRMSE ≤10%; power and torque errors ≤10%; t95 error ≤15%; empirical 95% interval coverage 90–95%; predictive CV ≤15%; no points outside training RPM, fill-height or viscosity bounds; and near-wall/impeller velocity bias ≥−5%. Include `wall_distance` and `impeller_distance` in truth data to activate spatial bias tests."""
        )
