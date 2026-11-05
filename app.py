import io
import json
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq
from scipy.special import factorial
import streamlit as st

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from thermo.chemical import Chemical
    HAS_THERMO = True
except ImportError:
    HAS_THERMO = False

st.set_page_config(
    page_title="Level-5 Multiphysics Mixing & Scale-Up Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# MODULE 1: N-COMPONENT MULTI-SOLVENT ENGINE (PUBCHEM API + DIPPR/NIST + MANUAL)
# ==============================================================================
class MultiSolventPropertyEngine:
    """
    Evaluates pure solvents, arbitrary N-component multi-solvent mixtures
    (Grunberg-Nissan viscosity matrix, Li thermal conductivity, Winterfeld surface
    tension, Raoult-Activity bubble point), live PubChem API lookups, and manual overrides.
    """
    DIPPR_DATABASE = {
        "Water": {
            "cas": "7732-18-5", "mw": 18.015, "rho_25": 997.0, "beta_rho": 0.26,
            "mu_25_cP": 0.890, "E_mu_R": 1850.0, "Cp": 4184.0, "k_f": 0.606,
            "sigma_mN": 72.0, "bp_C": 100.0, "antoine": (8.07131, 1730.63, 233.426), "polarity": "aqueous"
        },
        "Methanol": {
            "cas": "67-56-1", "mw": 32.04, "rho_25": 786.6, "beta_rho": 0.94,
            "mu_25_cP": 0.544, "E_mu_R": 1260.0, "Cp": 2531.0, "k_f": 0.200,
            "sigma_mN": 22.1, "bp_C": 64.7, "antoine": (8.08097, 1582.27, 239.726), "polarity": "protic"
        },
        "Ethanol": {
            "cas": "64-17-5", "mw": 46.07, "rho_25": 785.2, "beta_rho": 0.86,
            "mu_25_cP": 1.074, "E_mu_R": 1680.0, "Cp": 2440.0, "k_f": 0.167,
            "sigma_mN": 22.0, "bp_C": 78.3, "antoine": (8.11220, 1592.86, 226.184), "polarity": "protic"
        },
        "Isopropanol (IPA)": {
            "cas": "67-63-0", "mw": 60.10, "rho_25": 781.2, "beta_rho": 0.83,
            "mu_25_cP": 2.038, "E_mu_R": 2450.0, "Cp": 2680.0, "k_f": 0.135,
            "sigma_mN": 21.0, "bp_C": 82.3, "antoine": (8.11778, 1580.92, 219.610), "polarity": "protic"
        },
        "Acetone": {
            "cas": "67-64-1", "mw": 58.08, "rho_25": 784.5, "beta_rho": 1.12,
            "mu_25_cP": 0.306, "E_mu_R": 860.0, "Cp": 2150.0, "k_f": 0.159,
            "sigma_mN": 23.0, "bp_C": 56.1, "antoine": (7.11714, 1210.60, 229.664), "polarity": "aprotic"
        },
        "Acetonitrile (ACN)": {
            "cas": "75-05-8", "mw": 41.05, "rho_25": 776.6, "beta_rho": 1.08,
            "mu_25_cP": 0.341, "E_mu_R": 890.0, "Cp": 2230.0, "k_f": 0.188,
            "sigma_mN": 28.7, "bp_C": 81.6, "antoine": (7.11988, 1314.40, 230.000), "polarity": "aprotic"
        },
        "Ethyl Acetate": {
            "cas": "141-78-6", "mw": 88.11, "rho_25": 894.5, "beta_rho": 1.18,
            "mu_25_cP": 0.423, "E_mu_R": 1020.0, "Cp": 1935.0, "k_f": 0.144,
            "sigma_mN": 23.4, "bp_C": 77.1, "antoine": (7.10179, 1244.95, 217.881), "polarity": "aprotic"
        },
        "Dichloromethane (DCM)": {
            "cas": "75-09-2", "mw": 84.93, "rho_25": 1316.8, "beta_rho": 1.82,
            "mu_25_cP": 0.413, "E_mu_R": 850.0, "Cp": 1188.0, "k_f": 0.139,
            "sigma_mN": 27.2, "bp_C": 39.6, "antoine": (7.08030, 1138.91, 231.450), "polarity": "nonpolar"
        },
        "Tetrahydrofuran (THF)": {
            "cas": "109-99-9", "mw": 72.11, "rho_25": 883.3, "beta_rho": 1.05,
            "mu_25_cP": 0.456, "E_mu_R": 980.0, "Cp": 1765.0, "k_f": 0.120,
            "sigma_mN": 26.4, "bp_C": 66.0, "antoine": (6.99515, 1202.29, 226.254), "polarity": "aprotic"
        },
        "Toluene": {
            "cas": "108-88-3", "mw": 92.14, "rho_25": 862.3, "beta_rho": 0.93,
            "mu_25_cP": 0.560, "E_mu_R": 1070.0, "Cp": 1700.0, "k_f": 0.131,
            "sigma_mN": 27.9, "bp_C": 110.6, "antoine": (6.95464, 1343.94, 219.387), "polarity": "nonpolar"
        },
        "Dimethylformamide (DMF)": {
            "cas": "68-12-2", "mw": 73.09, "rho_25": 944.0, "beta_rho": 0.95,
            "mu_25_cP": 0.794, "E_mu_R": 1180.0, "Cp": 2060.0, "k_f": 0.184,
            "sigma_mN": 36.4, "bp_C": 153.0, "antoine": (7.10850, 1537.78, 210.390), "polarity": "aprotic"
        },
        "Dimethyl Sulfoxide (DMSO)": {
            "cas": "67-68-5", "mw": 78.13, "rho_25": 1095.5, "beta_rho": 1.00,
            "mu_25_cP": 1.991, "E_mu_R": 1720.0, "Cp": 1955.0, "k_f": 0.200,
            "sigma_mN": 42.9, "bp_C": 189.0, "antoine": (7.23039, 1733.00, 199.160), "polarity": "aprotic"
        },
    }

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_pubchem_live(query_str: str) -> dict:
        clean_q = query_str.split(" (")[0].strip()
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{urllib.parse.quote(clean_q)}/"
            f"property/IUPACName,MolecularWeight,MolecularFormula,XLogP/JSON"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MixingDigitalTwin/3.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                p = data["PropertyTable"]["Properties"][0]
                return {
                    "online": True,
                    "source_badge": f"PubChem Live Verified (CID {p.get('CID')})",
                    "cid": p.get("CID"),
                    "formula": p.get("MolecularFormula", "N/A"),
                    "mw": float(p.get("MolecularWeight", 60.0)),
                    "xlogp": float(p.get("XLogP", 0.0)),
                    "iupac": p.get("IUPACName", clean_q),
                }
        except Exception:
            return {
                "online": False,
                "source_badge": "Built-in DIPPR 801 / NIST Standard Table",
                "cid": None,
                "formula": "Verified Reference",
                "mw": 60.0,
                "xlogp": 0.0,
                "iupac": clean_q,
            }

    @classmethod
    def evaluate_pure(cls, name: str, T_C: float) -> dict:
        T_K = T_C + 273.15
        if HAS_THERMO and name not in cls.DIPPR_DATABASE:
            try:
                c = Chemical(name.split(" (")[0], T=T_K, P=101325.0)
                if c.rho and c.mu:
                    return {
                        "name": name, "cas": c.CAS, "mw": float(c.MW),
                        "rho": float(c.rho), "mu_cP": float(c.mu) * 1000.0,
                        "E_mu_R": 1500.0, "Cp": float(c.Cp or 2000.0),
                        "k_f": float(c.k or 0.16), "sigma_mN": float(c.sigma or 0.025) * 1000.0,
                        "P_vap_Pa": float(c.Psat or 3000.0), "bp_C": float(c.Tb or 350.0) - 273.15,
                        "antoine": (7.40, 1450.0, 225.0), "polarity": "aprotic"
                    }
            except Exception:
                pass

        s = cls.DIPPR_DATABASE.get(name, cls.DIPPR_DATABASE["Isopropanol (IPA)"])
        rho_T = s["rho_25"] - s["beta_rho"] * (T_C - 25.0)
        mu_T_cP = s["mu_25_cP"] * np.exp(s["E_mu_R"] * (1.0 / T_K - 1.0 / 298.15))
        A, B, C = s["antoine"]
        P_vap_Pa = 133.322 * (10.0 ** (A - B / (T_C + C)))
        return {
            "name": name, "cas": s["cas"], "mw": s["mw"],
            "rho": rho_T, "mu_cP": mu_T_cP, "E_mu_R": s["E_mu_R"],
            "Cp": s["Cp"], "k_f": s["k_f"],
            "sigma_mN": max(5.0, s["sigma_mN"] - 0.11 * (T_C - 25.0)),
            "P_vap_Pa": P_vap_Pa, "bp_C": s["bp_C"],
            "antoine": s["antoine"], "polarity": s["polarity"]
        }

    @classmethod
    def pair_grunberg_nissan_Gij(cls, pol_i: str, pol_j: str, scale_factor: float = 1.0) -> float:
        """Estimates binary Grunberg-Nissan viscosity interaction parameter G_ij from polarity classes."""
        pair = {pol_i, pol_j}
        if pair == {"aqueous", "protic"}:
            return 0.95 * scale_factor   # Strong H-bonding viscosity maximum (Water + Alcohols)
        if pair == {"aqueous", "aprotic"}:
            return 0.55 * scale_factor   # Water + DMSO / DMF / Acetone комплексация
        if pair == {"protic", "aprotic"}:
            return 0.15 * scale_factor
        if "nonpolar" in pair and ("aqueous" in pair or "protic" in pair):
            return -0.25 * scale_factor  # Structure-breaking negative deviation
        return 0.05 * scale_factor

    @classmethod
    def evaluate_multisolvent_mixture(
        cls, component_weights: dict, T_C: float, G_scale: float = 1.0
    ) -> dict:
        """
        Evaluates an arbitrary N-component solvent blend using:
          - Amagat Volume Additivity for Density
          - N-Component Grunberg-Nissan Matrix for Viscosity
          - Li Volume-Fraction Harmonic Rule for Thermal Conductivity
          - Winterfeld-Scriven-Davis Rule for Surface Tension
          - Raoult-Activity Law for Vapor Pressure & Bubble Point
        """
        names = list(component_weights.keys())
        raw_w = np.array([max(float(component_weights[k]), 0.0) for k in names], dtype=float)
        if np.sum(raw_w) <= 1e-12:
            raw_w = np.ones_like(raw_w)
        w = raw_w / np.sum(raw_w)
        n_comp = len(names)

        pure_list = [cls.evaluate_pure(nm, T_C) for nm in names]
        mws = np.array([p["mw"] for p in pure_list])
        rhos = np.array([p["rho"] for p in pure_list])
        mus = np.array([p["mu_cP"] for p in pure_list])
        cps = np.array([p["Cp"] for p in pure_list])
        kfs = np.array([p["k_f"] for p in pure_list])
        sigs = np.array([p["sigma_mN"] for p in pure_list])
        pvaps = np.array([p["P_vap_Pa"] for p in pure_list])

        # Mole fractions (x_i) and volume fractions (phi_i)
        moles = w / mws
        x = moles / np.sum(moles)
        vols = w / rhos
        phi = vols / np.sum(vols)

        # 1. Mixture Density (kg/m3)
        rho_mix = 1.0 / np.sum(w / rhos)

        # 2. N-Component Grunberg-Nissan Viscosity (cP)
        ln_mu = np.sum(x * np.log(mus))
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                G_ij = cls.pair_grunberg_nissan_Gij(
                    pure_list[i]["polarity"], pure_list[j]["polarity"], G_scale
                )
                ln_mu += x[i] * x[j] * G_ij
        mu_mix_cP = float(np.exp(ln_mu))

        # 3. Li Volume-Fraction Harmonic Thermal Conductivity (W/m·K)
        k_mix = 0.0
        sigma_mix = 0.0
        for i in range(n_comp):
            for j in range(n_comp):
                k_ij = (2.0 * kfs[i] * kfs[j]) / max(kfs[i] + kfs[j], 1e-9)
                k_mix += phi[i] * phi[j] * k_ij
                sigma_mix += phi[i] * phi[j] * np.sqrt(sigs[i] * sigs[j])

        # 4. Mass-Weighted Specific Heat Capacity (J/kg·K)
        Cp_mix = float(np.sum(w * cps))

        # 5. Multi-Component Vapor Pressure Function at any Temperature T_eval (°C)
        def calc_pvap_mix_at_T(T_eval_C: float) -> float:
            pv_tot = 0.0
            for idx, p in enumerate(pure_list):
                A, B, C = p["antoine"]
                pv_i = 133.322 * (10.0 ** (A - B / (T_eval_C + C)))
                pv_tot += x[idx] * pv_i
            return float(pv_tot)

        P_vap_mix = calc_pvap_mix_at_T(T_C)

        # Solve normal bubble point (at 101,325 Pa) via Brent's method
        try:
            bp_mix_C = brentq(lambda t_c: calc_pvap_mix_at_T(t_c) - 101325.0, -20.0, 260.0, xtol=0.05)
        except ValueError:
            bp_mix_C = float(np.sum(x * np.array([p["bp_C"] for p in pure_list])))

        label_parts = [f"{w[idx]*100:.0f}% {names[idx].split(' (')[0]}" for idx in range(n_comp) if w[idx] > 0.005]
        composition_table = pd.DataFrame({
            "Solvent Component": names,
            "CAS No.": [p["cas"] for p in pure_list],
            "Mass Frac (wt%)": np.round(w * 100.0, 2),
            "Mole Frac (mol%)": np.round(x * 100.0, 2),
            "Vol Frac (vol%)": np.round(phi * 100.0, 2),
            "Pure ρ (kg/m³)": np.round(rhos, 1),
            "Pure μ (cP)": np.round(mus, 3),
            "Pure Bp (°C)": np.round([p["bp_C"] for p in pure_list], 1),
        })

        return {
            "name": " + ".join(label_parts),
            "cas": f"{n_comp}-Solvent Blend",
            "mw": float(np.sum(x * mws)),
            "rho": float(rho_mix),
            "mu_cP": mu_mix_cP,
            "Cp": Cp_mix,
            "k_f": float(k_mix),
            "sigma_mN": float(sigma_mix),
            "P_vap_Pa": P_vap_mix,
            "bp_C": float(bp_mix_C),
            "pvap_fn": calc_pvap_mix_at_T,
            "composition_df": composition_table,
        }


# ==============================================================================
# MODULE 2: PARAMETRIC VESSEL, BAFFLE & AGITATOR GEOMETRY + 3D FLOW ANIMATION
# ==============================================================================
DISH_SHAPES = {
    "2:1 Ellipsoidal":          {"h_ratio": 0.250, "v_coeff": 0.1309, "a_coeff": 1.0840, "f_dish": 0.92},
    "Hemispherical":            {"h_ratio": 0.500, "v_coeff": 0.2618, "a_coeff": 1.5708, "f_dish": 0.86},
    "Torispherical (Klöpper)":  {"h_ratio": 0.194, "v_coeff": 0.0847, "a_coeff": 0.9300, "f_dish": 1.00},
    "Conical (45°)":            {"h_ratio": 0.500, "v_coeff": 0.1309, "a_coeff": 1.1107, "f_dish": 1.35},
    "Flat Bottom":              {"h_ratio": 0.000, "v_coeff": 0.0000, "a_coeff": 0.7854, "f_dish": 1.22},
}

IMPELLER_CATALOG = {
    "Wide-Blade Hydrofoil (A315 / Viscoprop)": {
        "Kp": 38.0, "Kt": 1.6, "Np_inf": 0.80, "Nq_inf": 0.73, "CQ": 120.0,
        "Ks": 9.5, "S0": 1.35, "alpha": 1.15, "Phi_eps": 3.2,
        "nb_ref": 4, "pitch_ref": 42.0, "wd_ref": 0.24
    },
    "Narrow-Blade Hydrofoil (A310)": {
        "Kp": 32.0, "Kt": 1.1, "Np_inf": 0.30, "Nq_inf": 0.56, "CQ": 110.0,
        "Ks": 8.5, "S0": 1.42, "alpha": 1.15, "Phi_eps": 2.4,
        "nb_ref": 3, "pitch_ref": 34.0, "wd_ref": 0.14
    },
    "Pitched Blade Turbine (45° 4-Blade)": {
        "Kp": 42.0, "Kt": 2.2, "Np_inf": 1.37, "Nq_inf": 0.79, "CQ": 150.0,
        "Ks": 11.0, "S0": 1.62, "alpha": 1.20, "Phi_eps": 5.5,
        "nb_ref": 4, "pitch_ref": 45.0, "wd_ref": 0.20
    },
    "Rushton Turbine (6-Blade Disc)": {
        "Kp": 72.0, "Kt": 3.0, "Np_inf": 5.00, "Nq_inf": 0.72, "CQ": 180.0,
        "Ks": 11.5, "S0": 1.50, "alpha": 1.45, "Phi_eps": 12.0,
        "nb_ref": 6, "pitch_ref": 90.0, "wd_ref": 0.20
    },
    "Retreat-Curve Impeller (RCI Glass-Lined)": {
        "Kp": 36.0, "Kt": 1.8, "Np_inf": 0.70, "Nq_inf": 0.40, "CQ": 200.0,
        "Ks": 10.5, "S0": 2.10, "alpha": 1.30, "Phi_eps": 4.5,
        "nb_ref": 3, "pitch_ref": 65.0, "wd_ref": 0.18
    },
    "Anchor Impeller (Close-Clearance)": {
        "Kp": 215.0, "Kt": 2.5, "Np_inf": 0.40, "Nq_inf": 0.20, "CQ": 250.0,
        "Ks": 22.0, "S0": 2.80, "alpha": 1.50, "Phi_eps": 1.8,
        "nb_ref": 2, "pitch_ref": 90.0, "wd_ref": 0.12
    },
}

ROTOR_STATOR_CATALOG = {
    "Fine-Screen Emulsor Head": {
        "Kp_rs": 155.0, "Np0_rs": 0.55, "Np1_rs": 8.0, "Nq_rs": 0.18, "Href_ratio": 0.30, "k_pass": 0.28
    },
    "Slotted Disintegrating Head": {
        "Kp_rs": 125.0, "Np0_rs": 0.45, "Np1_rs": 7.2, "Nq_rs": 0.24, "Href_ratio": 0.35, "k_pass": 0.22
    },
    "Axial Square-Hole High-Shear Head": {
        "Kp_rs": 140.0, "Np0_rs": 0.50, "Np1_rs": 7.6, "Nq_rs": 0.21, "Href_ratio": 0.32, "k_pass": 0.25
    },
}


def compute_parametric_geometry(
    V_L: float, T_m: float, dish_name: str, n_baffles: int, Wb_over_T: float,
    imp_name: str, D_ag: float, n_tiers: int, spacing_over_D: float, C1_over_T: float,
    n_blades: int, pitch_deg: float, Wimp_over_D: float
) -> dict:
    V_m3 = V_L / 1000.0
    dish = DISH_SHAPES[dish_name]
    imp = IMPELLER_CATALOG[imp_name]

    h_dish = dish["h_ratio"] * T_m
    V_dish = min(dish["v_coeff"] * (T_m ** 3), 0.85 * V_m3)
    H_cyl = max((V_m3 - V_dish) / (0.25 * np.pi * (T_m ** 2)), 0.05)
    H_total = H_cyl + h_dish
    A_jacket = np.pi * T_m * H_cyl + dish["a_coeff"] * (T_m ** 2)

    omega_b = n_baffles * Wb_over_T
    phi_P_baf = 0.28 + 0.72 * (1.0 - np.exp(-3.45 * (omega_b / 0.3333)))
    phi_Q_baf = 0.45 + 0.55 * np.tanh(2.60 * (omega_b / 0.3333))

    pitch_rad = np.deg2rad(pitch_deg)
    pitch_ref_rad = np.deg2rad(imp["pitch_ref"])
    sin_ratio = np.sin(pitch_rad) / max(np.sin(pitch_ref_rad), 1e-4)

    phi_P_blades = (n_blades / imp["nb_ref"]) ** 0.75
    phi_Q_blades = (n_blades / imp["nb_ref"]) ** 0.45
    phi_P_pitch = sin_ratio ** 2.15
    phi_Q_pitch = sin_ratio ** 0.85
    phi_P_width = (Wimp_over_D / imp["wd_ref"]) ** 1.15
    phi_Q_width = (Wimp_over_D / imp["wd_ref"]) ** 0.65
    phi_P_clear = (C1_over_T / 0.25) ** (-0.18 * np.cos(pitch_rad))

    f_tier = 1.0 - 0.65 * np.exp(-1.15 * spacing_over_D) if n_tiers > 1 else 1.0
    tier_P_mult = 1.0 + (n_tiers - 1) * f_tier
    tier_Q_mult = 1.0 + 0.75 * (n_tiers - 1) * f_tier

    Np_inf_single = imp["Np_inf"] * phi_P_baf * phi_P_blades * phi_P_pitch * phi_P_width * phi_P_clear
    Nq_inf_single = imp["Nq_inf"] * phi_Q_baf * phi_Q_blades * phi_Q_pitch * phi_Q_width
    Ks_geom = (
        imp["Ks"]
        * ((n_blades / imp["nb_ref"]) ** 0.25)
        * (sin_ratio ** 0.40)
        * ((Wimp_over_D / imp["wd_ref"]) ** 0.30)
    )
    S_geom = (
        dish["f_dish"] * imp["S0"] * ((T_m / D_ag) ** imp["alpha"])
        * np.exp(0.68 * (C1_over_T - 0.25))
        * ((phi_Q_blades * phi_Q_width) ** -0.45)
    )

    return {
        "V_m3": V_m3, "T_m": T_m, "D_ag": D_ag, "D_over_T": D_ag / T_m,
        "h_dish": h_dish, "H_cyl": H_cyl, "H_total": H_total, "H_over_T": H_total / T_m,
        "A_jacket": A_jacket, "f_tier": f_tier, "tier_P_mult": tier_P_mult, "tier_Q_mult": tier_Q_mult,
        "Kp": imp["Kp"] * phi_P_blades * phi_P_width,
        "Kt": imp["Kt"] * phi_P_baf,
        "Np_inf_single": Np_inf_single,
        "Nq_inf_single": Nq_inf_single,
        "CQ": imp["CQ"],
        "Ks_geom": Ks_geom,
        "S_geom": S_geom,
        "Phi_eps": imp["Phi_eps"],
    }


def evaluate_3d_velocity_field(
    r: np.ndarray, theta: np.ndarray, z: np.ndarray,
    geom: dict, comm_state: dict, n_tiers: int, spacing_over_D: float,
    C1_over_T: float, pitch_deg: float, n_baffles: int, Wb_over_T: float
):
    """
    Evaluates the 3D cylindrical velocity components (u_r, u_theta, u_z) and local
    speed magnitude |u| (m/s) at particle positions (r, theta, z), accounting for:
      - Axial floor-sweeping vs Radial figure-8 loops (governed by pitch angle theta_b)
      - Multi-tier circulation cells
      - Tangential swirl (governed by baffle solidity)
      - Yield-stress cavern boundary attenuation (Dc / T)
      - Bottom Rotor-Stator suction intake and radial slot jet
    """
    R = geom["T_m"] / 2.0
    H_tot = geom["H_total"]
    R_ag = geom["D_ag"] / 2.0
    pitch_rad = np.deg2rad(pitch_deg)
    ax_weight = float(np.cos(pitch_rad))
    rad_weight = float(np.sin(pitch_rad))

    V_circ = max(comm_state["v_bulk"] * (geom["T_m"] / geom["D_ag"]) * 2.2, 0.15)
    v_tip = comm_state["v_tip_ag"]

    r_norm = np.clip(r / R, 0.02, 0.96)
    u_r = np.zeros_like(r)
    u_z = np.zeros_like(z)

    z_bottom_imp = C1_over_T * geom["T_m"]
    cell_half_h = max(0.5 * spacing_over_D * geom["D_ag"], 0.22 * H_tot)

    for k in range(n_tiers):
        z_k = z_bottom_imp + k * spacing_over_D * geom["D_ag"]
        dz_norm = np.clip((z - z_k) / cell_half_h, -1.2, 1.2)
        env_k = np.exp(-1.6 * (dz_norm ** 2))

        # Axial loop: downflow in core (r_norm < 0.62), upflow near wall (r_norm > 0.62)
        uz_ax = -V_circ * np.cos(0.5 * np.pi * np.clip(dz_norm, -1.0, 1.0)) * (1.0 - 2.60 * (r_norm ** 2))
        ur_ax = -V_circ * np.sin(0.5 * np.pi * np.clip(dz_norm, -1.0, 1.0)) * r_norm * (1.0 - r_norm ** 2) * 1.4

        # Radial figure-8 twin loop: outward jet at z_k, splitting up/down at wall
        ur_rad = V_circ * np.cos(np.pi * np.clip(dz_norm, -1.0, 1.0)) * np.sin(np.pi * r_norm)
        uz_rad = V_circ * np.sin(np.pi * np.clip(dz_norm, -1.0, 1.0)) * (2.0 * r_norm - 1.0)

        u_r += env_k * (ax_weight * ur_ax + rad_weight * ur_rad)
        u_z += env_k * (ax_weight * uz_ax + rad_weight * uz_rad)

    # Bottom Rotor-Stator high-shear jet near dish floor (z < 0.18 * H_tot)
    rs_zone = np.exp(-((z / (0.14 * H_tot)) ** 2))
    u_r += rs_zone * 0.35 * comm_state["v_tip_rs"] * np.exp(-3.5 * r_norm) * (r_norm / 0.25)
    u_z -= rs_zone * 0.20 * comm_state["v_tip_rs"] * np.exp(-6.0 * (r_norm ** 2))

    # Tangential Swirl (damped by wall baffles)
    omega_b = n_baffles * Wb_over_T
    swirl_factor = 0.18 + 0.82 * np.exp(-3.8 * (omega_b / 0.3333))
    r_over_rag = r / max(R_ag, 0.05)
    u_theta = v_tip * swirl_factor * (2.0 * r_over_rag) / (1.0 + r_over_rag ** 2)

    # Yield-Stress Cavern Stagnation Filter (Dc / T)
    R_cav = comm_state["Dc_over_T"] * R
    psi_cav = 1.0 / (1.0 + np.exp(14.0 * (r - R_cav) / R))

    u_r *= psi_cav
    u_z *= psi_cav
    u_theta *= psi_cav
    speed_mag = np.sqrt(u_r ** 2 + u_theta ** 2 + u_z ** 2)
    return u_r, u_theta, u_z, speed_mag


def build_animated_3d_vessel_figure(
    geom: dict, comm_state: dict, dish_name: str, n_baffles: int, Wb_over_T: float,
    n_tiers: int, spacing_over_D: float, C1_over_T: float,
    n_blades: int, pitch_deg: float, Wimp_over_D: float, D_rs: float,
    n_particles: int = 320, n_frames: int = 28
) -> go.Figure:
    """
    Builds a 3D Plotly CAD reactor with multi-frame Lagrangian fluid particle animation
    and rotating impeller blades.
    """
    R = geom["T_m"] / 2.0
    h_dish = geom["h_dish"]
    H_tot = geom["H_total"]
    R_ag = geom["D_ag"] / 2.0

    # Initialize Lagrangian tracer particles throughout the liquid volume
    rng = np.random.default_rng(42)
    p_r = R * np.sqrt(rng.uniform(0.03, 0.88, n_particles))
    p_theta = rng.uniform(0.0, 2.0 * np.pi, n_particles)
    p_z = rng.uniform(0.06 * H_tot, 0.93 * H_tot, n_particles)

    # Dish floor clipping helper
    def clip_to_vessel(r_arr, z_arr):
        r_c = np.clip(r_arr, 0.03 * R, 0.92 * R)
        if dish_name == "Flat Bottom":
            z_floor = 0.02 * H_tot
        elif dish_name == "Conical (45°)":
            z_floor = h_dish * (r_c / R) + 0.02 * H_tot
        else:
            z_floor = h_dish * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - (r_c / R) ** 2))) + 0.02 * H_tot
        z_c = np.clip(z_arr, z_floor, 0.94 * H_tot)
        return r_c, z_c

    p_r, p_z = clip_to_vessel(p_r, p_z)
    u_r0, u_th0, u_z0, spd0 = evaluate_3d_velocity_field(
        p_r, p_theta, p_z, geom, comm_state, n_tiers, spacing_over_D,
        C1_over_T, pitch_deg, n_baffles, Wb_over_T
    )

    # Trace 0: Animated Lagrangian Fluid Particles (with velocity tail segments)
    px0 = p_r * np.cos(p_theta)
    py0 = p_r * np.sin(p_theta)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=px0, y=py0, z=p_z,
        mode="markers",
        marker=dict(
            size=4.5,
            color=spd0,
            colorscale="Turbo",
            cmin=0.0,
            cmax=max(float(np.percentile(spd0, 96)), 0.5),
            colorbar=dict(title="Fluid Velocity |u| (m/s)", thickness=14, x=0.98),
            opacity=0.92,
        ),
        name="Fluid Parcels",
    ))

    # Static Vessel Shell & Dish Surfaces
    theta_grid = np.linspace(0, 2 * np.pi, 40)
    z_cyl = np.linspace(h_dish, H_tot, 16)
    Th_c, Z_c = np.meshgrid(theta_grid, z_cyl)
    fig.add_trace(go.Surface(
        x=R * np.cos(Th_c), y=R * np.sin(Th_c), z=Z_c,
        opacity=0.18, showscale=False, colorscale=[[0, "#64b5f6"], [1, "#1e88e5"]], name="Vessel Wall"
    ))

    u_d = np.linspace(0.0, 1.0, 15)
    Th_d, U_d = np.meshgrid(theta_grid, u_d)
    if dish_name == "Flat Bottom":
        R_d, Z_d = R * U_d, np.zeros_like(U_d)
    elif dish_name == "Conical (45°)":
        R_d, Z_d = R * U_d, h_dish * U_d
    else:
        R_d = R * np.sin(0.5 * np.pi * U_d)
        Z_d = h_dish * (1.0 - np.cos(0.5 * np.pi * U_d))
    fig.add_trace(go.Surface(
        x=R_d * np.cos(Th_d), y=R_d * np.sin(Th_d), z=Z_d,
        opacity=0.25, showscale=False, colorscale=[[0, "#4fc3f7"], [1, "#0288d1"]], name=dish_name
    ))

    # Yield-Stress Cavern Boundary Cylinder Ring (if Dc/T < 1.0)
    if comm_state["Dc_over_T"] < 0.99:
        R_cav = comm_state["Dc_over_T"] * R
        z_cav = np.linspace(0.08 * H_tot, 0.90 * H_tot, 10)
        Th_cv, Z_cv = np.meshgrid(theta_grid, z_cav)
        fig.add_trace(go.Surface(
            x=R_cav * np.cos(Th_cv), y=R_cav * np.sin(Th_cv), z=Z_cv,
            opacity=0.16, showscale=False, colorscale=[[0, "#ef5350"], [1, "#e53935"]],
            name="Cavern Stagnation Boundary"
        ))

    # Wall Baffles
    if n_baffles > 0:
        W_b = Wb_over_T * geom["T_m"]
        c_wall = 0.018 * geom["T_m"]
        for b_idx in range(n_baffles):
            ang = (2.0 * np.pi * b_idx) / n_baffles
            r_out = R - c_wall
            r_in = max(r_out - W_b, 0.05)
            rb_g, zb_g = np.meshgrid([r_in, r_out], [h_dish + 0.03 * H_tot, 0.96 * H_tot])
            fig.add_trace(go.Surface(
                x=rb_g * np.cos(ang), y=rb_g * np.sin(ang), z=zb_g,
                opacity=0.75, showscale=False, colorscale=[[0, "#90a4ae"], [1, "#546e7a"]],
                name=f"Baffle {b_idx+1}"
            ))

    # Central Agitator Shaft
    z_bottom_imp = C1_over_T * geom["T_m"]
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[z_bottom_imp * 0.85, H_tot * 1.10],
        mode="lines", line=dict(color="#eceff1", width=10), name="Shaft"
    ))

    # Multi-Tier Impeller Blades
    blade_w = Wimp_over_D * geom["D_ag"]
    pitch_rad = np.deg2rad(pitch_deg)
    for t_idx in range(n_tiers):
        z_hub = z_bottom_imp + t_idx * spacing_over_D * geom["D_ag"]
        if z_hub > 0.95 * H_tot:
            break
        for bl_idx in range(n_blades):
            base_ang = (2.0 * np.pi * bl_idx) / n_blades
            r_span = np.linspace(0.18 * R_ag, R_ag, 5)
            w_span = np.linspace(-0.5 * blade_w, 0.5 * blade_w, 3)
            R_b, W_b_mesh = np.meshgrid(r_span, w_span)
            X_b = R_b * np.cos(base_ang) - W_b_mesh * np.cos(pitch_rad) * np.sin(base_ang)
            Y_b = R_b * np.sin(base_ang) + W_b_mesh * np.cos(pitch_rad) * np.cos(base_ang)
            Z_b = z_hub + W_b_mesh * np.sin(pitch_rad)
            fig.add_trace(go.Surface(
                x=X_b, y=Y_b, z=Z_b, opacity=0.95, showscale=False,
                colorscale=[[0, "#ffb300"], [1, "#fb8c00"]], name="Impeller Blade"
            ))

    # Bottom Rotor-Stator Head
    r_rs = D_rs / 2.0
    th_rs = np.linspace(0, 2 * np.pi, 20)
    z_rs_vals = np.linspace(0.02 * H_tot, 0.02 * H_tot + 0.35 * D_rs, 3)
    Th_r, Z_r = np.meshgrid(th_rs, z_rs_vals)
    fig.add_trace(go.Surface(
        x=r_rs * np.cos(Th_r), y=r_rs * np.sin(Th_r), z=Z_r,
        opacity=0.95, showscale=False, colorscale=[[0, "#ef5350"], [1, "#c62828"]],
        name="Rotor-Stator"
    ))

    # Generate Lagrangian Advection Frames for Trace 0
    dt = 0.045 * (geom["T_m"] / max(comm_state["v_tip_ag"], 0.5))
    frames = []
    cur_r, cur_th, cur_z = p_r.copy(), p_theta.copy(), p_z.copy()
    for fr_idx in range(n_frames):
        ur, uth, uz, spd = evaluate_3d_velocity_field(
            cur_r, cur_th, cur_z, geom, comm_state, n_tiers, spacing_over_D,
            C1_over_T, pitch_deg, n_baffles, Wb_over_T
        )
        cur_r = cur_r + ur * dt
        cur_th = (cur_th + (uth / np.maximum(cur_r, 0.04 * R)) * dt) % (2.0 * np.pi)
        cur_z = cur_z + uz * dt
        cur_r, cur_z = clip_to_vessel(cur_r, cur_z)

        fx = cur_r * np.cos(cur_th)
        fy = cur_r * np.sin(cur_th)
        frames.append(go.Frame(
            name=f"f{fr_idx}",
            data=[go.Scatter3d(
                x=fx, y=fy, z=cur_z,
                mode="markers",
                marker=dict(
                    size=4.5, color=spd, colorscale="Turbo",
                    cmin=0.0, cmax=max(float(np.percentile(spd0, 96)), 0.5), opacity=0.92
                )
            )],
            traces=[0]
        ))

    fig.frames = frames
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, b=0, t=35),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Elevation Z (m)",
            aspectmode="data", bgcolor="#0e1117"
        ),
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        showlegend=False,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.02, y=0.96,
            bgcolor="#1e293b",
            font=dict(color="#ffffff"),
            buttons=[
                dict(
                    label="▶️ Play 3D Fluid Animation",
                    method="animate",
                    args=[None, dict(frame=dict(duration=65, redraw=True), fromcurrent=True, transition=dict(duration=0))]
                ),
                dict(
                    label="⏸️ Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                ),
            ]
        )],
    )
    return fig


# ==============================================================================
# MODULE 3: LEVEL-5 MULTIPHYSICS SOLVER (RHEOLOGY, SCALE-UP, RTD, 30-BIN PBM)
# ==============================================================================
def evaluate_single_scale_state(
    N_ag_rpm: float, geom: dict, solv: dict,
    K_rel: float, n_rheo: float, tau_y: float,
    rho_S: float, dp_feed_um: float, X_wt: float,
    rs_name: str, D_rs: float, N_rs_rpm: float, gap_mm: float,
    L_shaft: float, d_shaft_mm: float, P_head_barg: float, T_bulk_C: float, T_cool_C: float
) -> dict:
    N_ag = max(N_ag_rpm / 60.0, 1e-4)
    D_ag = geom["D_ag"]
    T_m = geom["T_m"]
    V_m3 = geom["V_m3"]

    xs_frac = np.clip(X_wt / 100.0, 0.001, 0.65)
    rho_slurry = 1.0 / ((1.0 - xs_frac) / solv["rho"] + xs_frac / rho_S)
    phi_vol = xs_frac * (rho_slurry / rho_S)
    kd_mult = (1.0 - min(phi_vol / 0.62, 0.92)) ** (-2.5 * 0.62)

    K_T = K_rel * (solv["mu_cP"] / 1000.0) * kd_mult
    gamma_ag = geom["Ks_geom"] * N_ag
    mu_bulk = (tau_y / max(gamma_ag, 1e-4)) + K_T * (gamma_ag ** (n_rheo - 1.0))
    Re_ag = (rho_slurry * N_ag * (D_ag ** 2)) / mu_bulk

    Np_1 = (geom["Kp"] / Re_ag) + geom["Kt"] * (Re_ag ** -0.25) + geom["Np_inf_single"]
    Nq_1 = geom["Nq_inf_single"] * ((Re_ag / (Re_ag + geom["CQ"])) ** 0.5)
    Np_tot = Np_1 * geom["tier_P_mult"]
    Nq_tot = Nq_1 * geom["tier_Q_mult"]

    P_ag = Np_tot * rho_slurry * (N_ag ** 3) * (D_ag ** 5)
    P_ag_per_V = P_ag / V_m3
    torque_ag = P_ag / (2.0 * np.pi * N_ag)
    v_tip_ag = np.pi * D_ag * N_ag
    Q_ag = Nq_tot * N_ag * (D_ag ** 3)
    Q_ag_Lmin = Q_ag * 60000.0
    v_bulk = (4.0 * Q_ag) / (np.pi * (T_m ** 2))
    eps_max_ag = geom["Phi_eps"] * Np_1 * (N_ag ** 3) * (D_ag ** 2)

    theta_turb = (5.20 / ((Np_1 ** (1.0 / 3.0)) * N_ag)) * ((T_m / D_ag) ** 2) * (geom["H_over_T"] ** 1.45)
    theta_trans = (165.0 / ((Np_1 ** (2.0 / 3.0)) * N_ag * min(Re_ag, 1000.0) / 100.0)) * ((T_m / D_ag) ** 2)
    theta_95 = max(theta_turb, theta_trans)

    delta_rho = abs(rho_S - solv["rho"])
    buoy_term = ((9.81 * max(delta_rho, 5.0) / solv["rho"]) ** 0.45) * ((dp_feed_um * 1e-6) ** 0.2) * (max(X_wt, 0.1) ** 0.13)
    njs_base_rps = (
        geom["S_geom"] * ((K_T * (geom["Ks_geom"] ** (n_rheo - 1.0)) / rho_slurry) ** 0.1)
        * buoy_term * (D_ag ** -0.85)
    ) ** (1.0 / (1.1 - 0.1 * n_rheo))
    phi_trans = 1.0 + 0.35 * ((500.0 / max(Re_ag, 20.0)) ** 0.30) if Re_ag < 5000.0 else 1.0
    N_js_rpm = njs_base_rps * phi_trans * 60.0

    if tau_y > 0.01:
        Dc_over_D = ((1.36 / (np.pi ** 2)) * ((Np_tot * rho_slurry * (N_ag ** 2) * (D_ag ** 2)) / tau_y)) ** (1.0 / 3.0)
        Dc_over_T = Dc_over_D * (D_ag / T_m)
        N_cav_rpm = np.sqrt(((np.pi ** 2) * tau_y * ((T_m / D_ag) ** 3)) / (1.36 * Np_tot * rho_slurry * (D_ag ** 2))) * 60.0
    else:
        Dc_over_T = 1.25
        N_cav_rpm = 0.0

    rs = ROTOR_STATOR_CATALOG[rs_name]
    N_rs = max(N_rs_rpm / 60.0, 1e-4)
    delta_g = gap_mm * 1e-3
    v_tip_rs = np.pi * D_rs * N_rs
    gamma_rs = v_tip_rs / delta_g
    mu_rs = (tau_y / max(gamma_rs, 1.0)) + K_T * (gamma_rs ** (n_rheo - 1.0))
    nu_rs = mu_rs / rho_slurry
    Re_rs = (rho_slurry * N_rs * (D_rs ** 2)) / mu_rs

    Nq_rs = rs["Nq_rs"] * np.sqrt(Re_rs / (Re_rs + 200.0))
    Np_rs = (rs["Kp_rs"] / Re_rs) + rs["Np0_rs"] + rs["Np1_rs"] * Nq_rs
    P_rs = Np_rs * rho_slurry * (N_rs ** 3) * (D_rs ** 5)
    Q_rs = Nq_rs * N_rs * (D_rs ** 3)
    Q_rs_Lmin = Q_rs * 60000.0

    V_shear = np.pi * (D_rs ** 2) * rs["Href_ratio"] * (3.0 * delta_g)
    eps_max_rs = (0.70 * P_rs) / (rho_slurry * V_shear)
    lambda_K_rs_um = ((nu_rs ** 3) / max(eps_max_rs, 1e-6)) ** 0.25 * 1e6

    Pr = (solv["Cp"] * mu_bulk) / solv["k_f"]
    h_in = (0.74 * solv["k_f"] / T_m) * (Re_ag ** 0.67) * (Pr ** 0.33) * 0.88
    U_jacket = 1.0 / ((1.0 / h_in) + (0.006 / 16.3) + (1.0 / 1800.0))
    delta_T_slot = P_rs / (rho_slurry * solv["Cp"] * max(Q_rs, 1e-6))
    T_slot_C = T_bulk_C + delta_T_slot

    # Multi-solvent vapor pressure & pressure-adjusted bubble point evaluation
    if "pvap_fn" in solv:
        P_vap_slot_Pa = solv["pvap_fn"](T_slot_C)
        P_target_Pa = 101325.0 + P_head_barg * 1e5
        try:
            T_boil_P_C = brentq(lambda tc: solv["pvap_fn"](tc) - P_target_Pa, -20.0, 260.0, xtol=0.05)
        except ValueError:
            T_boil_P_C = solv["bp_C"] + 18.0 * P_head_barg
    else:
        A_ant, B_ant, C_ant = solv["antoine"]
        P_vap_slot_Pa = 133.322 * (10.0 ** (A_ant - B_ant / (T_slot_C + C_ant)))
        P_abs_mmHg = max((101325.0 + P_head_barg * 1e5) / 133.322, 10.0)
        T_boil_P_C = (B_ant / (A_ant - np.log10(P_abs_mmHg))) - C_ant

    P_abs_Pa = 101325.0 + (P_head_barg * 1e5) + (rho_slurry * 9.81 * geom["H_total"])
    sigma_cav = (P_abs_Pa - P_vap_slot_Pa) / (0.5 * rho_slurry * (v_tip_rs ** 2))

    E_Y = 193e9
    d_shaft_m = d_shaft_mm * 1e-3
    I_s = (np.pi * (d_shaft_m ** 4)) / 64.0
    m_shaft = 8000.0 * (np.pi / 4.0) * (d_shaft_m ** 2) * L_shaft
    m_imp = 42.0 * (D_ag ** 2.6) * geom["tier_P_mult"]
    visc_damp = np.sqrt(1.0 + 0.04 * np.log10(max(1.0, mu_bulk * 1000.0 / 100.0)))
    N_crit_rpm = (30.0 / np.pi) * np.sqrt((3.0 * E_Y * I_s) / ((L_shaft ** 3) * (0.75 * m_imp + 0.236 * m_shaft))) * visc_damp
    F_rad_N = 0.28 * P_ag / (2.0 * np.pi * N_ag * D_ag)
    delta_seal_um = ((F_rad_N * (L_shaft ** 3)) / (3.0 * E_Y * I_s)) * 0.35 * 1e6

    return {
        "N_ag_rpm": N_ag_rpm, "N_rs_rpm": N_rs_rpm,
        "rho_slurry": rho_slurry, "K_T": K_T,
        "mu_bulk_cP": mu_bulk * 1000.0, "mu_rs_cP": mu_rs * 1000.0,
        "Re_ag": Re_ag, "Np_tot": Np_tot, "Nq_tot": Nq_tot,
        "P_ag_W": P_ag, "P_ag_per_V": P_ag_per_V, "torque_Nm": torque_ag,
        "v_tip_ag": v_tip_ag, "Q_ag_Lmin": Q_ag_Lmin, "Q_ag_per_V_min": Q_ag_Lmin / (V_m3 * 1000.0),
        "v_bulk": v_bulk, "eps_max_ag": eps_max_ag, "theta_95_s": theta_95,
        "N_js_rpm": N_js_rpm, "Dc_over_T": min(Dc_over_T, 1.5), "N_cav_rpm": N_cav_rpm,
        "P_rs_W": P_rs, "Q_rs_Lmin": Q_rs_Lmin, "v_tip_rs": v_tip_rs,
        "eps_max_rs": eps_max_rs, "lambda_K_rs_um": lambda_K_rs_um,
        "feeder_ratio": Q_ag_Lmin / max(Q_rs_Lmin, 1e-6),
        "U_jacket": U_jacket, "delta_T_slot": delta_T_slot, "T_slot_C": T_slot_C,
        "P_vap_slot_kPa": P_vap_slot_Pa / 1000.0, "T_boil_P_C": T_boil_P_C, "sigma_cav": sigma_cav,
        "N_crit_rpm": N_crit_rpm, "speed_crit_pct": (N_ag_rpm / N_crit_rpm) * 100.0,
        "delta_seal_um": delta_seal_um,
    }


def solve_scaleup_operating_window(lab_state: dict, evaluate_comm_fn) -> dict:
    comm_baseline = evaluate_comm_fn(150.0)
    targets = {
        "1. Min Solids Suspension (N_js,2)": comm_baseline["N_js_rpm"],
        "2. Min Cavern Wall-Reach (Dc/T = 1.0)": max(comm_baseline["N_cav_rpm"], 20.0),
    }
    criteria_keys = [
        ("3. Constant Tip Speed (v_tip)", "v_tip_ag"),
        ("4. Constant Specific Power (P_ag/V)", "P_ag_per_V"),
        ("5. Constant Agitator Max Shear (eps_max)", "eps_max_ag"),
        ("6. Constant Bulk Fluid Velocity (v_b)", "v_bulk"),
        ("7. Constant Pumping Turnover (Q_ag/V)", "Q_ag_per_V_min"),
    ]
    for label, key in criteria_keys:
        target_val = lab_state[key]
        try:
            sol_rpm = brentq(lambda rpm: evaluate_comm_fn(rpm)[key] - target_val, 5.0, 1500.0, xtol=0.05)
        except ValueError:
            sol_rpm = np.nan
        targets[label] = sol_rpm
    return targets


def simulate_30bin_pbm_and_rtd(
    comm_state: dict, geom_comm: dict, solv: dict, rs_name: str,
    dp_feed_um: float, d90_spec_um: float, t_batch_min: float,
    T_init_C: float, T_cool_C: float, T_crit_C: float,
    alpha_sc: float, M_stages: int, Q_loop_Lmin: float, n_tiers: int
) -> dict:
    n_bins = 30
    d_bins = np.geomspace(0.3, 500.0, n_bins)
    rs = ROTOR_STATOR_CATALOG[rs_name]
    V_L = geom_comm["V_m3"] * 1000.0

    d_eq_um = max(0.8, 175.0 * (comm_state["eps_max_rs"] ** -0.35) * ((solv["sigma_mN"] / 30.0) ** 0.35))

    pdf_coarse = np.exp(-0.5 * ((np.log(d_bins / dp_feed_um)) / np.log(1.45)) ** 2)
    pdf_coarse /= np.sum(pdf_coarse)
    pdf_fine = np.exp(-0.5 * ((np.log(d_bins / d_eq_um)) / np.log(1.36)) ** 2)
    pdf_fine /= np.sum(pdf_fine)
    pdf_floc = np.exp(-0.5 * ((np.log(d_bins / 38.0)) / np.log(1.35)) ** 2)
    pdf_floc /= np.sum(pdf_floc)

    t_min_grid = np.linspace(0.0, 90.0, 181)
    t_sec_grid = t_min_grid * 60.0

    P_tot_W = comm_state["P_ag_W"] + comm_state["P_rs_W"]
    UA = comm_state["U_jacket"] * geom_comm["A_jacket"]
    tau_therm = (comm_state["rho_slurry"] * geom_comm["V_m3"] * solv["Cp"]) / max(UA, 1.0)
    T_ss_C = T_cool_C + P_tot_W / max(UA, 1.0)
    T_bulk_traj = T_ss_C + (T_init_C - T_ss_C) * np.exp(-t_sec_grid / tau_therm)
    T_slot_traj = T_bulk_traj + comm_state["delta_T_slot"]

    N_eff_intank = (1.0 - alpha_sc) * (comm_state["Q_rs_Lmin"] * t_min_grid) / V_L
    wc_intank = np.exp(-rs["k_pass"] * N_eff_intank)
    overheat_drive = np.maximum(0.0, T_slot_traj - (T_crit_C - 4.0))
    wa_intank = np.minimum(0.60, 0.012 * N_eff_intank * (np.exp(0.16 * overheat_drive) - 1.0))
    wf_intank = np.maximum(0.0, 1.0 - wc_intank - wa_intank)

    D90_intank = (
        d_eq_um * (1.36 ** 1.2816)
        + (wc_intank ** 0.75) * (dp_feed_um * (1.45 ** 1.2816) - d_eq_um * (1.36 ** 1.2816))
        + (wa_intank ** 0.85) * (38.0 * (1.35 ** 1.2816))
    )
    D99_intank = (
        d_eq_um * 1.65
        + (wc_intank ** 0.52) * (dp_feed_um * 1.85 - d_eq_um * 1.65)
        + (wa_intank ** 0.75) * 58.0
    )

    N_loop_inline = (Q_loop_Lmin * t_min_grid) / V_L
    rtd_boost = 1.0 + 0.08 * (n_tiers - 1)
    N_stage_inline = M_stages * (1.0 - 0.12) * 0.98 * N_loop_inline * rtd_boost
    wc_inline = np.exp(-rs["k_pass"] * N_stage_inline)
    wa_inline = np.minimum(0.60, 0.010 * N_stage_inline * (np.exp(0.16 * overheat_drive) - 1.0))
    wf_inline = np.maximum(0.0, 1.0 - wc_inline - wa_inline)

    D90_inline = (
        d_eq_um * (1.36 ** 1.2816)
        + (wc_inline ** 0.75) * (dp_feed_um * (1.45 ** 1.2816) - d_eq_um * (1.36 ** 1.2816))
        + (wa_inline ** 0.85) * (38.0 * (1.35 ** 1.2816))
    )
    D99_inline = (
        d_eq_um * 1.50
        + (wc_inline ** 0.82) * (dp_feed_um * 1.85 - d_eq_um * 1.50)
        + (wa_inline ** 0.75) * 52.0
    )

    idx_t = int(np.argmin(np.abs(t_min_grid - t_batch_min)))
    psd_initial = pdf_coarse * 100.0
    psd_final_intank = (wc_intank[idx_t] * pdf_coarse + wf_intank[idx_t] * pdf_fine + wa_intank[idx_t] * pdf_floc) * 100.0
    psd_final_inline = (wc_inline[idx_t] * pdf_coarse + wf_inline[idx_t] * pdf_fine + wa_inline[idx_t] * pdf_floc) * 100.0

    k_vals = np.arange(0, 36)
    lam_A = max(N_eff_intank[idx_t], 0.05)
    prob_A = ((lam_A ** k_vals) / factorial(k_vals)) * np.exp(-lam_A) * 100.0
    lam_B = max(N_stage_inline[idx_t], 0.05)
    prob_B = ((lam_B ** k_vals) / factorial(k_vals)) * np.exp(-lam_B) * np.exp(-0.04 * ((k_vals - lam_B) ** 2))
    prob_B = (prob_B / max(np.sum(prob_B), 1e-12)) * 100.0

    return {
        "d_bins": d_bins, "d_eq_um": d_eq_um, "t_min_grid": t_min_grid, "idx_t": idx_t,
        "T_bulk_traj": T_bulk_traj, "T_slot_traj": T_slot_traj,
        "D90_intank": D90_intank, "D99_intank": D99_intank,
        "D90_inline": D90_inline, "D99_inline": D99_inline,
        "psd_initial": psd_initial, "psd_final_intank": psd_final_intank, "psd_final_inline": psd_final_inline,
        "k_vals": k_vals, "prob_A": prob_A, "prob_B": prob_B,
        "f0_intank_pct": float(np.exp(-lam_A) * 100.0),
        "f0_inline_pct": float(np.exp(-lam_B * 1.4) * 100.0),
    }


# ==============================================================================
# MODULE 4: PYTORCH PARAMETRIC PINN SURROGATE & AUTOGRAD SETPOINT OPTIMIZER
# ==============================================================================
if HAS_TORCH:
    class ParametricMixingPINN(nn.Module):
        def __init__(self, theta_b0: float, hidden_dim: int = 64):
            super().__init__()
            self.theta_b0 = theta_b0
            self.net = nn.Sequential(
                nn.Linear(4, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )

        def forward(self, x):
            tau = x[:, 0:1]
            z = self.net(x)
            z1, z2, z3 = z[:, 0:1], z[:, 1:2], z[:, 2:3]
            w_c = torch.exp(-tau * nn.functional.softplus(z1))
            w_a = (1.0 - w_c) * torch.sigmoid(z2) * torch.tanh(3.0 * tau)
            Theta_b = self.theta_b0 + tau * z3
            return torch.cat([w_c, w_a, Theta_b], dim=1)


# ==============================================================================
# STREAMLIT UI & SIDEBAR CONFIGURATION
# ==============================================================================
st.title("⚙️ Level-5 Multiphysics Mixing & Scale-Up Digital Twin")
st.markdown(
    "**Unified Workbench:** $N$-Component Multi-Solvent Engine (Live PubChem / DIPPR) • "
    "Animated 3D Lagrangian Fluid Flow & Vessel CAD • Transitional Non-Newtonian Scale-Up • "
    "30-Bin PBM & RTD Kinetics • PyTorch PINN Autograd Optimizer"
)

with st.sidebar:
    st.header("🧪 1. Solvent & Multi-Solvent Engine")
    solvent_mode = st.radio(
        "Property Source Mode",
        [
            "1. Pure Solvent (DIPPR / Live PubChem)",
            "2. Multi-Solvent Blend (N-Component)",
            "3. Manual Custom Entry",
        ],
        index=1,
    )
    T_init_C = st.slider("Initial Batch Temperature (°C)", 10.0, 45.0, 22.0, 0.5)

    if solvent_mode == "1. Pure Solvent (DIPPR / Live PubChem)":
        solv_name = st.selectbox(
            "Select Industrial Solvent",
            list(MultiSolventPropertyEngine.DIPPR_DATABASE.keys()),
            index=3,
        )
        custom_pubchem = st.text_input("Or Fetch Any Chemical by Name / CAS (PubChem API)", value="")
        target_query = custom_pubchem.strip() if custom_pubchem.strip() else solv_name
        solv_props = MultiSolventPropertyEngine.evaluate_pure(target_query, T_init_C)
        pubchem_meta = MultiSolventPropertyEngine.fetch_pubchem_live(target_query)
        st.caption(f"🌐 **{pubchem_meta['source_badge']}** | Formula: `{pubchem_meta['formula']}`")

    elif solvent_mode == "2. Multi-Solvent Blend (N-Component)":
        solv_list = list(MultiSolventPropertyEngine.DIPPR_DATABASE.keys())
        selected_solvents = st.multiselect(
            "Select 2 to 6 Co-Solvents",
            options=solv_list,
            default=["Water", "Isopropanol (IPA)", "Acetone", "Ethyl Acetate"],
        )
        if len(selected_solvents) == 0:
            selected_solvents = ["Water", "Isopropanol (IPA)"]

        st.caption("Adjust relative mass weights (automatically normalized to 100 wt%):")
        comp_weights = {}
        default_w = [45.0, 30.0, 15.0, 10.0, 5.0, 5.0]
        for idx, s_name in enumerate(selected_solvents):
            init_val = default_w[idx] if idx < len(default_w) else 10.0
            comp_weights[s_name] = st.slider(
                f"{s_name.split(' (')[0]} Weight Share", 0.0, 100.0, float(init_val), 1.0
            )
        G_scale = st.slider("Grunberg-Nissan Non-Ideal H-Bonding Scale", 0.0, 2.0, 1.0, 0.05)
        solv_props = MultiSolventPropertyEngine.evaluate_multisolvent_mixture(
            comp_weights, T_init_C, G_scale=G_scale
        )
        st.caption(f"🧪 **Normalized Blend:** `{solv_props['name']}`")

    else:
        st.caption("✏️ **Manual Thermophysical Override Active**")
        m_rho = st.number_input("Liquid Density ρ_L (kg/m³)", 600.0, 1800.0, 920.0, 10.0)
        m_mu_cP = st.number_input("Base Solvent Viscosity μ_0 (cP)", 0.2, 5000.0, 2.5, 0.5)
        m_Cp = st.number_input("Specific Heat Cp (J/kg·K)", 900.0, 4500.0, 2350.0, 50.0)
        m_kf = st.number_input("Thermal Conductivity k_f (W/m·K)", 0.08, 0.70, 0.16, 0.01)
        m_sig = st.number_input("Surface Tension σ (mN/m)", 10.0, 80.0, 26.0, 1.0)
        m_bp = st.number_input("Normal Boiling Point (°C)", 30.0, 220.0, 82.0, 1.0)
        solv_props = {
            "name": "Custom Manual Fluid", "cas": "Manual-Entry", "mw": 65.0,
            "rho": m_rho, "mu_cP": m_mu_cP, "E_mu_R": 1600.0,
            "Cp": m_Cp, "k_f": m_kf, "sigma_mN": m_sig,
            "P_vap_Pa": 101325.0 * np.exp(10.5 * (1.0 - (m_bp + 273.15) / (T_init_C + 273.15))),
            "bp_C": m_bp, "antoine": (7.50, 1450.0, 225.0),
        }

    with st.expander("🧱 Slurry Rheology & Solids Loading", expanded=False):
        K_rel = st.slider("Polymer / Binder Consistency Multiplier (K_rel)", 1.0, 2500.0, 650.0, 25.0)
        n_rheo = st.slider("Shear-Thinning Flow Index (n)", 0.25, 1.00, 0.52, 0.02)
        tau_y = st.slider("Yield Stress τ_y (Pa)", 0.0, 15.0, 1.5, 0.25)
        X_wt = st.slider("Solids Mass Loading X (wt%)", 1.0, 45.0, 12.0, 1.0)
        rho_S = st.slider("Solid Particle Density ρ_S (kg/m³)", 1050.0, 2500.0, 1450.0, 25.0)
        dp_feed_um = st.slider("Initial Feed Particle Size D90,0 (µm)", 40.0, 400.0, 180.0, 10.0)
        d90_spec_um = st.slider("Target Product Specification D90 (µm)", 2.0, 30.0, 8.0, 0.5)

    st.header("🔬 2. Lab Scale (Vessel 1) Baseline")
    with st.expander("Configure Lab Vessel 1", expanded=False):
        V1_L = st.slider("Lab Volume V1 (L)", 5.0, 100.0, 20.0, 5.0)
        T1_m = st.slider("Lab Tank Diameter T1 (m)", 0.15, 0.50, 0.28, 0.01)
        imp1_name = st.selectbox("Lab Impeller Family", list(IMPELLER_CATALOG.keys()), index=2)
        D1_ag = st.slider("Lab Impeller Diameter D1 (m)", 0.06, 0.25, 0.12, 0.01)
        N1_ag_rpm = st.slider("Lab Agitator Speed N1 (RPM)", 100.0, 800.0, 360.0, 10.0)
        D1_rs = st.slider("Lab Rotor-Stator Diameter D_rs,1 (m)", 0.025, 0.080, 0.045, 0.005)
        N1_rs_rpm = st.slider("Lab Rotor-Stator Speed N_rs,1 (RPM)", 2000.0, 10000.0, 6000.0, 250.0)

    st.header("🏭 3. Commercial Scale (Vessel 2) CAD")
    V2_L = st.slider("Commercial Batch Volume V2 (L)", 200.0, 5000.0, 1000.0, 100.0)
    T2_m = st.slider("Commercial Tank Diameter T2 (m)", 0.55, 1.80, 1.05, 0.05)
    dish2_name = st.selectbox("Bottom Dish Head Shape", list(DISH_SHAPES.keys()), index=0)
    n_baffles2 = st.select_slider("Wall Baffle Count (N_b)", options=[0, 2, 3, 4, 6], value=4)
    Wb2_over_T = st.slider("Baffle Width Ratio (W_b / T)", 0.05, 0.12, 0.083, 0.005)

    imp2_name = st.selectbox("Commercial Impeller Family", list(IMPELLER_CATALOG.keys()), index=0)
    D2_over_T = st.slider("Impeller-to-Tank Ratio (D2 / T2)", 0.28, 0.68, 0.46, 0.02)
    D2_ag = D2_over_T * T2_m
    n_tiers2 = st.select_slider("Number of Agitator Tiers", options=[1, 2, 3], value=2)
    spacing2_over_D = st.slider("Inter-Tier Spacing (ΔZ / D2)", 0.70, 2.20, 1.40, 0.10)
    C1_over_T2 = st.slider("Bottom Impeller Clearance (C1 / T2)", 0.10, 0.40, 0.22, 0.02)

    with st.expander("🔪 Customize Blade Pitch, Width & Shaft", expanded=False):
        imp2_defaults = IMPELLER_CATALOG[imp2_name]
        n_blades2 = st.select_slider("Blade Count per Tier (n_b)", options=[2, 3, 4, 6, 8], value=int(imp2_defaults["nb_ref"]))
        pitch2_deg = st.slider("Blade Pitch Angle θ_b (deg)", 15.0, 90.0, float(imp2_defaults["pitch_ref"]), 1.0)
        Wimp2_over_D = st.slider("Blade Width Ratio (W_imp / D2)", 0.10, 0.35, float(imp2_defaults["wd_ref"]), 0.01)
        d_shaft2_mm = st.slider("Shaft Outer Diameter (mm)", 40.0, 140.0, 75.0, 5.0)
        L_shaft2_m = st.slider("Shaft Overhang Length L_shaft (m)", 1.0, 3.2, 1.65, 0.05)

    st.header("⚡ 4. Commercial Operating Setpoints")
    N2_ag_rpm = st.slider("Commercial Agitator Speed N_ag,2 (RPM)", 40.0, 350.0, 150.0, 5.0)
    rs2_name = st.selectbox("Rotor-Stator Generator Type", list(ROTOR_STATOR_CATALOG.keys()), index=0)
    D2_rs = st.slider("Commercial Rotor-Stator Diameter D_rs,2 (m)", 0.10, 0.30, 0.175, 0.005)
    N2_rs_rpm = st.slider("Commercial Rotor-Stator Speed N_rs,2 (RPM)", 1000.0, 4800.0, 2800.0, 50.0)
    gap2_mm = st.slider("Rotor-Stator Shear Gap δ_g (mm)", 0.20, 1.00, 0.45, 0.05)

    with st.expander("🔁 Inline Loop, Jacket Cooling & Pressure", expanded=False):
        M_stages = st.select_slider("Inline Homogenizer Stages (M)", options=[1, 2, 3, 4], value=3)
        Q_loop_Lmin = st.slider("External Feeder Pump Flow Q_loop (L/min)", 80.0, 600.0, 260.0, 10.0)
        alpha_sc = st.slider("In-Tank Bottom Short-Circuit Fraction α_sc", 0.05, 0.50, 0.28, 0.02)
        T_cool_C = st.slider("Jacket Coolant Supply Temp T_cool (°C)", 5.0, 35.0, 14.0, 1.0)
        T_crit_C = st.slider("Critical Thermal / DLVO Limit T_crit (°C)", 30.0, 65.0, 42.0, 1.0)
        P_head_barg = st.slider("Nitrogen Headspace Pressure (bar(g))", 0.0, 2.5, 0.3, 0.1)
        t_batch_min = st.slider("Selected Batch Homogenization Time (min)", 5.0, 90.0, 35.0, 1.0)


# ==============================================================================
# EXECUTE CORE MULTIPHYSICS CALCULATIONS
# ==============================================================================
imp1_def = IMPELLER_CATALOG[imp1_name]
geom_lab = compute_parametric_geometry(
    V1_L, T1_m, "2:1 Ellipsoidal", 4, 0.083,
    imp1_name, D1_ag, 1, 1.0, 0.25,
    imp1_def["nb_ref"], imp1_def["pitch_ref"], imp1_def["wd_ref"]
)
geom_comm = compute_parametric_geometry(
    V2_L, T2_m, dish2_name, n_baffles2, Wb2_over_T,
    imp2_name, D2_ag, n_tiers2, spacing2_over_D, C1_over_T2,
    n_blades2, pitch2_deg, Wimp2_over_D
)

lab_state = evaluate_single_scale_state(
    N1_ag_rpm, geom_lab, solv_props, K_rel, n_rheo, tau_y,
    rho_S, dp_feed_um, X_wt, rs2_name, D1_rs, N1_rs_rpm, 0.35,
    0.45, 25.0, P_head_barg, T_init_C, T_cool_C
)

eval_comm_at_rpm = lambda rpm: evaluate_single_scale_state(
    rpm, geom_comm, solv_props, K_rel, n_rheo, tau_y,
    rho_S, dp_feed_um, X_wt, rs2_name, D2_rs, N2_rs_rpm, gap2_mm,
    L_shaft2_m, d_shaft2_mm, P_head_barg, T_init_C, T_cool_C
)
comm_state = eval_comm_at_rpm(N2_ag_rpm)
scaleup_targets = solve_scaleup_operating_window(lab_state, eval_comm_at_rpm)

pbm_rtd_results = simulate_30bin_pbm_and_rtd(
    comm_state, geom_comm, solv_props, rs2_name,
    dp_feed_um, d90_spec_um, t_batch_min,
    T_init_C, T_cool_C, T_crit_C,
    alpha_sc, M_stages, Q_loop_Lmin, n_tiers2
)


# ==============================================================================
# TOP DIAGNOSTIC STATUS BANNER
# ==============================================================================
b1, b2, b3, b4, b5 = st.columns(5)
susp_ok = comm_state["N_ag_rpm"] >= comm_state["N_js_rpm"]
cav_wall_ok = comm_state["Dc_over_T"] >= 1.0
therm_ok = (comm_state["T_slot_C"] <= T_crit_C) and (comm_state["T_slot_C"] < comm_state["T_boil_P_C"] - 2.0)
cav_ok = comm_state["sigma_cav"] >= 0.28
mech_ok = (comm_state["speed_crit_pct"] <= 65.0) and (comm_state["delta_seal_um"] <= 50.0)

b1.metric(
    "1. Solids Suspension (N_ag vs N_js)",
    f"{comm_state['N_ag_rpm']:.0f} vs {comm_state['N_js_rpm']:.0f} RPM",
    "PASS: Suspended" if susp_ok else "FAIL: Settling Risk",
    delta_color="normal" if susp_ok else "inverse",
)
b2.metric(
    "2. Yield-Stress Cavern (Dc / T)",
    f"{comm_state['Dc_over_T']*100:.0f}% of Tank Wall",
    "PASS: Full Wall Reach" if cav_wall_ok else "FAIL: Stagnant Wall Ring",
    delta_color="normal" if cav_wall_ok else "inverse",
)
b3.metric(
    "3. Stator Flash Temp (T_slot)",
    f"{comm_state['T_slot_C']:.1f} °C (Boil: {comm_state['T_boil_P_C']:.1f} °C)",
    "PASS: Safe Thermal Margin" if therm_ok else "ALERT: Overheat / Boil Risk",
    delta_color="normal" if therm_ok else "inverse",
)
b4.metric(
    "4. Stator Cavitation (σ_cav)",
    f"σ = {comm_state['sigma_cav']:.2f} (Pv = {comm_state['P_vap_slot_kPa']:.1f} kPa)",
    "PASS: Liquid Shear" if cav_ok else "ALERT: Cavitation Choking",
    delta_color="normal" if cav_ok else "inverse",
)
b5.metric(
    "5. Shaft & Seal Integrity",
    f"{comm_state['speed_crit_pct']:.1f}% N_crit | {comm_state['delta_seal_um']:.1f} µm",
    "PASS: Rigid Envelope" if mech_ok else "ALERT: Whip / Seal Runout",
    delta_color="normal" if mech_ok else "inverse",
)


# ==============================================================================
# MAIN WORKBENCH TABS
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌊 1. 3D Animated Fluid Flow & Multi-Solvent Matrix",
    "📊 2. Scale-Up Window & Scorecard",
    "📈 3. 30-Bin PBM, Inline RTD & Thermal Curves",
    "🧠 4. PyTorch PINN & Autograd Optimizer",
    "📋 5. Qualification Report & Excel Export",
])

# ------------------------------------------------------------------------------
# TAB 1: 3D ANIMATED FLUID FLOW & MULTI-SOLVENT MATRIX
# ------------------------------------------------------------------------------
with tab1:
    col_cad, col_props = st.columns([1.35, 1.0])
    with col_cad:
        st.subheader("3D Animated Lagrangian Fluid Movement & Parametric Vessel CAD")
        st.caption(
            "Click **▶️ Play 3D Fluid Animation** inside the plot to watch 320 Lagrangian fluid parcels circulate "
            "through the multi-tier impellers, baffles, dish bottom, and rotor-stator jet (colored by local speed $|\\mathbf{u}|$)."
        )
        fig_3d = build_animated_3d_vessel_figure(
            geom_comm, comm_state, dish2_name, n_baffles2, Wb2_over_T,
            n_tiers2, spacing2_over_D, C1_over_T2,
            n_blades2, pitch2_deg, Wimp2_over_D, D2_rs
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_props:
        if "composition_df" in solv_props:
            st.subheader("Multi-Solvent Composition Breakdown (Mass / Mole / Vol %)")
            st.dataframe(solv_props["composition_df"], use_container_width=True, hide_index=True)

        st.subheader("Resolved Multi-Solvent & Slurry Thermophysical Matrix")
        solv_df = pd.DataFrame([
            {"Property": "Active Fluid / Multi-Solvent Blend", "Value": solv_props["name"], "Unit": solv_props["cas"]},
            {"Property": "Mixture / Solvent Density ρ_m (T)", "Value": f"{solv_props['rho']:.1f}", "Unit": "kg/m³"},
            {"Property": "Slurry Bulk Density ρ_slurry", "Value": f"{comm_state['rho_slurry']:.1f}", "Unit": "kg/m³"},
            {"Property": "Solvent Continuum Viscosity μ_m (T)", "Value": f"{solv_props['mu_cP']:.3f}", "Unit": "cP (Grunberg-Nissan)"},
            {"Property": "Slurry Bulk Apparent Viscosity μ_bulk", "Value": f"{comm_state['mu_bulk_cP']:.1f}", "Unit": "cP (at K_s·N_ag)"},
            {"Property": "Stator-Slot Apparent Viscosity μ_rs", "Value": f"{comm_state['mu_rs_cP']:.2f}", "Unit": "cP (at γ_rs)"},
            {"Property": "Mixture Specific Heat Cp,m", "Value": f"{solv_props['Cp']:.0f}", "Unit": "J/kg·K"},
            {"Property": "Mixture Thermal Conductivity k_f,m", "Value": f"{solv_props['k_f']:.3f}", "Unit": "W/m·K (Li Rule)"},
            {"Property": "Mixture Surface Tension σ_m", "Value": f"{solv_props['sigma_mN']:.1f}", "Unit": "mN/m"},
            {"Property": "Mixture Vapor Pressure at T_slot", "Value": f"{comm_state['P_vap_slot_kPa']:.2f}", "Unit": "kPa (Raoult)"},
            {"Property": "Mixture Bubble Point at Headspace P", "Value": f"{comm_state['T_boil_P_C']:.1f}", "Unit": f"°C (at {P_head_barg:.1f} bar(g))"},
        ])
        st.dataframe(solv_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: DISSIMILAR SCALE-UP OPERATING WINDOW & SCORECARD
# ------------------------------------------------------------------------------
with tab2:
    c_win, c_score = st.columns([1.15, 1.0])
    with c_win:
        st.subheader("Dissimilar Agitator Scale-Up Operating Window (RPM)")
        win_labels = list(scaleup_targets.keys())
        win_rpms = [scaleup_targets[k] for k in win_labels]
        bar_colors = [
            "#ef5350" if ("Min " in k and N2_ag_rpm < val) else "#42a5f5"
            for k, val in zip(win_labels, win_rpms)
        ]
        fig_win = go.Figure()
        fig_win.add_trace(go.Bar(
            y=win_labels, x=win_rpms, orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f} RPM" if not np.isnan(v) else "N/A" for v in win_rpms],
            textposition="outside",
            name="Required Commercial RPM"
        ))
        fig_win.add_vline(
            x=N2_ag_rpm, line_width=3, line_dash="dash", line_color="#00e676",
            annotation_text=f"Selected N_ag,2 = {N2_ag_rpm:.0f} RPM",
            annotation_position="top right"
        )
        fig_win.update_layout(
            height=460, xaxis_title="Commercial Primary Agitator Speed N_ag,2 (RPM)",
            margin=dict(l=10, r=40, t=30, b=20)
        )
        st.plotly_chart(fig_win, use_container_width=True)

    with c_score:
        st.subheader("Side-by-Side Scale-Up Process Scorecard")
        score_df = pd.DataFrame([
            {"Engineering Metric": "Primary Agitator Speed (RPM)", "Lab Vessel 1": f"{lab_state['N_ag_rpm']:.1f}", "Commercial Vessel 2": f"{comm_state['N_ag_rpm']:.1f}"},
            {"Engineering Metric": "Agitator Specific Power P_ag/V (W/m³)", "Lab Vessel 1": f"{lab_state['P_ag_per_V']:.1f}", "Commercial Vessel 2": f"{comm_state['P_ag_per_V']:.1f}"},
            {"Engineering Metric": "Rotor-Stator Power P_rs (kW)", "Lab Vessel 1": f"{lab_state['P_rs_W']/1000:.2f}", "Commercial Vessel 2": f"{comm_state['P_rs_W']/1000:.2f}"},
            {"Engineering Metric": "Bulk Apparent Viscosity μ_bulk (cP)", "Lab Vessel 1": f"{lab_state['mu_bulk_cP']:.1f}", "Commercial Vessel 2": f"{comm_state['mu_bulk_cP']:.1f}"},
            {"Engineering Metric": "Non-Newtonian Reynolds Number Re_NN", "Lab Vessel 1": f"{lab_state['Re_ag']:.0f}", "Commercial Vessel 2": f"{comm_state['Re_ag']:.0f}"},
            {"Engineering Metric": "Operating Total Power Number Np_tot", "Lab Vessel 1": f"{lab_state['Np_tot']:.2f}", "Commercial Vessel 2": f"{comm_state['Np_tot']:.2f}"},
            {"Engineering Metric": "Bulk Pumping Turnover Q_ag/V (min⁻¹)", "Lab Vessel 1": f"{lab_state['Q_ag_per_V_min']:.2f}", "Commercial Vessel 2": f"{comm_state['Q_ag_per_V_min']:.2f}"},
            {"Engineering Metric": "Multi-Tier Blend Time θ_95 (s)", "Lab Vessel 1": f"{lab_state['theta_95_s']:.1f}", "Commercial Vessel 2": f"{comm_state['theta_95_s']:.1f}"},
            {"Engineering Metric": "Min Solids Suspension Speed N_js (RPM)", "Lab Vessel 1": f"{lab_state['N_js_rpm']:.1f}", "Commercial Vessel 2": f"{comm_state['N_js_rpm']:.1f}"},
            {"Engineering Metric": "Stator Max Dissipation ε_max,rs (W/kg)", "Lab Vessel 1": f"{lab_state['eps_max_rs']:.0f}", "Commercial Vessel 2": f"{comm_state['eps_max_rs']:.0f}"},
            {"Engineering Metric": "Kolmogorov Microscale λ_K,rs (µm)", "Lab Vessel 1": f"{lab_state['lambda_K_rs_um']:.2f}", "Commercial Vessel 2": f"{comm_state['lambda_K_rs_um']:.2f}"},
            {"Engineering Metric": "Jacket Overall U-Value (W/m²·K)", "Lab Vessel 1": f"{lab_state['U_jacket']:.1f}", "Commercial Vessel 2": f"{comm_state['U_jacket']:.1f}"},
        ])
        st.dataframe(score_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 3: 30-BIN PBM, IN-TANK VS INLINE RTD & THERMAL TRAJECTORY
# ------------------------------------------------------------------------------
with tab3:
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.subheader(f"30-Bin Discretized Volume PSD at t = {t_batch_min:.0f} min")
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(
            x=pbm_rtd_results["d_bins"], y=pbm_rtd_results["psd_initial"],
            mode="lines", name="Initial Feed PSD (t = 0)", line=dict(color="#90a4ae", dash="dot", width=2)
        ))
        fig_psd.add_trace(go.Scatter(
            x=pbm_rtd_results["d_bins"], y=pbm_rtd_results["psd_final_intank"],
            mode="lines", fill="tozeroy", name=f"In-Tank Bottom RS (t = {t_batch_min:.0f} min)",
            line=dict(color="#ffa726", width=2.5)
        ))
        fig_psd.add_trace(go.Scatter(
            x=pbm_rtd_results["d_bins"], y=pbm_rtd_results["psd_final_inline"],
            mode="lines", fill="tozeroy", name=f"External {M_stages}-Stage Inline Loop (t = {t_batch_min:.0f} min)",
            line=dict(color="#29b6f6", width=3)
        ))
        fig_psd.add_vline(
            x=d90_spec_um, line_dash="dash", line_color="#00e676",
            annotation_text=f"Target D90 = {d90_spec_um:.1f} µm"
        )
        fig_psd.update_layout(
            height=400, xaxis_type="log",
            xaxis_title="Particle / Droplet Diameter d (µm, log scale)",
            yaxis_title="Volume Fraction q3(d) (% per bin)",
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_psd, use_container_width=True)

    with r1_col2:
        st.subheader("In-Tank Bottom vs. External Multi-Stage Inline Loop D90(t) & D99(t)")
        fig_kin = go.Figure()
        fig_kin.add_trace(go.Scatter(
            x=pbm_rtd_results["t_min_grid"], y=pbm_rtd_results["D90_intank"],
            mode="lines", name="In-Tank Bottom D90(t)", line=dict(color="#ffa726", width=2.5)
        ))
        fig_kin.add_trace(go.Scatter(
            x=pbm_rtd_results["t_min_grid"], y=pbm_rtd_results["D99_intank"],
            mode="lines", name="In-Tank Coarse Tail D99(t)", line=dict(color="#ffa726", dash="dot", width=2)
        ))
        fig_kin.add_trace(go.Scatter(
            x=pbm_rtd_results["t_min_grid"], y=pbm_rtd_results["D90_inline"],
            mode="lines", name=f"Inline {M_stages}-Stage Loop D90(t)", line=dict(color="#29b6f6", width=3)
        ))
        fig_kin.add_trace(go.Scatter(
            x=pbm_rtd_results["t_min_grid"], y=pbm_rtd_results["D99_inline"],
            mode="lines", name=f"Inline {M_stages}-Stage Coarse Tail D99(t)", line=dict(color="#29b6f6", dash="dot", width=2)
        ))
        fig_kin.add_hline(y=d90_spec_um, line_dash="dash", line_color="#00e676", annotation_text="Target D90 Spec")
        fig_kin.add_vline(x=t_batch_min, line_dash="dot", line_color="#e0e0e0")
        fig_kin.update_layout(
            height=400, xaxis_title="Batch Homogenization Time (min)",
            yaxis_title="Particle Diameter D90 / D99 (µm)",
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_kin, use_container_width=True)

    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.subheader(f"Stage-Pass RTD Probability Distribution P(k) at t = {t_batch_min:.0f} min")
        fig_rtd = go.Figure()
        fig_rtd.add_trace(go.Bar(
            x=pbm_rtd_results["k_vals"], y=pbm_rtd_results["prob_A"],
            name=f"In-Tank Bottom (Zero-Pass f0 = {pbm_rtd_results['f0_intank_pct']:.2f}%)",
            marker_color="#ffa726", opacity=0.80
        ))
        fig_rtd.add_trace(go.Bar(
            x=pbm_rtd_results["k_vals"], y=pbm_rtd_results["prob_B"],
            name=f"Inline {M_stages}-Stage Loop (Zero-Pass f0 = {pbm_rtd_results['f0_inline_pct']:.3f}%)",
            marker_color="#29b6f6", opacity=0.85
        ))
        fig_rtd.update_layout(
            height=380, barmode="overlay",
            xaxis_title="Effective Stator Shear Stage Passes (k)",
            yaxis_title="Batch Volume Probability P(k) (%)",
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_rtd, use_container_width=True)

    with r2_col2:
        st.subheader("Non-Isothermal Bulk & Stator-Slot Flash Temperature Trajectory")
        fig_therm = go.Figure()
        fig_therm.add_trace(go.Scatter(
            x=pbm_rtd_results["t_min_grid"], y=pbm_rtd_results["T_bulk_traj"],
            mode="lines", name="Bulk Vessel Temp T_bulk(t)", line=dict(color="#4fc3f7", width=3)
        ))
        fig_therm.add_trace(go.Scatter(
            x=pbm_rtd_results["t_min_grid"], y=pbm_rtd_results["T_slot_traj"],
            mode="lines", name="Stator-Slot Flash Temp T_slot(t)", line=dict(color="#ff7043", width=3)
        ))
        fig_therm.add_hline(
            y=T_crit_C, line_dash="dash", line_color="#ef5350",
            annotation_text=f"DLVO / Thermal Limit ({T_crit_C:.0f} °C)"
        )
        fig_therm.add_hline(
            y=comm_state["T_boil_P_C"], line_dash="dot", line_color="#ab47bc",
            annotation_text=f"Mixture Bubble Point ({comm_state['T_boil_P_C']:.1f} °C)"
        )
        fig_therm.update_layout(
            height=380, xaxis_title="Batch Homogenization Time (min)",
            yaxis_title="Temperature (°C)",
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_therm, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: PYTORCH PINN SURROGATE & INSTANT AUTOGRAD SETPOINT OPTIMIZER
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🧠 PyTorch Physics-Informed Neural Network (PINN) Inverse Setpoint Optimizer")
    if not HAS_TORCH:
        st.warning("PyTorch (`torch`) is not installed. Install via `pip install torch` to run live PINN training.")
    else:
        col_pinn_btn, _ = st.columns([1.0, 2.0])
        with col_pinn_btn:
            pinn_epochs = st.slider("PINN Physics Training Epochs", 200, 1500, 500, 100)
            opt_steps = st.slider("Autograd Inverse Optimization Steps", 50, 300, 120, 10)
            run_pinn_btn = st.button("🚀 Train PINN & Run Autograd Setpoint Optimization", type="primary")

        if run_pinn_btn:
            with st.spinner("Training 4D Parametric PINN on coupled PBM + Thermal ODEs..."):
                torch.manual_seed(42)
                theta_b0 = (T_init_C - 5.0) / 60.0
                pinn = ParametricMixingPINN(theta_b0=theta_b0, hidden_dim=64)
                opt_pinn = torch.optim.Adam(pinn.parameters(), lr=3e-3)
                loss_history = []

                for _ in range(pinn_epochs):
                    opt_pinn.zero_grad()
                    x_col = torch.rand((256, 4), dtype=torch.float32, requires_grad=True)
                    y_pred = pinn(x_col)
                    wc, wa, Th_b = y_pred[:, 0:1], y_pred[:, 1:2], y_pred[:, 2:3]

                    ones = torch.ones_like(wc)
                    dwc_dtau = torch.autograd.grad(wc, x_col, grad_outputs=ones, create_graph=True)[0][:, 0:1]
                    dwa_dtau = torch.autograd.grad(wa, x_col, grad_outputs=ones, create_graph=True)[0][:, 0:1]
                    dTh_dtau = torch.autograd.grad(Th_b, x_col, grad_outputs=ones, create_graph=True)[0][:, 0:1]

                    N_ag_t = 60.0 + 240.0 * x_col[:, 1:2]
                    N_rs_t = 1200.0 + 3300.0 * x_col[:, 2:3]
                    T_cool_t = 6.0 + 24.0 * x_col[:, 3:4]
                    T_b_t = 5.0 + 60.0 * Th_b

                    omega_t = 0.00038 * (N_rs_t / 60.0) * (D2_rs ** 3) / geom_comm["V_m3"]
                    k_eff_t = M_stages * 0.88 * 0.25 * omega_t
                    res_wc = dwc_dtau + 5400.0 * k_eff_t * wc

                    P_rs_t = 1.95 * comm_state["rho_slurry"] * ((N_rs_t / 60.0) ** 3) * (D2_rs ** 5)
                    Q_rs_t = torch.clamp(0.19 * (N_rs_t / 60.0) * (D2_rs ** 3), min=1e-5)
                    T_slot_t = T_b_t + P_rs_t / (comm_state["rho_slurry"] * solv_props["Cp"] * Q_rs_t)
                    floc_gate = nn.functional.softplus(T_slot_t - T_crit_C + 4.0)
                    res_wa = dwa_dtau - 5400.0 * 0.012 * omega_t * (1.0 - wc - wa) * floc_gate

                    P_ag_t = comm_state["Np_tot"] * comm_state["rho_slurry"] * ((N_ag_t / 60.0) ** 3) * (D2_ag ** 5)
                    UA_t = comm_state["U_jacket"] * geom_comm["A_jacket"]
                    res_Tb = dTh_dtau - (5400.0 / (comm_state["rho_slurry"] * geom_comm["V_m3"] * solv_props["Cp"] * 60.0)) * (
                        P_ag_t + P_rs_t - UA_t * (T_b_t - T_cool_t)
                    )

                    loss = torch.mean(res_wc ** 2) + torch.mean(res_wa ** 2) + torch.mean(res_Tb ** 2)
                    loss.backward()
                    opt_pinn.step()
                    loss_history.append(float(loss.item()))

                t_opt_start = time.perf_counter()
                pinn.eval()
                for p in pinn.parameters():
                    p.requires_grad_(False)

                z_u = torch.tensor([0.1, 0.2, -0.3, -0.2], dtype=torch.float32, requires_grad=True)
                opt_ctrl = torch.optim.Adam([z_u], lr=0.08)
                opt_trajectory = []

                for _ in range(opt_steps):
                    opt_ctrl.zero_grad()
                    u = torch.sigmoid(z_u)
                    tau_b = 0.10 + 0.85 * u[3]
                    N_ag_opt = 60.0 + 240.0 * u[0]
                    N_rs_opt = 1200.0 + 3300.0 * u[1]
                    T_cool_opt = 6.0 + 24.0 * u[2]

                    x_q = torch.stack([tau_b, u[0], u[1], u[2]]).unsqueeze(0)
                    pred = pinn(x_q)[0]
                    wc_q, wa_q, Th_q = pred[0], pred[1], pred[2]
                    T_b_q = 5.0 + 60.0 * Th_q

                    P_rs_q = 1.95 * comm_state["rho_slurry"] * ((N_rs_opt / 60.0) ** 3) * (D2_rs ** 5)
                    Q_rs_q = 0.19 * (N_rs_opt / 60.0) * (D2_rs ** 3)
                    T_slot_q = T_b_q + P_rs_q / (comm_state["rho_slurry"] * solv_props["Cp"] * Q_rs_q)

                    V_sh = np.pi * (D2_rs ** 2) * 0.32 * (3.0 * gap2_mm * 1e-3)
                    eps_q = (0.70 * P_rs_q) / (comm_state["rho_slurry"] * V_sh)
                    d_eq_q = torch.clamp(175.0 * (eps_q ** -0.35), min=0.8)
                    D90_q = (
                        d_eq_q * 1.50
                        + (wc_q ** 0.75) * (dp_feed_um * 1.60 - d_eq_q * 1.50)
                        + (wa_q ** 0.85) * 52.0
                    )

                    l_d90 = ((D90_q - d90_spec_um) / d90_spec_um) ** 2
                    l_time = 0.25 * tau_b
                    pen_T = 20.0 * torch.relu(T_slot_q - (T_crit_C - 1.5)) ** 2
                    pen_njs = 15.0 * torch.relu(1.15 * comm_state["N_js_rpm"] - N_ag_opt) ** 2
                    pen_floc = 40.0 * torch.relu(wa_q - 0.015) ** 2

                    tot_ctrl_loss = l_d90 + l_time + pen_T + pen_njs + pen_floc
                    tot_ctrl_loss.backward()
                    opt_ctrl.step()
                    opt_trajectory.append(float(D90_q.item()))

                opt_ms = (time.perf_counter() - t_opt_start) * 1000.0
                u_final = torch.sigmoid(z_u).detach().numpy()
                st.success(
                    f"✅ **PINN Converged in {opt_ms:.1f} ms!** | Optimal Setpoints: "
                    f"**N_ag* = {60.0 + 240.0 * u_final[0]:.0f} RPM**, "
                    f"**N_rs* = {1200.0 + 3300.0 * u_final[1]:.0f} RPM**, "
                    f"**T_cool* = {6.0 + 24.0 * u_final[2]:.1f} °C**, "
                    f"**Optimal Batch Time t* = {(0.10 + 0.85 * u_final[3]) * 90.0:.1f} min**"
                )

# ------------------------------------------------------------------------------
# TAB 5: EQUIPMENT QUALIFICATION REPORT & CSV / EXCEL EXPORT
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("📋 Complete Equipment Design & Scale-Up Qualification Sheet")
    qual_df = pd.DataFrame([
        {"Category": "Multi-Solvent System", "Parameter": "Active Fluid / Solvent Blend", "Unit": "-", "Lab Scale (Vessel 1)": solv_props["name"], "Commercial Scale (Vessel 2)": solv_props["name"]},
        {"Category": "Multi-Solvent System", "Parameter": "Slurry Density (ρ_slurry)", "Unit": "kg/m³", "Lab Scale (Vessel 1)": f"{lab_state['rho_slurry']:.1f}", "Commercial Scale (Vessel 2)": f"{comm_state['rho_slurry']:.1f}"},
        {"Category": "Vessel Geometry", "Parameter": "Working Batch Volume (V)", "Unit": "L", "Lab Scale (Vessel 1)": f"{V1_L:.1f}", "Commercial Scale (Vessel 2)": f"{V2_L:.1f}"},
        {"Category": "Vessel Geometry", "Parameter": "Tank Inner Diameter (T) & Dish Shape", "Unit": "m", "Lab Scale (Vessel 1)": f"{T1_m:.3f} (Ellipsoidal)", "Commercial Scale (Vessel 2)": f"{T2_m:.3f} ({dish2_name})"},
        {"Category": "Primary Agitator", "Parameter": "Operating Rotational Speed (N_ag)", "Unit": "RPM", "Lab Scale (Vessel 1)": f"{lab_state['N_ag_rpm']:.1f}", "Commercial Scale (Vessel 2)": f"{comm_state['N_ag_rpm']:.1f}"},
        {"Category": "Primary Agitator", "Parameter": "Min Solids Suspension Speed (N_js)", "Unit": "RPM", "Lab Scale (Vessel 1)": f"{lab_state['N_js_rpm']:.1f}", "Commercial Scale (Vessel 2)": f"{comm_state['N_js_rpm']:.1f}"},
        {"Category": "Rotor-Stator Loop", "Parameter": "Stator Flash Temp (T_slot) & Cavitation σ_cav", "Unit": "°C / -", "Lab Scale (Vessel 1)": f"{lab_state['T_slot_C']:.1f} °C (σ={lab_state['sigma_cav']:.2f})", "Commercial Scale (Vessel 2)": f"{comm_state['T_slot_C']:.1f} °C (σ={comm_state['sigma_cav']:.2f})"},
    ])
    st.dataframe(qual_df, use_container_width=True, hide_index=True)

    csv_bytes = qual_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Qualification Report (CSV)",
        data=csv_bytes,
        file_name="multisolvent_agitator_scaleup_report.csv",
        mime="text/csv",
    )
