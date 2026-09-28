#!/usr/bin/env python3
"""Reconciling 2/3 and 3/4. Metabolic scaling is curved: data dominated by small mammals give 2/3, by large mammals
3/4 (Kolokotrones et al. 2010; a break near 10 kg in Dodds et al. 2001). The canonical fit imposed Q ~ M^(3/4) and
HR ~ M^(-1/4). Here cardiac output follows a metabolic law with local exponent p(M), heart rate follows HR ~ M^(p-1)
(stroke volume ~ M), and the maintenance exponent b is fitted (i) on all 92 measurements, (ii) on small (< 10 kg) and
(iii) large (>= 10 kg) species separately. Question: does the vessel-level b track the organism-level p?

Scenarios for p(M): Kleiber 3/4; Rubner 2/3; curved: p rises from 2/3 to 3/4 with a logistic in ln M centred at 10 kg
(width 1.5 in ln M), anchored at the human value.
Canonical sensing: S = tau_mean (1 + w phi G), w = 0.4, phi = 1.32 / 5.56, carotid calibration.

    python3 src/metabolic_curvature.py --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systemic_allometry as SA
from sigma_exact_lib import G_tube, alpha_of

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--boot", type=int, default=300)
a = ap.parse_args()
W = 0.4
HIGH = {"femoral", "superficial femoral", "brachial", "iliac", "interdigital", "lateral digital", "abdominal aorta", "supraceliac aorta"}
PHI_LOW = 1.9 / float(G_tube(alpha_of(0.0033, 70))[0])
PHI_HIGH = float(np.mean([9.0 / G_tube(alpha_of(0.0039, 70))[0], 6.2 / G_tube(alpha_of(0.0020, 70))[0]]))
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "seymour2019_points.tsv"), sep="\t")
high = D.artery.str.strip().isin(HIGH).values
rS = D.r.values / 100; QS = D.Q.values * 1e-6; tau = 4 * SA.MU * QS / (np.pi * rS ** 3)
phis = np.where(high, PHI_HIGH, PHI_LOW)
hc = D[(D.species == "Homo sapiens") & (D.artery == "common carotid")]
RC, QC = float(hc.r.median()) / 100, float(hc.Q.median()) * 1e-6
small = (D.M < 10).values


def p_local(M, scen):
    if scen == "Kleiber 3/4": return 0.75
    if scen == "Rubner 2/3": return 2 / 3
    return 2 / 3 + (1 / 12) / (1 + np.exp(-(np.log(M) - np.log(10.0)) / 1.5))


def lnQ_rel(M, scen):
    """ln(Q(M)/Q(70 kg)) = integral of p(M') dlnM' from ln 70 to ln M."""
    x = np.linspace(np.log(SA.M_H), np.log(M), 200)
    return float(np.trapezoid([p_local(np.exp(v), scen) for v in x], x))


def tree(M, b, tau0, scen, step=0.5):
    Q0 = SA.Q_H * np.exp(lnQ_rel(M, scen)); hr = SA.HR_H * np.exp(lnQ_rel(M, scen) - np.log(M / SA.M_H))
    L0 = SA.L0_H * (M / SA.M_H) ** (1 / 3)
    rs, ls, g = [], [], 0.0
    while True:
        Q = Q0 / 2 ** g; L = L0 * 2 ** (-g / 3)
        r = SA.radius(Q, L, hr, b, PHI_LOW, tau0); rs.append(r); ls.append(L)
        if r < SA.R_MIN or g > 70: break
        g += step
    return np.array(rs), np.array(ls), hr


def log_sigma(b, scen):
    sf = lambda r, h, phi: float(1 + W * phi * G_tube(alpha_of(r, h))[0]); SA.sensed_factor = sf; SA.VISC = "constant"
    gC = np.log2(SA.Q_H / QC); LC = SA.L0_H * 2 ** (-gC / 3)
    tau0 = 4 * SA.MU * QC / (np.pi * RC ** 3) * sf(RC, 70, PHI_LOW) / (RC ** (b - 1) * LC ** ((b - 1) / 2))
    out = np.empty(len(D))
    for M in D.M.unique():
        m = (D.M == M).values
        rs, ls, hr = tree(M, b, tau0, scen)
        l = np.exp(np.interp(np.log(rS[m]), np.log(rs[::-1]), np.log(ls[::-1])))
        G = G_tube(alpha_of(rS[m], hr))
        out[m] = np.log10(tau[m] * (1 + W * phis[m] * G) / (tau0 * rS[m] ** (b - 1) * l ** ((b - 1) / 2)))
    return out


bgrid = np.round(np.arange(0.45, 1.001, 0.025), 3)
rng = np.random.default_rng(13); studies = D.study.values; us = np.unique(studies)
rows = []
for scen in ("Kleiber 3/4", "Rubner 2/3", "curved 2/3 -> 3/4 (break 10 kg)"):
    E = np.array([log_sigma(b, scen) for b in bgrid])
    def best(mask):
        r = np.sqrt((E[:, mask] ** 2).mean(1)); return float(bgrid[int(r.argmin())]), float(r.min())
    b_all, r_all = best(np.ones(len(D), bool)); b_s, r_s = best(small); b_l, r_l = best(~small)
    def boot(mask):
        idxs = np.where(mask)[0]; st = studies[idxs]; u = np.unique(st); o = []
        for _ in range(a.boot):
            pick = rng.choice(u, len(u)); idx = np.concatenate([idxs[st == k] for k in pick])
            o.append(bgrid[int(np.sqrt((E[:, idx] ** 2).mean(1)).argmin())])
        return np.percentile(o, [2.5, 97.5])
    ca, cs, cl = boot(np.ones(len(D), bool)), boot(small), boot(~small)
    rows.append(dict(metabolic_scenario=scen, b_all=b_all, CI_all=f"{ca[0]:.3f}-{ca[1]:.3f}", rms_all=round(r_all, 3),
                     b_small_lt10kg=b_s, CI_small=f"{cs[0]:.3f}-{cs[1]:.3f}", n_small=int(small.sum()),
                     b_large_ge10kg=b_l, CI_large=f"{cl[0]:.3f}-{cl[1]:.3f}", n_large=int((~small).sum())))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out, "metabolic_curvature_b.tsv"), sep="\t", index=False)
pd.set_option("display.width", 250); print(R.to_string(index=False))
