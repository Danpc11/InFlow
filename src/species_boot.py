#!/usr/bin/env python3
"""Bootstrap of the maintenance exponent b over species (9) and over source studies (51), canonical model (w = 0.4).
Measurements from one species, or from one study, are not independent, so whole groups are resampled.

    python3 src/species_boot.py --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systemic_allometry as SA
from sigma_exact_lib import G_tube, alpha_of

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--w", type=float, default=0.4); a = ap.parse_args()
HIGH = {"femoral", "superficial femoral", "brachial", "iliac", "interdigital", "lateral digital", "abdominal aorta", "supraceliac aorta"}
PHI_LOW = 1.9 / float(G_tube(alpha_of(0.0033, 70))[0])
PHI_HIGH = float(np.mean([9.0 / G_tube(alpha_of(0.0039, 70))[0], 6.2 / G_tube(alpha_of(0.0020, 70))[0]]))
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "seymour2019_points.tsv"), sep="\t")
high = D.artery.str.strip().isin(HIGH).values
hrD = SA.HR_H * (D.M.values / SA.M_H) ** -0.25
rS = D.r.values / 100; QS = D.Q.values * 1e-6
tau = 4 * SA.MU * QS / (np.pi * rS ** 3); GS = G_tube(alpha_of(rS, hrD))
phis = np.where(high, PHI_HIGH, PHI_LOW)
hc = D[(D.species == "Homo sapiens") & (D.artery == "common carotid")]
RC, QC = float(hc.r.median()) / 100, float(hc.Q.median()) * 1e-6
LC = SA.L0_H * 2 ** (-np.log2(SA.Q_H / QC) / 3)


def log_sigma(b, w):
    sf = lambda r, h, phi: float(1 + w * phi * G_tube(alpha_of(r, h))[0])
    SA.sensed_factor = sf; SA.VISC = "constant"
    tau0 = 4 * SA.MU * QC / (np.pi * RC ** 3) * sf(RC, 70, PHI_LOW) / (RC ** (b - 1) * LC ** ((b - 1) / 2))
    out = np.empty(len(D))
    for M in D.M.unique():
        m = (D.M == M).values
        t = SA.tree(M, b, PHI_LOW, 0, tau0, g_step=0.5)
        l = np.exp(np.interp(np.log(rS[m]), np.log(t.r.values[::-1]), np.log(t.L.values[::-1])))
        out[m] = np.log10(tau[m] * (1 + w * phis[m] * GS[m]) / (tau0 * rS[m] ** (b - 1) * l ** ((b - 1) / 2)))
    return out


bgrid = np.round(np.arange(0.50, 0.951, 0.025), 3)
E = np.array([log_sigma(b, a.w) for b in bgrid])
rng = np.random.default_rng(17)
rows = []
for name, groups in (("studies", D.study.values), ("species", D.species.values)):
    u = np.unique(groups); bs = []
    for _ in range(a.boot):
        idx = np.concatenate([np.where(groups == k)[0] for k in rng.choice(u, len(u))])
        bs.append(bgrid[int(np.sqrt((E[:, idx] ** 2).mean(1)).argmin())])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append(dict(resampled=name, n_groups=len(u), b=float(bgrid[int(np.sqrt((E ** 2).mean(1)).argmin())]), CI95=f"{lo:.3f}-{hi:.3f}"))
R = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True); R.to_csv(os.path.join(a.out, "species_boot.tsv"), sep="\t", index=False)
print(R.to_string(index=False))
