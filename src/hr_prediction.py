#!/usr/bin/env python3
"""Predicted change of arterial radius with resting heart rate at fixed flow (paper Fig. 4D).
Canonical model: b = 0.675, w = 0.4, phi = 1.32, target fixed on the human common carotid (r 3.3 mm, Q 7.4 mL/s).
Radius at 63 and 77 bpm (70 +- 10%) gives the elasticity d ln r / d ln HR; reported for +20 bpm (70 -> 90).

    python3 src/hr_prediction.py --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systemic_allometry as SA
from sigma_exact_lib import G_tube, alpha_of
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
b, w, phi = 0.675, 0.4, 1.32


def radius(Q, L, hr, tau0):
    f = lambda x: np.log(4 * SA.MU * Q / (np.pi * np.exp(3 * x)) * (1 + w * phi * G_tube(alpha_of(np.exp(x), hr))[0])) - \
        np.log(tau0 * np.exp(x) ** (b - 1) * L ** ((b - 1) / 2))
    return float(np.exp(brentq(f, np.log(1e-7), np.log(0.2))))


RC, QC = 0.0033, 7.4e-6; LC = 0.30 * 2 ** (-np.log2(88 / 7.4) / 3)
tau0 = 4 * SA.MU * QC / (np.pi * RC ** 3) * (1 + w * phi * G_tube(alpha_of(RC, 70))[0]) / (RC ** (b - 1) * LC ** ((b - 1) / 2))
rows = []
for name, Q, L in (("aorta", 88e-6, 0.30), ("common carotid", QC, LC), ("femoral size", 5e-6, 0.30 * 2 ** (-np.log2(88 / 5) / 3)),
                   ("small artery 0.5 mm", 0.05e-6, 0.03)):
    e = np.log(radius(Q, L, 77, tau0) / radius(Q, L, 63, tau0)) / np.log(77 / 63)
    rows.append(dict(vessel=name, radius_mm=round(radius(Q, L, 70, tau0) * 1e3, 2), dlnr_dlnHR=round(e, 3),
                     radius_change_pct_plus20bpm=round(100 * ((90 / 70) ** e - 1), 1), mean_WSS_change_pct=round(100 * ((90 / 70) ** (-3 * e) - 1), 1)))
T = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True); T.to_csv(os.path.join(a.out, "hr_prediction.tsv"), sep="\t", index=False)
print(T.to_string(index=False))
