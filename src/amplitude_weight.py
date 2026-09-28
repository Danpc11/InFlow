#!/usr/bin/env python3
"""Resolving the amplitude weight: in vitro (Feaver 2013) the sensed shear is S = A0 + 2.9 A1, where A1 is the FIRST
HARMONIC amplitude; in vivo the bed correction used the PEAK EXCURSION (peak - mean), which contains all harmonics.
Converting with the harmonic content of human carotid-derived waveforms (Feaver Table S1: A1/(max - A0) = 0.110 for the
CCA waveform, 0.151 for the ICS waveform) gives an in vitro weight on the peak excursion of 2.9 x (0.11-0.15) = 0.32-0.44.
This script (1) scans the in vivo weight w on the peak excursion in the Sigma analysis (92 measurements, bed-specific
phi from Reneman 2009 Table 3), (2) scans w for the invariance of S across the four human beds of Reneman Table 3.
S = tau_mean (1 + w phi_bed G_exact) ; b fixed at 0.70 (then re-profiled at the best w).

    python3 src/amplitude_weight.py --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systemic_allometry as SA
from sigma_exact_lib import G_tube, alpha_of
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()

c_cca = 2.96 / (41.38 - 14.57); c_ics = 0.46 / (3.26 - 0.22); W_FEAVER = 2.865
w_vitro = (W_FEAVER * c_cca, W_FEAVER * c_ics)
print(f"A1/(peak-mean): CCA {c_cca:.3f}, ICS {c_ics:.3f} -> in vitro weight on peak excursion {w_vitro[0]:.2f}-{w_vitro[1]:.2f}")

HIGH = {"femoral", "superficial femoral", "brachial", "iliac", "interdigital", "lateral digital", "abdominal aorta", "supraceliac aorta"}
PHI_LOW = 1.9 / float(G_tube(alpha_of(0.0033, 70))[0])
PHI_HIGH = float(np.mean([9.0 / G_tube(alpha_of(0.0039, 70))[0], 6.2 / G_tube(alpha_of(0.0020, 70))[0]]))
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "seymour2019_points.tsv"), sep="\t")
high = D.artery.str.strip().isin(HIGH).values
hr = SA.HR_H * (D.M.values / SA.M_H) ** -0.25
rS = D.r.values / 100; QS = D.Q.values * 1e-6
tau = 4 * SA.MU * QS / (np.pi * rS ** 3); G = G_tube(alpha_of(rS, hr))
phis = np.where(high, PHI_HIGH, PHI_LOW)
hc = D[(D.species == "Homo sapiens") & (D.artery == "common carotid")]
RC, QC = float(hc.r.median()) / 100, float(hc.Q.median()) * 1e-6


def log_sigma(b, w):
    sf = lambda r, h, phi: float(1 + w * phi * G_tube(alpha_of(r, h))[0])
    SA.sensed_factor = sf
    gC = np.log2(SA.Q_H / QC); lC = SA.L0_H * 2 ** (-gC / 3)
    tau0 = 4 * SA.MU * QC / (np.pi * RC ** 3) * sf(RC, 70, PHI_LOW) / (RC ** (b - 1) * lC ** ((b - 1) / 2))
    out = np.empty(len(D))
    for M in D.M.unique():
        m = (D.M == M).values
        t = SA.tree(M, b, PHI_LOW, 0, tau0, g_step=0.5)
        l = np.exp(np.interp(np.log(rS[m]), np.log(t.r.values[::-1]), np.log(t.L.values[::-1])))
        out[m] = np.log10(tau[m] * (1 + w * phis[m] * G[m]) / (tau0 * rS[m] ** (b - 1) * l ** ((b - 1) / 2)))
    return out


rows = []
for w in (0.0, 0.2, 0.3, 0.38, 0.45, 0.6, 0.8, 1.0):
    e = log_sigma(0.70, w)
    rows.append(dict(w=w, rms_log_sigma=round(float(np.sqrt((e ** 2).mean())), 3),
                     median_sigma_low=round(float(10 ** np.median(e[~high])), 2), median_sigma_high=round(float(10 ** np.median(e[high])), 2),
                     high_over_low=round(float(10 ** (np.median(e[high]) - np.median(e[~high]))), 2)))
    print(rows[-1], flush=True)
W = pd.DataFrame(rows)
w_eq = float(np.interp(0.0, np.log10(W.high_over_low.values), W.w.values))
print(f"weight that equalizes low and high beds (Seymour data): w = {w_eq:.2f}")

# re-profile b at w_eq
bg = np.round(np.arange(0.55, 0.901, 0.025), 3)
rb = [float(np.sqrt((log_sigma(b, w_eq) ** 2).mean())) for b in bg]
b_best = bg[int(np.argmin(rb))]
print(f"best b at w = {w_eq:.2f}: {b_best} (rms {min(rb):.3f})")

# Reneman Table 3: invariance of S = mean + w (peak - mean) across four human beds
T3 = pd.DataFrame([("CCA", 1.3, 3.8, 1.2, 2.6), ("CFA", 0.4, 4.0, 0.3, 3.8), ("SFA", 0.5, 3.4, 0.5, 4.0), ("BA", 0.5, 3.6, 0.5, 3.3)],
                  columns=["bed", "mean_y", "peak_y", "mean_o", "peak_o"])
ws = np.linspace(0, 1.2, 25); cvy = []; cvo = []
for w in ws:
    Sy = T3.mean_y + w * (T3.peak_y - T3.mean_y); So = T3.mean_o + w * (T3.peak_o - T3.mean_o)
    cvy.append(float(Sy.std(ddof=0) / Sy.mean())); cvo.append(float(So.std(ddof=0) / So.mean()))
wy, wo = ws[int(np.argmin(cvy))], ws[int(np.argmin(cvo))]
at = lambda w: (float(np.interp(w, ws, cvy)), float(np.interp(w, ws, cvo)))
print(f"Reneman beds: CV minimal at w = {wy:.2f} (young, CV {min(cvy):.3f}) and {wo:.2f} (old, CV {min(cvo):.3f}); "
      f"CV at w = 0.38: {at(0.38)[0]:.3f}/{at(0.38)[1]:.3f}; at w = 0: {at(0)[0]:.3f}/{at(0)[1]:.3f}")
summary = pd.DataFrame([dict(source="Feaver in vitro (A1 weight 2.87 x harmonic content)", weight_on_peak_excursion=f"{w_vitro[0]:.2f}-{w_vitro[1]:.2f}"),
                        dict(source="Seymour 92 measurements, bed equalization", weight_on_peak_excursion=f"{w_eq:.2f}"),
                        dict(source="Reneman 4 human beds, minimal CV (young / old)", weight_on_peak_excursion=f"{wy:.2f} / {wo:.2f}")])
summary.to_csv(os.path.join(a.out, "amplitude_weight_summary.tsv"), sep="\t", index=False)
W.to_csv(os.path.join(a.out, "amplitude_weight_scan.tsv"), sep="\t", index=False)
print(summary.to_string(index=False))
