#!/usr/bin/env python3
"""Canonical pipeline: one set of physics and choices for every headline number of the allometry paper.

Audit finding that motivated it: earlier scripts mixed two sensing models (a heuristic Womersley factor with phi = 0.8
and aortic calibration: systemic_allometry, fit_seymour, robustness_seymour, human_tests; and the exact tube Womersley
factor with a linear law, phi from Reneman waveforms and carotid calibration: sigma_exact, amplitude_weight,
size_dependent_b, feaver_rabbit), so b ranged 0.65-0.80 depending on the script. Here everything uses:

  sensed shear   S = tau_mean (1 + w phi G(alpha))       G = exact Womersley tube ratio (sigma_exact_lib.G_tube)
  phi            1.32 (low-resistance beds; carotid waveform, Reneman 2009 Table 3) and 5.56 (limb beds)
  w              weight on the peak excursion: 0.12 (bed equalisation, Seymour), 0.40 (in vitro converted / older
                 adults), 1.0 (young adults); reported as a systematic range
  set point      tau0 r^(b-1) l^((b-1)/2), tau0 calibrated on the human common carotid (median of Table S1)
  tree           symmetric, Q ~ M^3/4, HR ~ M^-1/4, lengths 30 cm x (M/70)^(1/3) x 2^(-g/3)
  uncertainty    bootstrap over the 51 source studies (1,000 resamples)

    python3 src/canonical_results.py --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systemic_allometry as SA
from sigma_exact_lib import G_tube, alpha_of

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args()
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
gC = np.log2(SA.Q_H / QC); LC = SA.L0_H * 2 ** (-gC / 3)


def setup(b, w):
    sf = lambda r, h, phi: float(1 + w * phi * G_tube(alpha_of(r, h))[0])
    SA.sensed_factor = sf; SA.VISC = "constant"
    return 4 * SA.MU * QC / (np.pi * RC ** 3) * sf(RC, 70, PHI_LOW) / (RC ** (b - 1) * LC ** ((b - 1) / 2))


def trees(b, w, masses, step=0.5):
    tau0 = setup(b, w)
    return tau0, {M: SA.tree(M, b, PHI_LOW, 0, tau0, g_step=step) for M in masses}


def log_sigma(b, w):
    tau0, T = trees(b, w, D.M.unique())
    out = np.empty(len(D))
    for M, t in T.items():
        m = (D.M == M).values
        l = np.exp(np.interp(np.log(rS[m]), np.log(t.r.values[::-1]), np.log(t.L.values[::-1])))
        out[m] = np.log10(tau[m] * (1 + w * phis[m] * GS[m]) / (tau0 * rS[m] ** (b - 1) * l ** ((b - 1) / 2)))
    return out


bgrid = np.round(np.arange(0.50, 0.951, 0.025), 3)
rng = np.random.default_rng(11)
studies = D.study.values; us = np.unique(studies)
rows = []
for w in (0.12, 0.40, 1.00):
    E = np.array([log_sigma(b, w) for b in bgrid])
    rms = np.sqrt((E ** 2).mean(1)); ib = int(rms.argmin()); b = float(bgrid[ib])
    bs = []
    for _ in range(a.boot):
        idx = np.concatenate([np.where(studies == u)[0] for u in rng.choice(us, len(us))])
        bs.append(bgrid[int(np.sqrt((E[:, idx] ** 2).mean(1)).argmin())])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    e = E[ib]
    # human tree: local exponents, carotid WSS, bed ratio
    tau0, T = trees(b, w, [70.0], step=0.1); t = T[70.0]
    lr, lq = np.log(t.r.values), np.log(t.Q.values); n = np.diff(lq) / np.diff(lr); rm = np.exp(0.5 * (lr[1:] + lr[:-1]))
    nloc = lambda r1, r2: float(n[(rm >= r1) & (rm <= r2)].mean())
    tau_at = lambda r: float(np.exp(np.interp(np.log(r), lr[::-1], np.log(t.tau_mean.values[::-1]))))
    S = lambda r, phi: 1 + w * phi * G_tube(alpha_of(r, 70))[0]
    ratio_cf = (0.0033 ** (b - 1)) / (0.0039 ** (b - 1)) * S(0.0039, PHI_HIGH) / S(0.0033, PHI_LOW)
    # cross-species peak-like set point ratios (mouse descending aorta, rabbit aorta) vs human carotid
    tau0, T2 = trees(b, w, [0.025, 2.75])
    def sp_ratio(M, r):
        t2 = T2[M]; l = np.exp(np.interp(np.log(r), np.log(t2.r.values[::-1]), np.log(t2.L.values[::-1])))
        return float((r ** (b - 1) * l ** ((b - 1) / 2)) / (RC ** (b - 1) * LC ** ((b - 1) / 2)))
    rows.append(dict(w=w, b=b, b_CI95_by_study=f"{lo:.3f}-{hi:.3f}", rms_log_sigma=round(float(rms[ib]), 3),
                     sd_log_sigma=round(float(e.std()), 3), sd_log_raw_wss=round(float(np.log10(tau).std()), 3),
                     variance_removed_pct=round(100 * (1 - e.var() / np.log10(tau).var()), 0),
                     median_sigma_low_high=f"{10 ** np.median(e[~high]):.2f} / {10 ** np.median(e[high]):.2f}",
                     n_coronary_r075_2mm=round(nloc(0.75e-3, 2e-3), 2), n_retinal_r30_60um=round(nloc(30e-6, 60e-6), 2),
                     human_CCA_mean_WSS_Pa=round(tau_at(RC), 2), carotid_over_femoral_WSS=round(float(ratio_cf), 2),
                     setpoint_mouse_vs_human=round(sp_ratio(0.025, np.sqrt(1.2e-6 / np.pi)), 2),
                     setpoint_rabbit_vs_human=round(sp_ratio(2.75, 1.4e-3), 2)))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out, "canonical_results.tsv"), sep="\t", index=False)
pd.set_option("display.width", 300); print(R.T.to_string())
print("data: coronary 2.39 (2.24-2.54); retinal 2.76; CCA WSS 1.1-1.3 Pa (CIRCULAR under carotid calibration: tau0 fixes the "
      "sensed CCA shear at the median Table S1 carotid, so the CCA mean is not a test here); carotid/femoral 2.2-4.3; peak ratios mouse 1.58-2.1, rabbit 1.16-1.32")
