#!/usr/bin/env python3
"""Canonical pipeline: one set of physics and choices for every headline number of the paper.

Sensing law (default, --law harmonic): the SAME law in vitro and in vivo, written on the harmonics of wall shear,
      S = tau_mean + w1 |tau_hat1|,      |tau_hat1| / tau_mean = phi1 |T(alpha)|,
  with T the complex Womersley transfer function (sigma_exact_lib.T_tube; only |T| is needed for an amplitude), phi1 the
  first-harmonic amplitude of flow relative to its mean, by territory (data/phi1_by_territory.tsv, from measured
  waveforms), and w1 the in vitro weight of Feaver et al. 2013 (2.9), applied WITHOUT refitting. The in vivo analysis
  therefore has no free sensing parameter: b is the only fitted exponent, tau0 the single calibration constant.
Legacy law (--law peak): S = tau_mean (1 + w phi G), phi from the peak excursion of Reneman 2009 and w in {0.12, 0.4, 1}.

  set point      tau0 r^(b-1) l^((b-1)/2), tau0 calibrated on the human common carotid (median of Seymour table S1)
  tree           symmetric, Q ~ M^3/4, HR ~ M^-1/4, lengths 30 cm x (M/70)^(1/3) x 2^(-g/3), built with phi1(low resistance)
  uncertainty    bootstrap over the 51 source studies

    python3 src/canonical_results.py --out results [--law harmonic|peak] [--w1 2.9] [--phi1-low X] [--boot 1000]
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systemic_allometry as SA
from sigma_exact_lib import G_tube, alpha_of

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results"); ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--law", choices=["harmonic", "peak"], default="harmonic")
ap.add_argument("--w1", type=float, default=2.9, help="in vitro first-harmonic weight (Feaver 2013)")
ap.add_argument("--phi1-low", type=float, default=None, help="override phi1 of the low-resistance territory")
ap.add_argument("--phi1-limb", type=float, default=None, help="override phi1 of the limb territory")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
HIGH = {"femoral", "superficial femoral", "brachial", "iliac", "interdigital", "lateral digital", "abdominal aorta", "supraceliac aorta"}

if a.law == "harmonic":
    P = pd.read_csv(os.path.join(HERE, "..", "data", "phi1_by_territory.tsv"), sep="\t").set_index("territory").phi1
    PHI_LOW = a.phi1_low if a.phi1_low is not None else float(P["low_resistance"])
    PHI_HIGH = a.phi1_limb if a.phi1_limb is not None else float(P["limb"])
    WEIGHTS = [a.w1]
else:   # legacy: phi from peak excursion (Reneman 2009 table 3, young), weight on the peak excursion
    PHI_LOW = 1.9 / float(G_tube(alpha_of(0.0033, 70))[0])
    PHI_HIGH = float(np.mean([9.0 / G_tube(alpha_of(0.0039, 70))[0], 6.2 / G_tube(alpha_of(0.0020, 70))[0]]))
    WEIGHTS = [0.12, 0.40, 1.00]
print(f"law = {a.law}; phi (low / limb) = {PHI_LOW:.3f} / {PHI_HIGH:.3f}; weights = {WEIGHTS}")

D = pd.read_csv(os.path.join(HERE, "..", "data", "seymour2019_points.tsv"), sep="\t")
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


def predicted_flow_error(b, w):
    """log10(measured / predicted flow) at the measured radius, from the tree of the animal's body mass."""
    tau0, T = trees(b, w, D.M.unique())
    out = np.empty(len(D))
    for M, t in T.items():
        m = (D.M == M).values
        out[m] = np.log10(QS[m]) - np.interp(np.log(rS[m]), np.log(t.r.values[::-1]), np.log10(t.Q.values[::-1]))
    return out


bgrid = np.round(np.arange(0.50, 0.951, 0.025), 3)
rng = np.random.default_rng(11)
studies = D.study.values; us = np.unique(studies)
rows = []
for w in WEIGHTS:
    E = np.array([log_sigma(b, w) for b in bgrid])
    rms = np.sqrt((E ** 2).mean(1)); ib = int(rms.argmin()); b = float(bgrid[ib])
    bs = []
    for _ in range(a.boot):
        idx = np.concatenate([np.where(studies == u)[0] for u in rng.choice(us, len(us))])
        bs.append(bgrid[int(np.sqrt((E[:, idx] ** 2).mean(1)).argmin())])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    e = E[ib]
    fe = predicted_flow_error(b, w); rmsQ = float(np.sqrt((fe ** 2).mean()))
    # human tree: local exponents, carotid WSS, bed ratio
    tau0, T = trees(b, w, [70.0], step=0.1); t = T[70.0]
    lr, lq = np.log(t.r.values), np.log(t.Q.values); n = np.diff(lq) / np.diff(lr); rm = np.exp(0.5 * (lr[1:] + lr[:-1]))
    nloc = lambda r1, r2: float(n[(rm >= r1) & (rm <= r2)].mean())
    tau_at = lambda r: float(np.exp(np.interp(np.log(r), lr[::-1], np.log(t.tau_mean.values[::-1]))))
    S = lambda r, phi: 1 + w * phi * G_tube(alpha_of(r, 70))[0]
    ratio_cf = (0.0033 ** (b - 1)) / (0.0039 ** (b - 1)) * S(0.0039, PHI_HIGH) / S(0.0033, PHI_LOW)
    tau0, T2 = trees(b, w, [0.025, 2.75])
    def sp_ratio(M, r):
        t2 = T2[M]; l = np.exp(np.interp(np.log(r), np.log(t2.r.values[::-1]), np.log(t2.L.values[::-1])))
        return float((r ** (b - 1) * l ** ((b - 1) / 2)) / (RC ** (b - 1) * LC ** ((b - 1) / 2)))
    rows.append(dict(law=a.law, w=w, b=b, b_CI95_by_study=f"{lo:.3f}-{hi:.3f}", rms_log_sigma=round(float(rms[ib]), 3),
                     rms_log10_flow=round(rmsQ, 3), flow_factor=round(10 ** rmsQ, 2),
                     frac_within_2x=round(float((np.abs(fe) <= np.log10(2)).mean()), 2), frac_within_3x=round(float((np.abs(fe) <= np.log10(3)).mean()), 2),
                     sd_log_sigma=round(float(e.std()), 3), sd_log_raw_wss=round(float(np.log10(tau).std()), 3),
                     variance_removed_pct=round(100 * (1 - e.var() / np.log10(tau).var()), 0),
                     median_sigma_low_high=f"{10 ** np.median(e[~high]):.2f} / {10 ** np.median(e[high]):.2f}",
                     n_coronary_r075_2mm=round(nloc(0.75e-3, 2e-3), 2), n_retinal_r30_60um=round(nloc(30e-6, 60e-6), 2),
                     human_CCA_mean_WSS_Pa=round(tau_at(RC), 2), human_CCA_sensed_over_mean=round(S(RC, PHI_LOW), 2),
                     carotid_over_femoral_WSS=round(float(ratio_cf), 2),
                     setpoint_mouse_vs_human=round(sp_ratio(0.025, np.sqrt(1.2e-6 / np.pi)), 2),
                     setpoint_rabbit_vs_human=round(sp_ratio(2.75, 1.4e-3), 2)))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True)
R.to_csv(os.path.join(a.out, f"canonical_results{'' if a.law == 'harmonic' else '_peaklaw'}.tsv"), sep="\t", index=False)
pd.set_option("display.width", 300); print(R.T.to_string())
print("data: coronary 2.39 (2.24-2.54); retinal 2.76; carotid/femoral 2.2-4.3 (PROVISIONAL phi1_limb -> not yet an external test); "
      "peak ratios mouse 1.58-2.1, rabbit 1.16-1.32. The CCA mean WSS is fixed by calibration, not a test.")
