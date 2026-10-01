#!/usr/bin/env python3
"""Harmonic content of published human carotid and vertebral flow waveforms.

Feature points: Ford et al. 2005 (young adults, ICA and VA, PC-MRI, N = 17) and Hoi et al. 2010 (older adults, CCA,
ICA, ECA, N = 94); amplitudes normalised to the cycle mean, times from H0. Each waveform is rebuilt by a periodic cubic
spline, renormalised to mean 1 and Fourier analysed.

Wall shear follows from Womersley theory with the COMPLEX transfer function T (magnitude and phase):
    tau_hat_k = (4 mu / pi r^3) Q_hat_k T(alpha sqrt k)
The amplitude of each shear harmonic needs only |T|; the time course and peak need the full T. (The previous version
multiplied by |T| only, which misplaces the harmonics in time and biases the peak excursion and c = A1/(peak - mean).)

Outputs per waveform:
    phi1            |Q_hat1| / Q_mean, first-harmonic amplitude of flow relative to the mean (the pulsatility index used
                    throughout the analysis: trees, predicted flow, Sigma)
    wss_A1          |tau_hat1| / tau_mean = phi1 |T(alpha1)|
    peak excursion  (tau_peak - tau_mean) / tau_mean with phase, all harmonics; also with 8 harmonics and magnitude only
                    (previous method) for comparison
    c_wss           A1 / peak excursion (harmonic content; no longer needed by the model, kept for the record)
    sensed/mean     S / tau_mean = 1 + w1 |tau_hat1| / tau_mean with the in vitro weight w1 (Feaver et al. 2013)
Radii: ICA 2.3 mm, VA 1.7 mm, CCA 3.3 mm, ECA 2.0 mm (typical adult values; assumption).

    python3 src/carotid_harmonics.py --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigma_exact_lib import T_tube
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--w1", type=float, default=2.9)
a = ap.parse_args()
MU, RHO = 4e-3, 1060.0
R = {"ICA": 2.3e-3, "VA": 1.7e-3, "CCA": 3.3e-3, "ECA": 2.0e-3}
HERE = os.path.dirname(os.path.abspath(__file__))
D = pd.read_csv(os.path.join(HERE, "..", "data", "carotid_waveforms_feature_points.tsv"), sep="\t")
N = 2048
rows = []
for (src, v, grp), d in D.groupby(["source", "vessel", "group"], sort=False):
    t = d.t_ms.values.astype(float); y = d.amp.values.astype(float); hr = float(d.HR_bpm.iloc[0])
    T = t[-1] - t[0]
    cs = CubicSpline(np.append(t[:-1], t[0] + T), np.append(y[:-1], y[0]), bc_type="periodic")
    tt = np.linspace(t[0], t[0] + T, N, endpoint=False); q = cs(tt); q = q / q.mean()
    F = np.fft.rfft(q) / N                                   # complex flow harmonics, F[0] = 1
    A = 2 * np.abs(F); A[0] = 1.0                            # amplitudes relative to the mean
    alpha1 = R[v] * np.sqrt(2 * np.pi * (1000 / T) * RHO / MU)
    k = np.arange(len(F)); Tk = T_tube(alpha1 * np.sqrt(k)); Tk[0] = 1.0
    tau = np.fft.irfft(F * Tk * N, n=N)                      # wall shear / mean, with phase, all harmonics
    F8 = np.where(k <= 8, F, 0); tau_mag8 = np.fft.irfft(F8 * np.abs(Tk) * N, n=N)   # previous method
    wss_A1 = A[1] * abs(Tk[1])
    rows.append(dict(source=src, vessel=v, group=grp, HR=hr, period_ms=round(T), alpha1=round(alpha1, 2),
                     phase1_deg=round(float(np.degrees(np.angle(Tk[1]))), 1),
                     phi1=round(A[1], 3), flow_A2_8=round(float(np.sqrt((A[2:9] ** 2).sum())), 3),
                     flow_peak_minus_mean=round(q.max() - 1, 3),
                     wss_A1=round(wss_A1, 3), wss_A2_8=round(float(np.sqrt(((A[2:9] * np.abs(Tk[2:9])) ** 2).sum())), 3),
                     wss_peak_minus_mean=round(tau.max() - 1, 3), wss_peak_minus_mean_prev=round(tau_mag8.max() - 1, 3),
                     c_flow=round(A[1] / (q.max() - 1), 3), c_wss=round(wss_A1 / (tau.max() - 1), 3),
                     c_wss_prev=round(A[1] * abs(Tk[1]) / (tau_mag8.max() - 1), 3),
                     sensed_over_mean=round(1 + a.w1 * wss_A1, 3)))
T = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True); T.to_csv(os.path.join(a.out, "carotid_harmonics.tsv"), sep="\t", index=False)
pd.set_option("display.width", 300); print(T.to_string(index=False))
y_ = T[(T.vessel == "ICA") & (T.group == "young")].iloc[0]; o_ = T[(T.vessel == "ICA") & (T.group == "older")].iloc[0]
ratio = y_.sensed_over_mean / o_.sensed_over_mean
b = 0.675
print(f"\nICA older vs young at the same target (S = tau_mean + {a.w1} |tau_hat1|): mean WSS ratio {ratio:.3f}; "
      f"radius ratio at fixed flow {ratio ** (-1 / (2 + b)):.3f}, with flow 245/274 {(ratio ** -1 * 245 / 274) ** (1 / (2 + b)):.3f} "
      f"(b = {b}; local approximation, target held at the young radius -- the full equilibrium is work-plan item 4)")
