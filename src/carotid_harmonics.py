#!/usr/bin/env python3
"""Harmonic content of published human carotid/vertebral flow waveforms (feature points: Ford et al. 2005, young adults,
PC-MRI, N = 17; Hoi et al. 2010, older adults, PC-MRI, N = 94; amplitudes normalised to cycle mean, times from H0).
Each waveform is rebuilt by a periodic cubic spline (D4 closes the cycle onto M0), renormalised to mean 1, and Fourier
analysed. Wall shear harmonics follow from Womersley: the k-th harmonic of wall shear, relative to its quasi-steady value,
is G(alpha sqrt k). Outputs: flow and wall-shear first harmonic, peak excursion, the ratio A1/(peak - mean) that converts
the in vitro weight (Feaver: S = A0 + 2.87 A1) to a weight on the peak excursion, and the predicted age effect.
Radii: ICA 2.3 mm, VA 1.7 mm, CCA 3.3 mm, ECA 2.0 mm (typical adult values; assumption)."""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigma_exact_lib import G_tube
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
MU, RHO = 4e-3, 1060.0
R = {"ICA": 2.3e-3, "VA": 1.7e-3, "CCA": 3.3e-3, "ECA": 2.0e-3}
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "carotid_waveforms_feature_points.tsv"), sep="\t")
rows = []
for (src, v, grp), d in D.groupby(["source", "vessel", "group"], sort=False):
    t = d.t_ms.values.astype(float); y = d.amp.values.astype(float); hr = float(d.HR_bpm.iloc[0])
    T = t[-1] - t[0]
    cs = CubicSpline(np.append(t[:-1], t[0] + T), np.append(y[:-1], y[0]), bc_type="periodic")
    tt = np.linspace(t[0], t[0] + T, 2048, endpoint=False); q = cs(tt); q = q / q.mean()
    F = np.fft.rfft(q) / len(q); A = 2 * np.abs(F); A[0] = 1.0
    alpha1 = R[v] * np.sqrt(2 * np.pi * (1000 / T) * RHO / MU)
    Gk = np.array([1.0] + [float(G_tube(np.array([alpha1 * np.sqrt(k)]))[0]) for k in range(1, 9)])
    tau = np.real(np.fft.irfft(np.concatenate([[F[0]], F[1:9] * Gk[1:9], np.zeros(len(F) - 9)]) * len(q), n=len(q)))
    tau = tau / tau.mean()
    rows.append(dict(source=src, vessel=v, group=grp, HR=hr, period_ms=round(T), alpha1=round(alpha1, 2),
                     flow_A1=round(A[1], 3), flow_A2_8=round(float(np.sqrt((A[2:9] ** 2).sum())), 3), flow_peak_minus_mean=round(q.max() - 1, 3),
                     wss_A1=round(A[1] * Gk[1], 3), wss_peak_minus_mean=round(tau.max() - 1, 3),
                     c_flow=round(A[1] / (q.max() - 1), 3), c_wss=round(A[1] * Gk[1] / (tau.max() - 1), 3),
                     sensed_factor_feaver=round(1 + 2.87 * A[1] * Gk[1], 3)))
T = pd.DataFrame(rows); os.makedirs(a.out, exist_ok=True); T.to_csv(os.path.join(a.out, "carotid_harmonics.tsv"), sep="\t", index=False)
pd.set_option("display.width", 250); print(T.to_string(index=False))
y_ = T[(T.vessel == "ICA") & (T.group == "young")].iloc[0]; o_ = T[(T.vessel == "ICA") & (T.group == "older")].iloc[0]
ratio = y_.sensed_factor_feaver / o_.sensed_factor_feaver
b = 0.675
print(f"ICA older vs young, same set point: mean WSS ratio {ratio:.3f}; at fixed flow radius ratio {ratio ** (-1 / (2 + b)):.3f}"
      f" (b = {b}); with flow falling 245/274: radius ratio {(ratio ** -1 * (245 / 274)) ** (1 / (2 + b)):.3f}")
