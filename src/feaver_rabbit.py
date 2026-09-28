#!/usr/bin/env python3
"""(1) Sensing laws against Feaver et al. 2013 (20 cone-and-plate waveforms, NF-kB fold; Supplementary Tables S1-S2).
A saturating response NF = N_inf + (N_0 - N_inf) exp(-S / S*) of a single sensed shear S is fitted for each law;
Feaver's own 4-coefficient regression (Model 2: NF = 3.66 - 0.16 A0 - 0.63 A1 + 0.05 A0 A1, R2 = 0.89) is the benchmark.
Amplitudes in dyn/cm2. (2) Rabbit aorta (Riemer et al. 2020, ultrasound image velocimetry, 10 NZW rabbits 2.2-3.3 kg):
diameter 2.8 mm, time-averaged WSS 0.54 Pa (Poiseuille estimate 0.46 Pa), peak 4.4-5 Pa; compared with the model and
with peak WSS in human beds (Reneman 2009: 3.4-4.0 Pa) and the mouse arch (Winter 2019: 6-8 Pa).

    python3 src/feaver_rabbit.py --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
F = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "feaver2013_tableS1.tsv"), sep="\t")
y = F.NFkB_obs.values; A0, A1, A2, mx = F.A0.values, F.A1.values, F.A2_8.values, F["max"].values
sst = ((y - y.mean()) ** 2).sum()
feaver = 3.66 - 0.16 * A0 - 0.63 * A1 + 0.05 * A0 * A1
R2_feaver = 1 - ((y - feaver) ** 2).sum() / sst


def fit(Sfun, npar_extra):
    def res(p):
        S = Sfun(p[3:]); return p[1] + (p[0] - p[1]) * np.exp(-S / abs(p[2])) - y
    best = None
    for s0 in (1, 3, 10):
        p0 = [3.8, 1.3, s0] + [1.0] * npar_extra
        r = least_squares(res, p0)
        if best is None or r.cost < best.cost: best = r
    rss = 2 * best.cost; k = 3 + npar_extra; n = len(y)
    return dict(R2=1 - rss / sst, AIC=n * np.log(rss / n) + 2 * k, params=np.round(best.x, 3).tolist(), k=k)


laws = {
    "mean only (A0)": (lambda q: A0, 0),
    "linear: A0 + w A1": (lambda q: A0 + abs(q[0]) * A1, 1),
    "linear: A0 + w A1 + v A2-8": (lambda q: A0 + abs(q[0]) * A1 + abs(q[1]) * A2, 2),
    "rms of 0th and 1st: sqrt(A0^2 + A1^2/2)": (lambda q: np.sqrt(A0 ** 2 + A1 ** 2 / 2), 0),
    "peak shear (max of waveform)": (lambda q: mx, 0),
}
rows = [dict(law="Feaver regression (4 coefficients)", R2=round(R2_feaver, 3), k=4, AIC=round(len(y) * np.log(((y - feaver) ** 2).sum() / len(y)) + 8, 2), params="3.66, 0.16, 0.63, 0.05")]
for name, (fun, k) in laws.items():
    r = fit(fun, k); rows.append(dict(law=name, R2=round(r["R2"], 3), k=r["k"], AIC=round(r["AIC"], 2), params=str(r["params"])))
T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "feaver_sensing_laws.tsv"), sep="\t", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
print(T.to_string(index=False))

# (2) rabbit aorta vs model
import systemic_allometry as SA
from sigma_exact_lib import G_tube, alpha_of
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "seymour2019_points.tsv"), sep="\t")
hc = D[(D.species == "Homo sapiens") & (D.artery == "common carotid")]
RC, QC = float(hc.r.median()) / 100, float(hc.Q.median()) * 1e-6
PHI = 1.32
def sf(r, hr, phi): return float(1 + phi * G_tube(alpha_of(r, hr))[0])
SA.sensed_factor = sf
out = []
for b in (0.70, 0.75):
    g = np.log2(SA.Q_H / QC); L = SA.L0_H * 2 ** (-g / 3)
    tau0 = 4 * SA.MU * QC / (np.pi * RC ** 3) * sf(RC, 70, PHI) / (RC ** (b - 1) * L ** ((b - 1) / 2))
    for sp, M, r in (("rabbit", 2.75, 1.4e-3), ("mouse", 0.025, np.sqrt(1.2e-6 / np.pi))):
        t = SA.tree(M, b, PHI, 0, tau0, g_step=0.25)
        Lr = np.exp(np.interp(np.log(r), np.log(t.r.values[::-1]), np.log(t.L.values[::-1])))
        Q = np.exp(np.interp(np.log(r), np.log(t.r.values[::-1]), np.log(t.Q.values[::-1])))
        ratio = (r ** (b - 1) * Lr ** ((b - 1) / 2)) / (RC ** (b - 1) * L ** ((b - 1) / 2))
        out.append(dict(b=b, species=sp, radius_mm=round(r * 1e3, 2), model_mean_WSS_Pa=round(4 * SA.MU * Q / (np.pi * r ** 3), 2),
                        setpoint_ratio_vs_human_CCA=round(float(ratio), 2)))
R = pd.DataFrame(out); R.to_csv(os.path.join(a.out, "rabbit_mouse_check.tsv"), sep="\t", index=False)
print(R.to_string(index=False))
print("observed: rabbit mean 0.54 Pa (Poiseuille 0.46), peak 4.4-5.0 Pa; mouse mean 1.52 Pa (MRI), peak 6-8 Pa; "
      "human peak 3.4-4.0 Pa (CCA 3.8) -> observed peak ratios rabbit 1.16-1.32, mouse 1.58-2.1 vs human CCA")
