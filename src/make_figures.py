#!/usr/bin/env python3
"""Main figures of "An invariant endothelial flow signal across species" (Figures 1-4).
Canonical model: exact Womersley tube factor, linear sensing S = tau_mean (1 + w phi G), w = 0.4, phi 1.32 (low-resistance
beds) / 5.56 (limb beds), target constant fixed on the human common carotid, b profiled (best 0.675).
Outputs: results/figures/rsif_fig1..4 (.png 300 dpi, .pdf vector, .tif 600 dpi). Run the analysis scripts first.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systemic_allometry as SA
from sigma_exact_lib import G_tube, alpha_of

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "savefig.dpi": 300})
C = dict(blue="#1F5FA8", orange="#C0392B", green="#2E8B57", red="#B2182B", purple="#4A90C8", grey="#7F7F7F", sky="#A8C8E6")
W, B = 0.4, 0.675
HIGH = {"femoral", "superficial femoral", "brachial", "iliac", "interdigital", "lateral digital", "abdominal aorta", "supraceliac aorta"}
PHI_LOW = 1.9 / float(G_tube(alpha_of(0.0033, 70))[0])
PHI_HIGH = float(np.mean([9.0 / G_tube(alpha_of(0.0039, 70))[0], 6.2 / G_tube(alpha_of(0.0020, 70))[0]]))
D = pd.read_csv(os.path.join(ROOT, "data", "seymour2019_points.tsv"), sep="\t")
high = D.artery.str.strip().isin(HIGH).values
rS = D.r.values / 100; QS = D.Q.values * 1e-6; tau = 4 * SA.MU * QS / (np.pi * rS ** 3)
hrD = SA.HR_H * (D.M.values / SA.M_H) ** -0.25; GS = G_tube(alpha_of(rS, hrD))
phis = np.where(high, PHI_HIGH, PHI_LOW)
hc = D[(D.species == "Homo sapiens") & (D.artery == "common carotid")]
RC, QC = float(hc.r.median()) / 100, float(hc.Q.median()) * 1e-6
LC = SA.L0_H * 2 ** (-np.log2(SA.Q_H / QC) / 3)
SPC = {"Mus musculus": "mouse", "Rattus norvegicus": "rat", "Oryctolagus cuniculus": "rabbit", "Felis catus": "cat",
       "Macaca fascicularis": "macaque", "Canis familiaris": "dog", "Homo sapiens": "human", "Equus caballus": "horse", "Bos taurus": "cow"}


def setup(b, w):
    sf = lambda r, h, phi: float(1 + w * phi * G_tube(alpha_of(r, h))[0])
    SA.sensed_factor = sf; SA.VISC = "constant"
    return 4 * SA.MU * QC / (np.pi * RC ** 3) * sf(RC, 70, PHI_LOW) / (RC ** (b - 1) * LC ** ((b - 1) / 2))


def trees(b, w, step=0.5):
    tau0 = setup(b, w)
    return tau0, {M: SA.tree(M, b, PHI_LOW, 0, tau0, g_step=step) for M in D.M.unique()}


def sigma_and_pred(b, w):
    tau0, T = trees(b, w)
    ls = np.empty(len(D)); lq = np.empty(len(D))
    for M, t in T.items():
        m = (D.M == M).values
        lr = np.log(t.r.values[::-1])
        l = np.exp(np.interp(np.log(rS[m]), lr, np.log(t.L.values[::-1])))
        ls[m] = np.log10(tau[m] * (1 + w * phis[m] * GS[m]) / (tau0 * rS[m] ** (b - 1) * l ** ((b - 1) / 2)))
        lq[m] = np.interp(np.log(rS[m]), lr, np.log10(t.Q.values[::-1] * 1e6))
    return ls, lq


print("computing profiles ...", flush=True)
bgrid = np.round(np.arange(0.50, 1.001, 0.025), 3)
prof = {}
for lab, w in (("pulsatile sensing", W), ("mean shear only", 0.0)):
    prof[lab] = np.array([np.sqrt((sigma_and_pred(b, w)[0] ** 2).mean()) for b in bgrid])
ls_best, lq_pred = sigma_and_pred(B, W)
ls_murray, lq_murray = sigma_and_pred(1.0, 0.0)
cols = D.species.map(lambda s: SPC.get(s, s))
masses = D.groupby("species").M.median().sort_values()
SPECIES_COLORS = ["#1B3A8A", "#3A7BD5", "#4FB3D9", "#1A9E77", "#7CB342", "#E6AB02", "#E6550D", "#C0392B", "#8E44AD"]
col_of = {sp: SPECIES_COLORS[i] for i, sp in enumerate(masses.index)}   # ordered by body mass, mouse -> cow

# ---------------- Figure 1
fig = plt.figure(figsize=(6.85, 5.1))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32)
ax = fig.add_subplot(gs[0, 0])
for art, mk, lab in (("abdominal aorta", "o", "abdominal aorta"), ("common carotid", "s", "common carotid")):
    d = D[D.artery.str.strip() == art]
    ax.scatter(d.M, d.tau.astype(float) / 10, marker=mk, s=16, c=[col_of[s] for s in d.species], edgecolor="k", lw=0.3, label=lab)
ax.set(xscale="log", yscale="log", xlabel="Body mass (kg)", ylabel="Mean wall shear stress (Pa)")
ax.set_title("A  The same artery, twentyfold apart", loc="left", fontweight="bold")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="0.75", mec="k", mew=0.3, label="abdominal aorta"),
                   Line2D([], [], marker="s", ls="", mfc="0.75", mec="k", mew=0.3, label="common carotid")], loc="upper right")
ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
ax.set_title("B  One target, sensed through the pulse", loc="left", fontweight="bold")
boxes = [(0.00, 0.60, "Target scales with size\n\u03c4* \u221d r$^{b-1}$ \u2113$^{(b-1)/2}$", C["blue"]),
         (0.00, 0.20, "Cells sense the pulse\nS = \u03c4$_{mean}$[1 + w\u03c6G(\u03b1)]", C["orange"]),
         (0.66, 0.42, "S = \u03c4*\none target,\nall mammals", C["green"])]
for x, y, txt, c in boxes:
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=6.6, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=c, lw=1.2))
for y0 in (0.70, 0.30):
    ax.annotate("", xy=(0.64, 0.52), xytext=(0.52, y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
ax.text(0.0, 0.02, "sensors: VEGFR3, curvature, Cx37/Cx40,\nPECAM-1, Piezo1, glycocalyx", transform=ax.transAxes, fontsize=6, color=C["grey"])
ax = fig.add_subplot(gs[1, 0])
for sp in masses.index:
    m = (D.species == sp).values
    ax.scatter(lq_pred[m], np.log10(D.Q.values[m]), s=10, color=col_of[sp], edgecolor="k", lw=0.25, label=SPC[sp])
lim = [min(lq_pred.min(), np.log10(D.Q).min()) - 0.4, max(lq_pred.max(), np.log10(D.Q).max()) + 0.4]
ax.plot(lim, lim, "k-", lw=0.6); ax.fill_between(lim, np.array(lim) - np.log10(2), np.array(lim) + np.log10(2), color="0.9", zorder=0)
ax.set(xlim=lim, ylim=lim, xlabel="Predicted flow, log$_{10}$ (mL s$^{-1}$)", ylabel="Measured flow, log$_{10}$ (mL s$^{-1}$)")
ax.set_title("C  92 arteries, 9 species", loc="left", fontweight="bold")
ax.legend(ncol=2, loc="upper left", fontsize=5.8, handletextpad=0.2, columnspacing=0.6)
ax.text(0.97, 0.05, "grey band: factor 2", transform=ax.transAxes, ha="right", fontsize=6, color=C["grey"])
ax = fig.add_subplot(gs[1, 1])
ax.axvspan(0.55, 0.725, color=C["sky"], alpha=0.25, lw=0)
ax.plot(bgrid, prof["pulsatile sensing"], "-", color=C["blue"], label="pulsatile sensing")
ax.plot(bgrid, prof["mean shear only"], "--", color=C["grey"], label="mean shear only")
ax.scatter([1.0], [np.sqrt((ls_murray ** 2).mean())], color=C["red"], zorder=5, s=18, label="Murray (b = 1, mean only)")
for x, lab in ((2 / 3, "Rubner 2/3"), (0.75, "Kleiber 3/4")):
    ax.axvline(x, color="k", lw=0.4, ls=":"); ax.text(x, ax.get_ylim()[1] if False else 0.02, "", fontsize=6)
ax.set(xlabel="Maintenance-cost exponent b", ylabel="Error, rms log$_{10}$ \u03a3")
ax.text(0.98, 0.35, "cross-species data fix b;\npulse is required by\nFigs. 2C\u2013D, 3, 4", transform=ax.transAxes, ha="right", fontsize=5.8, color=C["grey"])
ax.set_title("D  The data fix b near 2/3", loc="left", fontweight="bold")
ax.legend(loc="upper left", fontsize=6); ax.text(0.56, 0.55, "95% CI\n(by study)", transform=ax.get_xaxis_transform(), color=C["blue"], fontsize=5.8)
ax.text(2/3, 0.02, "2/3", transform=ax.get_xaxis_transform(), fontsize=5.8); ax.text(0.755, 0.02, "3/4", transform=ax.get_xaxis_transform(), fontsize=5.8)
fig.savefig(os.path.join(OUT, "rsif_fig1.png"), bbox_inches="tight", dpi=300); fig.savefig(os.path.join(OUT, "rsif_fig1.tif"), bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(os.path.join(OUT, "rsif_fig1.pdf"), bbox_inches="tight")
print("fig1", flush=True)

# ---------------- Figure 2
fig, axs = plt.subplots(2, 2, figsize=(6.85, 5.1)); plt.subplots_adjust(hspace=0.45, wspace=0.32)
ax = axs[0, 0]
for sp in masses.index:
    m = (D.species == sp).values
    ax.scatter(D.M.values[m], tau[m], s=10, color=col_of[sp], edgecolor="k", lw=0.25)
med = D.assign(t=tau).groupby("species").agg(M=("M", "median"), t=("t", "median"))
ax.plot(med.M, med.t, "kD", ms=4, label="species median"); ax.legend(loc="lower left")
ax.set(xscale="log", yscale="log", xlabel="Body mass (kg)", ylabel="Mean wall shear stress (Pa)")
ax.set_title("A  Raw shear: 150-fold range", loc="left", fontweight="bold")
ax = axs[0, 1]
for sp in masses.index:
    m = (D.species == sp).values
    ax.scatter(D.M.values[m], 10 ** ls_best[m], s=10, color=col_of[sp], edgecolor="k", lw=0.25, label=SPC[sp])
ax.axhline(1, color="k", lw=0.6)
meds = D.assign(s=10 ** ls_best).groupby("species").agg(M=("M", "median"), s=("s", "median"))
ax.plot(meds.M, meds.s, "kD", ms=4)
ax.set(xscale="log", yscale="log", ylim=(0.1, 10), xlabel="Body mass (kg)", ylabel="\u03a3 = sensed shear / target")
ax.set_title("B  Normalized: collapse near 1", loc="left", fontweight="bold")
ax.text(0.03, 0.9, f"variance removed: {100 * (1 - ls_best.var() / np.log10(tau).var()):.0f}%", transform=ax.transAxes, fontsize=6.5)
ax.legend(ncol=3, loc="lower right", fontsize=5.6, handletextpad=0.1, columnspacing=0.4)
ax = axs[1, 0]
beds = ["CCA", "CFA", "SFA", "BA"]
T3r = pd.read_csv(os.path.join(ROOT, "data", "reneman2009_table3_human_beds.tsv"), sep="\t")
mean_y, peak_y = T3r.mean_WSS_young_Pa.tolist(), T3r.peak_WSS_young_Pa.tolist(); mean_o, peak_o = T3r.mean_WSS_old_Pa.tolist(), T3r.peak_WSS_old_Pa.tolist()
x = np.arange(4)
ax.bar(x - 0.2, mean_y, 0.38, color=C["grey"], label="mean"); ax.bar(x + 0.2, peak_y, 0.38, color=C["orange"], label="peak (includes the pulse)")
cv = lambda v: 100 * np.std(v) / np.mean(v)
ax.set_xticks(x); ax.set_xticklabels(["carotid", "common\nfemoral", "superficial\nfemoral", "brachial"])
ax.set(ylabel="Wall shear stress, young adults (Pa)", ylim=(0, 7.0))
ax.text(0.02, 0.77, f"variation across arteries: mean {cv(mean_y):.0f}%,\nwith pulse {cv(peak_y):.0f}% (older adults {cv(peak_o):.0f}%)",
        transform=ax.transAxes, fontsize=6, ha="left", va="top")
ax.set_title("C  Human arteries share one signal", loc="left", fontweight="bold"); ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
ax = axs[1, 1]
ws = np.linspace(0, 1.2, 61); w70 = 2 * np.pi * 70 / 60
Sf = lambda r, phi, w: 1 + w * phi * (np.sqrt(1) * 0 + G_tube(alpha_of(r, 70))[0])
ratio = [(0.0033 ** (B - 1) / 0.0039 ** (B - 1)) * Sf(0.0039, PHI_HIGH, w) / Sf(0.0033, PHI_LOW, w) for w in ws]
ax.axhspan(2.2, 4.3, color=C["sky"], alpha=0.3, lw=0, label="measured (Reneman 2008)")
ax.plot(ws, ratio, color=C["blue"], label="predicted, one target")
ax.axvspan(0.32, 0.43, color=C["orange"], alpha=0.2, lw=0); ax.text(0.33, 0.6, "in vitro\nweight", fontsize=6, color=C["orange"])
ax.set(xlabel="Weight of the pulsatile component, w", ylabel="Carotid / femoral mean shear", ylim=(0, 5.5))
ax.set_title("D  Pulsatility sets the bed ratio", loc="left", fontweight="bold"); ax.legend(loc="upper left")
fig.savefig(os.path.join(OUT, "rsif_fig2.png"), bbox_inches="tight", dpi=300); fig.savefig(os.path.join(OUT, "rsif_fig2.tif"), bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(os.path.join(OUT, "rsif_fig2.pdf"), bbox_inches="tight")
print("fig2", flush=True)

# ---------------- Figure 3
F = pd.read_csv(os.path.join(ROOT, "data", "feaver2013_tableS1.tsv"), sep="\t")
y = F.NFkB_obs.values; A0, A1 = F.A0.values, F.A1.values
res = lambda p, S: p[1] + (p[0] - p[1]) * np.exp(-S / abs(p[2])) - y
fit = least_squares(lambda p: res(p[:3], A0 + abs(p[3]) * A1), [3.8, 1.3, 3.0, 2.0])
p = fit.x; S = A0 + abs(p[3]) * A1
fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.4)); plt.subplots_adjust(wspace=0.7)
ax = axs[0]
sc = ax.scatter(A0, A1, c=y, cmap="Reds", s=28, edgecolor="k", lw=0.3)
cb = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.04); cb.set_label("NF-\u03baB (fold)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
ax.set(xlabel="Mean shear, A$_0$ (dyn cm$^{-2}$)", ylabel="1st harmonic, A$_1$ (dyn cm$^{-2}$)")
ax.set_title("A  20 waveforms", loc="left", fontweight="bold")
ax = axs[1]
ss = np.linspace(0, S.max() * 1.05, 200)
ax.plot(ss, p[1] + (p[0] - p[1]) * np.exp(-ss / abs(p[2])), color=C["blue"])
ax.scatter(S, y, s=16, color=C["orange"], edgecolor="k", lw=0.3)
ax.set(xlabel=f"Sensed shear, A$_0$ + {abs(p[3]):.1f} A$_1$", ylabel="NF-\u03baB (fold)")
ax.set_title("B  One saturating signal", loc="left", fontweight="bold")
ax = axs[2]
T = pd.read_csv(os.path.join(ROOT, "results", "feaver_sensing_laws.tsv"), sep="\t")
labs = {"mean only (A0)": "mean only", "peak shear (max of waveform)": "peak", "rms of 0th and 1st: sqrt(A0^2 + A1^2/2)": "rms",
        "linear: A0 + w A1": "mean + w A$_1$", "Feaver regression (4 coefficients)": "4-coef.\nregression"}
T = T[T.law.isin(labs)]
order = ["mean only (A0)", "peak shear (max of waveform)", "rms of 0th and 1st: sqrt(A0^2 + A1^2/2)", "linear: A0 + w A1", "Feaver regression (4 coefficients)"]
T = T.set_index("law").loc[order]
ax.bar(range(len(T)), T.R2, color=[C["grey"], C["grey"], C["sky"], C["blue"], "0.3"])
ax.set_xticks(range(len(T))); ax.set_xticklabels([labs[k] for k in T.index], rotation=35, ha="right")
ax.set(ylabel="R$^2$", ylim=(0, 1)); ax.set_title("C  Sensing laws", loc="left", fontweight="bold")
fig.savefig(os.path.join(OUT, "rsif_fig3.png"), bbox_inches="tight", dpi=300); fig.savefig(os.path.join(OUT, "rsif_fig3.tif"), bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(os.path.join(OUT, "rsif_fig3.pdf"), bbox_inches="tight")
print("fig3", flush=True)

# ---------------- Figure 4
WV = pd.read_csv(os.path.join(ROOT, "data", "carotid_waveforms_feature_points.tsv"), sep="\t")
H = pd.read_csv(os.path.join(ROOT, "results", "carotid_harmonics.tsv"), sep="\t")
fig = plt.figure(figsize=(6.85, 4.8)); gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.55)
ax = fig.add_subplot(gs[0, 0:2])
spec = {}
for (grp, lab, c) in (("young", "young adults (Ford 2005)", C["blue"]), ("older", "older adults (Hoi 2010)", C["red"])):
    d = WV[(WV.vessel == "ICA") & (WV.group == grp)]
    t = d.t_ms.values.astype(float); a = d.amp.values.astype(float); Tp = t[-1] - t[0]
    cs = CubicSpline(np.append(t[:-1], t[0] + Tp), np.append(a[:-1], a[0]), bc_type="periodic")
    tt = np.linspace(t[0], t[0] + Tp, 600); q = cs(tt); q /= q.mean()
    ax.plot(tt - t[0], q, color=c, label=lab); ax.scatter(t - t[0], a / cs(np.linspace(t[0], t[0] + Tp, 600, endpoint=False)).mean(), s=8, color=c)
    Fq = np.fft.rfft(cs(np.linspace(t[0], t[0] + Tp, 2048, endpoint=False))); Fq = 2 * np.abs(Fq) / 2048; spec[grp] = Fq[1:9] / (Fq[0] / 2)
ax.set(xlabel="Time in cardiac cycle (ms)", ylabel="Internal carotid flow / mean"); ax.legend()
ax.set_title("A  The carotid pulse grows with age", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[0, 2])
k = np.arange(1, 9)
ax.bar(k - 0.2, spec["young"], 0.38, color=C["blue"], label="young"); ax.bar(k + 0.2, spec["older"], 0.38, color=C["red"], label="older")
ax.set(xlabel="Harmonic", ylabel="Amplitude / mean flow"); ax.legend()
ax.set_title("B  Harmonic spectra", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[1, 0])
yv = H[(H.vessel == "ICA") & (H.group == "young")].sensed_factor_feaver.iloc[0]; ov = H[(H.vessel == "ICA") & (H.group == "older")].sensed_factor_feaver.iloc[0]
ratio_wss = yv / ov; rad_fixed = ratio_wss ** (-1 / (2 + B)); rad_meas = (1 / ratio_wss * 245 / 274) ** (1 / (2 + B))
ax.bar([0, 1], [100 * (ratio_wss - 1), 100 * (rad_meas - 1)], color=[C["grey"], C["green"]])
ax.errorbar([1], [100 * (rad_meas - 1) / 2 + 100 * (rad_fixed - 1) / 2], yerr=[[100 * (rad_meas - rad_fixed) / 2 * -1]], fmt="none", ecolor="k", lw=0.6) if False else None
ax.text(1, 100 * (rad_meas - 1) + 0.8, f"+{100 * (rad_meas - 1):.0f} to +{100 * (rad_fixed - 1):.0f}%", ha="center", fontsize=6)
ax.text(0, 100 * (ratio_wss - 1) - 2.5, f"{100 * (ratio_wss - 1):.0f}%", ha="center", fontsize=6)
ax.axhline(0, color="k", lw=0.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["mean shear", "lumen radius"])
ax.set(ylabel="Older vs young, predicted (%)", ylim=(-27, 12)); ax.set_title("C  Older pulse, same target", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[1, 1])
HRt = pd.read_csv(os.path.join(ROOT, "results", "hr_prediction.tsv"), sep="\t")
hr_eff = dict(zip(["aorta", "carotid", "femoral", "0.5 mm"], HRt.radius_change_pct_plus20bpm.tolist()))
ax.bar(range(4), list(hr_eff.values()), color=[C["purple"], C["purple"], C["purple"], C["grey"]])
ax.set_xticks(range(4)); ax.set_xticklabels(list(hr_eff.keys()), rotation=30, ha="right")
ax.set(ylabel="Radius change (%)\n+20 bpm, fixed flow"); ax.set_title("D  Heart rate by size", loc="left", fontweight="bold")
ax = fig.add_subplot(gs[1, 2])
Aa = np.linspace(0, 2.5, 101); tt_ = np.linspace(0, 2 * np.pi, 800, endpoint=False)
laws = {"mean + amplitude": 1 + Aa, "root-mean-square": np.sqrt(1 + Aa ** 2 / 2),
        "time-averaged magnitude": np.array([np.mean(np.abs(1 + e * np.cos(tt_))) for e in Aa])}
for (lab, f), c in zip(laws.items(), (C["blue"], C["green"], C["orange"])):
    ax.plot(Aa, 1 / f, color=c, label=lab)
ax.set(xlabel="Oscillatory / mean shear", ylabel="Apparent target\n(mean shear, vs steady)", ylim=(0, 1.1))
ax.legend(fontsize=5.5, loc="lower left"); ax.set_title("E  In vitro test", loc="left", fontweight="bold")
fig.savefig(os.path.join(OUT, "rsif_fig4.png"), bbox_inches="tight", dpi=300); fig.savefig(os.path.join(OUT, "rsif_fig4.tif"), bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(os.path.join(OUT, "rsif_fig4.pdf"), bbox_inches="tight")
print("fig4 done", flush=True)
