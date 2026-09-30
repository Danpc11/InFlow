#!/usr/bin/env python3
"""Electronic supplementary material: figures S1-S2 and tables S2-S6 (markdown), built from the data and from the outputs
of the analysis scripts. Run after run_all.sh.

    python3 src/make_esm.py --out results
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sigma_exact_lib import G_tube

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(a.out, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 6.5, "savefig.dpi": 300})
BLUE, RED, GREEN, GREY = "#1F5FA8", "#B2182B", "#2E8B57", "#7F7F7F"

# Figure S1: exact Womersley ratio
al = np.logspace(-1, np.log10(40), 300)
fig, ax = plt.subplots(figsize=(3.35, 2.6))
ax.loglog(al, G_tube(al), color=BLUE, label="exact solution, rigid tube")
ax.loglog(al, np.ones_like(al), ":", color=GREY, label="quasi-steady limit (G = 1)")
ax.loglog(al[al > 5], al[al > 5] / (2 * np.sqrt(2)) * (G_tube(np.array([40.0]))[0] / (40 / (2 * np.sqrt(2)))), "--", color=RED, lw=0.8,
          label="linear growth at large \u03b1")
for x, y, lab in ((0.9, 1.08, "0.5 mm artery"), (4.6, 1.75, "carotid"), (15.0, 1.08, "aorta")):
    ax.axvline(x, color=GREY, lw=0.4, ls="--", zorder=0); ax.text(x * 1.06, y, lab, fontsize=5.6, color=GREY, rotation=90, va="bottom")
ax.set(xlabel="Womersley number \u03b1", ylabel="G(\u03b1)", ylim=(0.8, 12))
ax.legend(loc="upper left", fontsize=6, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
fig.savefig(os.path.join(FIG, "esm_figS1.png"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "esm_figS1.pdf"), bbox_inches="tight")

# Figure S2: variation of sensed shear across four human arteries versus w
T3 = pd.read_csv(os.path.join(ROOT, "data", "reneman2009_table3_human_beds.tsv"), sep="\t")
ws = np.linspace(0, 1.2, 121)
cv = lambda v: 100 * np.std(v) / np.mean(v)
cvy = [cv(T3.mean_WSS_young_Pa + w * (T3.peak_WSS_young_Pa - T3.mean_WSS_young_Pa)) for w in ws]
cvo = [cv(T3.mean_WSS_old_Pa + w * (T3.peak_WSS_old_Pa - T3.mean_WSS_old_Pa)) for w in ws]
AW = pd.read_csv(os.path.join(a.out, "amplitude_weight_summary.tsv"), sep="\t")
fig, ax = plt.subplots(figsize=(3.35, 2.6))
ax.plot(ws, cvy, color=BLUE, label="young adults")
ax.plot(ws, cvo, "--", color=RED, label="older adults")
ax.axvspan(0.32, 0.43, color=GREEN, alpha=0.15, lw=0); ax.text(0.46, 56, "in vitro weight, converted", fontsize=5.8, color=GREEN, va="top")
ax.axvline(0.4, color="k", lw=0.5); ax.text(0.46, 49, "w = 0.4, used in the main text", fontsize=5.8, va="top")
ax.set(xlabel="Weight of the pulsatile component, w", ylabel="Variation across four arteries (CV, %)", ylim=(0, 60))
ax.legend(loc="center right")
fig.savefig(os.path.join(FIG, "esm_figS2.png"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "esm_figS2.pdf"), bbox_inches="tight")


# Tables (markdown)
def md(df):
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out


H = pd.read_csv(os.path.join(a.out, "carotid_harmonics.tsv"), sep="\t")
t2 = H[["source", "vessel", "group", "HR", "alpha1", "flow_A1", "wss_A1", "wss_peak_minus_mean", "c_wss", "sensed_factor_feaver"]].copy()
t2.columns = ["Source", "Artery", "Group", "Heart rate (bpm)", "\u03b1\u2081", "A\u2081 of flow / mean", "A\u2081 of wall shear / mean",
              "Peak excursion / mean", "c = A\u2081/\u0394\u03c4", "S/\u03c4_mean (w\u2081 = 2.9)"]
C = pd.read_csv(os.path.join(a.out, "canonical_results.tsv"), sep="\t")
t3 = C[["w", "b", "b_CI95_by_study", "rms_log_sigma", "variance_removed_pct", "median_sigma_low_high", "n_coronary_r075_2mm",
        "carotid_over_femoral_WSS", "setpoint_mouse_vs_human", "setpoint_rabbit_vs_human"]].copy()
t3.columns = ["w", "b", "95% CI (studies)", "rms log\u2081\u2080 \u03a3", "Variance removed (%)", "Median \u03a3, low/high beds",
              "Coronary exponent", "Carotid/femoral", "Mouse/human target", "Rabbit/human target"]
Mc = pd.read_csv(os.path.join(a.out, "metabolic_curvature_b.tsv"), sep="\t")
t4 = Mc[["metabolic_scenario", "b_all", "CI_all", "b_small_lt10kg", "CI_small", "b_large_ge10kg", "CI_large"]].copy()
t4.columns = ["Cardiac-output law", "b (all)", "95% CI", "b (< 10 kg)", "95% CI ", "b (\u2265 10 kg)", "95% CI  "]
F = pd.read_csv(os.path.join(a.out, "feaver_sensing_laws.tsv"), sep="\t")
t5 = F[["law", "R2", "k", "AIC", "params"]].copy(); t5.columns = ["Sensing law", "R\u00b2", "Parameters", "AIC", "Fitted values"]
Rb = pd.read_csv(os.path.join(a.out, "rabbit_mouse_check.tsv"), sep="\t")
Hr = pd.read_csv(os.path.join(a.out, "hr_prediction.tsv"), sep="\t")
t6 = Hr.copy(); t6.columns = ["Vessel", "Radius (mm)", "d ln r / d ln HR", "Radius change, +20 bpm (%)", "Mean WSS change, +20 bpm (%)"]
t7 = Rb.copy(); t7.columns = ["b", "Species", "Radius (mm)", "Model mean WSS (Pa)", "Target ratio vs human carotid"]
with open(os.path.join(a.out, "esm_tables.md"), "w") as f:
    for name, t in (("S2", t2), ("S3", t3), ("S4", t4), ("S5", t5), ("S6", t6), ("S7", t7)):
        f.write(f"TABLE {name}\n" + md(t) + "\n")
print("ESM figures and tables written")
