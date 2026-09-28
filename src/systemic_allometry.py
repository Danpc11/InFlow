#!/usr/bin/env python3
"""Systemic arterial tree at the set-point rest point, across mammals, compared with data.

Model (per body mass M). A symmetric bifurcating tree from the aorta: generation g carries Q/2^g, has
length L_g = L0 2^(-g/3); its radius is the rest point of the metabolic set-point rule (Vazquez-Victorio
et al., submitted to PRL),
    tau_sensed(r) = tau0 r^(b-1) L^((b-1)/2),
with sensed shear = Poiseuille mean shear x [(1 - phi) + phi sqrt(1 + (alpha/alpha_c)^2)], alpha the
Womersley number (r sqrt(w rho/mu)), phi the pulsatile fraction (optionally damped along the tree).
Generations are added until the radius falls below 3.65 um (the smallest vessel in the data).
One set-point constant tau0 for all species, calibrated once on the human aorta (Q = 88 cm3/s, r = 1.16 cm,
the point of the pooled curve at that flow). Across species: Q ~ M^(3/4), heart rate ~ M^(-1/4),
L0 ~ M^(1/3). Blood viscosity 0.04 P (as in the data), density 1.06 g/cm3.

Data (Seymour et al., J Exp Biol 2019; 92 points, 20 named arteries, 9 species, 23 g to 652 kg):
  pooled curve  log Q = -0.20 (log r)^2 + 1.91 log r + 1.82       (Q cm3/s, r cm; slope ~2 -> ~3)
  (the sign of the quadratic term follows from their WSS curve, log tau = -0.20 (log r)^2 - 1.09 log r + 0.53)
  WSS vs body mass: aorta -0.38 (Greve 2006; Weinberg & Ethier 2007), -0.44 (Seymour); common carotid -0.14
  (Seymour), -0.21 and -0.23 (Weinberg & Ethier; Cheng 2007); internal carotid -0.20; vertebral -0.22;
  femoral -0.49 (resting data; sized for exercise flow, Seymour).  Aortic radius ~ M^0.36 (Holt 1981).
Named arteries are placed at the generation carrying the same fraction of cardiac output as in humans
(abdominal aorta 0.30, common carotid 0.082, internal carotid 0.05, femoral 0.041, vertebral 0.017;
from Seymour's allometric intercepts at 70 kg and standard physiology).

    python3 src/systemic_allometry.py --out results
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU, RHO = 4e-3, 1060.0                    # Pa s (0.04 P), kg/m3
Q_H, R_H, M_H, HR_H, L0_H = 88e-6, 0.0116, 70.0, 70.0, 0.30
SPECIES = {"mouse": 0.023, "rat": 0.3, "rabbit": 3.0, "cat": 4.0, "macaque": 5.0, "dog": 20.0,
           "human": 70.0, "horse": 500.0, "cow": 652.0}
ARTERIES = {"aorta": 0.30, "common carotid": 0.082, "internal carotid": 0.05, "femoral": 0.041, "vertebral": 0.017}
DATA_WSS = {"aorta": [-0.38, -0.44], "common carotid": [-0.14, -0.21, -0.23], "internal carotid": [-0.20],
            "vertebral": [-0.22], "femoral": [-0.49]}
R_MIN = 3.65e-6
VISC = "constant"          # "constant" | "vitro" (Pries 1992) | "vivo" (endothelial surface layer, Pries & Secomb 2005 form)


def mu_rel_vitro(D):
    """Relative apparent viscosity of blood at discharge hematocrit 0.45 in glass tubes (Pries et al. 1992);
    D in um. Tends to 3.2 in large tubes, minimum ~1.3 near 7 um (Fahraeus-Lindqvist effect)."""
    return 220 * np.exp(-1.3 * D) + 3.2 - 2.44 * np.exp(-0.06 * D ** 0.645)


def mu_rel_vivo(D):
    """In vivo apparent viscosity: an endothelial surface layer of thickness W reduces the hydrodynamic
    diameter, D_ph = D - 2W, and mu = mu_vitro(D_ph) (D/D_ph)^4. W = W_max (D - D_off)/(D + D_crit - 2 D_off) for
    D > D_off, with W_max = 2.6, D_off = 2.4, D_crit = 10.5 um (hematocrit-independent part of the Pries & Secomb
    2005 description; approximate)."""
    W = np.where(D > 2.4, 2.6 * (D - 2.4) / (D + 10.5 - 4.8), 0.0)
    Dph = np.maximum(D - 2 * W, 1.0)
    return mu_rel_vitro(Dph) * (D / Dph) ** 4


def mu_of(r):
    if VISC == "constant":
        return MU
    D = 2e6 * np.asarray(r, float)
    f = mu_rel_vitro if VISC == "vitro" else mu_rel_vivo
    return MU * f(D) / f(2e4)          # 4 mPa s in large arteries (Seymour's 0.04 P)


def sensed_factor(r, hr, phi):
    a = r * np.sqrt(2 * np.pi * hr / 60 * RHO / mu_of(r))
    return (1 - phi) + phi * np.sqrt(1 + (a / 2.83) ** 2)


def radius(Q, L, hr, b, phi, tau0):
    f = lambda x: np.log(4 * mu_of(np.exp(x)) * Q / (np.pi * np.exp(3 * x)) * sensed_factor(np.exp(x), hr, phi)) - \
        np.log(tau0 * np.exp(x) ** (b - 1) * L ** ((b - 1) / 2))
    return float(np.exp(brentq(f, np.log(1e-8), np.log(1.0))))


def tree(M, b, phi0, damp, tau0, g_step=0.25):
    Q0 = Q_H * (M / M_H) ** 0.75; hr = HR_H * (M / M_H) ** -0.25; L0 = L0_H * (M / M_H) ** (1 / 3)
    out, g = [], 0.0
    while True:
        phi = phi0 * (np.exp(-g / damp) if damp > 0 else 1.0)
        Q = Q0 / 2 ** g; L = L0 * 2 ** (-g / 3)
        r = radius(Q, L, hr, b, phi, tau0)
        out.append((g, r, Q, L, 4 * MU * Q / (np.pi * r ** 3)))   # WSS as in the data: constant 0.04 P
        if r < R_MIN or g > 60: break
        g += g_step
    return pd.DataFrame(out, columns=["g", "r", "Q", "L", "tau_mean"])


def calibrate(b, phi0, damp):
    hr, L0 = HR_H, L0_H
    fac = sensed_factor(R_H, hr, phi0)
    return 4 * MU * Q_H / (np.pi * R_H ** 3) * fac / (R_H ** (b - 1) * L0 ** ((b - 1) / 2))


def evaluate(b, phi0, damp, visc="constant"):
    global VISC
    VISC = visc
    tau0 = calibrate(b, phi0, damp)
    pts, named = [], []
    for sp, M in SPECIES.items():
        t = tree(M, b, phi0, damp, tau0)
        t["species"], t["M"] = sp, M
        pts.append(t)
        for art, frac in ARTERIES.items():
            g = np.log2(1 / frac)
            r = np.exp(np.interp(g, t.g, np.log(t.r))); Q = Q_H * (M / M_H) ** 0.75 * frac
            named.append(dict(species=sp, M=M, artery=art, r=r, tau=4 * MU * Q / (np.pi * r ** 3)))
    P = pd.concat(pts); N = pd.DataFrame(named)
    lr = np.log10(P.r * 100); lq = np.log10(P.Q * 1e6)
    pred = -0.20 * lr ** 2 + 1.91 * lr + 1.82
    rms = float(np.sqrt(np.mean((lq - pred) ** 2)))
    ex = {}
    for art, d in N.groupby("artery"):
        lm = np.log(d.M.values)
        ex[art] = dict(radius=np.polyfit(lm, np.log(d.r.values), 1)[0], wss=np.polyfit(lm, np.log(d.tau.values), 1)[0])
    return P, N, rms, ex


def score(rms, ex):
    """Mismatch: pooled Q(r) curve + WSS exponents of the resting-sized arteries (femoral excluded,
    see docstring) + aortic radius exponent (Holt 0.36)."""
    e = [(ex[a]["wss"] - np.mean(DATA_WSS[a])) ** 2 for a in ("aorta", "common carotid", "internal carotid", "vertebral")]
    e.append((ex["aorta"]["radius"] - 0.36) ** 2)
    return float(np.sqrt(np.mean(e))), rms


def main():
    global VISC
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    cases = {"b=3/4, pulsatile": (0.75, 0.8, 0), "b=3/4, pulsatile, viscosity in vitro": (0.75, 0.8, 0, "vitro"),
             "b=3/4, pulsatile, viscosity in vivo": (0.75, 0.8, 0, "vivo"), "b=0.775, pulsatile, viscosity in vivo": (0.775, 0.8, 0, "vivo"), "b=3/4, pulsatile damped (6 gen)": (0.75, 0.8, 6),
             "b=3/4, Poiseuille": (0.75, 0.0, 0), "b=1, pulsatile": (1.0, 0.8, 0), "b=1, Poiseuille (Murray)": (1.0, 0.0, 0),
             "b=2/3, pulsatile": (2 / 3, 0.8, 0)}
    rows, keep = [], {}
    for name, spec in cases.items():
        b, phi, damp = spec[:3]; visc = spec[3] if len(spec) > 3 else "constant"
        P, N, rms, ex = evaluate(b, phi, damp, visc)
        tr = []
        for M in SPECIES.values():
            tr.append(P[P.M == M].r.iloc[0])
        lmM = np.log(list(SPECIES.values()))
        trunk_r = np.polyfit(lmM, np.log(tr), 1)[0]
        d = P[P.species == "human"]; lr_, lq_ = np.log10(d.r * 100).values, np.log10(d.Q * 1e6).values
        nloc = np.diff(lq_) / np.diff(lr_); mid = 0.5 * (lr_[1:] + lr_[:-1])
        n_small = float(np.interp(-2.5, mid[::-1], nloc[::-1])); n_large = float(np.interp(-0.2, mid[::-1], nloc[::-1]))
        s, _ = score(rms, ex)
        keep[name] = (P, N, ex)
        r = dict(case=name, b=b, phi=phi, damp=damp, viscosity=visc, rms_logQ_pooled=rms, exponent_mismatch=s,
                 trunk_radius=trunk_r, trunk_wss=0.75 - 3 * trunk_r, n_at_r_30um=n_small, n_at_r_6mm=n_large)
        for art in ARTERIES: r[f"wss_{art}"] = ex[art]["wss"]
        rows.append(r)
    T = pd.DataFrame(rows)
    T.round(3).to_csv(os.path.join(a.out, "systemic_comparison.tsv"), sep="\t", index=False)
    print(T.round(3).to_string(index=False))

    # profile over b (pulsatile, undamped and damped)
    prof = []
    for damp, visc in ((0, "constant"), (6, "constant"), (0, "vivo")):
        for b in np.round(np.arange(0.55, 1.101, 0.025), 3):
            P, N, rms, ex = evaluate(b, 0.8, damp, visc)
            s, _ = score(rms, ex)
            prof.append(dict(damp=damp, viscosity=visc, b=b, exponent_mismatch=s, rms_logQ_pooled=rms, aorta_radius=ex["aorta"]["radius"],
                             wss_aorta=ex["aorta"]["wss"], wss_carotid=ex["common carotid"]["wss"]))
    PR = pd.DataFrame(prof); PR.round(4).to_csv(os.path.join(a.out, "systemic_b_profile.tsv"), sep="\t", index=False)
    for (damp, visc), d in PR.groupby(["damp", "viscosity"]):
        i1 = d.exponent_mismatch.idxmin(); i2 = d.rms_logQ_pooled.idxmin()
        print(f"damp={damp} viscosity={visc}: best b by exponents {d.loc[i1,'b']:.3f} (mismatch {d.loc[i1,'exponent_mismatch']:.3f}); "
              f"best b by pooled Q(r) curve {d.loc[i2,'b']:.3f} (rms {d.loc[i2,'rms_logQ_pooled']:.3f})")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    P, N, ex = keep["b=3/4, pulsatile"]
    cm = plt.cm.viridis
    for i, (sp, d) in enumerate(P.groupby("species", sort=False)):
        ax[0].plot(np.log10(d.r * 100), np.log10(d.Q * 1e6), ".", ms=3, color=cm(np.log(SPECIES[sp] / 0.02) / np.log(700 / 0.02)), label=sp)
    x = np.linspace(-3.5, 0.2, 100)
    ax[0].plot(x, -0.20 * x ** 2 + 1.91 * x + 1.82, "k-", lw=2, label="Seymour 2019 (92 datos)")
    PB, NB, _ = keep["b=1, Poiseuille (Murray)"]
    ax[0].set(xlabel="log radio (cm)", ylabel="log flujo (cm³/s)", title="flujo contra radio: b=3/4 pulsátil")
    ax[0].legend(fontsize=6, ncol=2)
    for name, st in (("b=3/4, pulsatile", "-"), ("b=1, Poiseuille (Murray)", "--"), ("b=3/4, Poiseuille", ":")):
        Pn = keep[name][0]; d = Pn[Pn.species == "human"]
        lr, lq = np.log10(d.r * 100).values, np.log10(d.Q * 1e6).values
        ax[1].plot(lr[1:], np.diff(lq) / np.diff(lr), st, label=name)
    ax[1].plot(x, -0.40 * x + 1.91, "k-", lw=2, label="Seymour (derivada)")
    ax[1].set(xlabel="log radio (cm)", ylabel="exponente local n (Q ∝ rⁿ)", title="da Vinci (2) → Murray (3)", ylim=(1.5, 3.3))
    ax[1].legend(fontsize=7)
    arts = ["aorta", "common carotid", "internal carotid", "vertebral", "femoral"]
    xx = np.arange(len(arts))
    for k, (name, mk) in enumerate((("b=3/4, pulsatile", "o"), ("b=3/4, Poiseuille", "s"), ("b=1, Poiseuille (Murray)", "^"))):
        ax[2].plot(xx + 0.12 * (k - 1), [keep[name][2][a_]["wss"] for a_ in arts], mk, label=name)
    for i, a_ in enumerate(arts):
        ax[2].plot([i] * len(DATA_WSS[a_]), DATA_WSS[a_], "kx", ms=8, label="datos" if i == 0 else None)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xticks(xx); ax[2].set_xticklabels(["aorta", "carótida\ncomún", "carótida\ninterna", "vertebral", "femoral*"], fontsize=8)
    ax[2].set(ylabel="exponente del cortante medio con M", title="cortante por arteria entre especies"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", "fig17_systemic_seymour.png"), dpi=150)
    print("saved")


if __name__ == "__main__":
    main()
