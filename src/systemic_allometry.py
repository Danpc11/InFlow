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
