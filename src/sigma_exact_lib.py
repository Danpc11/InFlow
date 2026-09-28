import numpy as np
from scipy.special import jv
import systemic_allometry as SA


def G_tube(alpha):
    alpha = np.atleast_1d(np.asarray(alpha, float)); x = np.minimum(alpha, 40.0); out = np.ones_like(x)
    m = x > 1e-3; L = (1j) ** 1.5 * x[m]
    out[m] = np.abs(L * jv(1, L) / jv(0, L) / (1 - 2 * jv(1, L) / (L * jv(0, L)))) / 4.0
    big = alpha > 40
    if big.any():
        g40 = out[big][0] if False else G_tube(np.array([40.0]))[0]; g39 = G_tube(np.array([39.0]))[0]
        out[big] = g40 + (alpha[big] - 40) * (g40 - g39)
    return out


def alpha_of(r, hr): return r * np.sqrt(2 * np.pi * hr / 60 * SA.RHO / SA.MU)
