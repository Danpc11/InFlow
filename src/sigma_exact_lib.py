"""Shared helpers for pulsatile wall shear in a rigid tube (Womersley 1955).

T_tube(alpha)  complex transfer function between the k-th harmonic of flow and the k-th harmonic of wall shear,
               relative to the quasi-steady value 4 mu Q_hat / (pi r^3):
                   tau_hat_k = (4 mu / (pi r^3)) * Q_hat_k * T(alpha sqrt(k)),
                   T(alpha) = -Lambda J1(Lambda) / [J0(Lambda) - 2 J1(Lambda)/Lambda] / 4,   Lambda = i^(3/2) alpha.
               T -> 1 as alpha -> 0 (phase 0); |T| grows ~linearly and the phase lead tends to 45 deg at large alpha.
               The minus sign is required for T(0) = +1; the published expression (ESM eq. S6) omitted it, which does
               not affect |T| but would invert every phase.
G_tube(alpha)  magnitude |T(alpha)| (kept for backward compatibility).
alpha_of(r,hr) Womersley number alpha = r sqrt(omega rho / mu), omega = 2 pi HR / 60 (HR in beats per minute).

Above alpha = 40 the Bessel functions of large complex argument overflow; |T| is continued linearly from its exact
values at 39 and 40 and the phase is held at its asymptote, 45 deg.
"""
import numpy as np
from scipy.special import jv
import systemic_allometry as SA

_A_MAX = 40.0


def _T_exact(alpha):
    L = (1j) ** 1.5 * alpha
    return -L * jv(1, L) / (jv(0, L) - 2 * jv(1, L) / L) / 4.0


def T_tube(alpha):
    alpha = np.atleast_1d(np.asarray(alpha, float))
    out = np.ones(alpha.shape, dtype=complex)
    m = (alpha > 1e-3) & (alpha <= _A_MAX)
    if m.any():
        out[m] = _T_exact(alpha[m])
    big = alpha > _A_MAX
    if big.any():
        g40, g39 = abs(_T_exact(_A_MAX)), abs(_T_exact(_A_MAX - 1))
        out[big] = (g40 + (alpha[big] - _A_MAX) * (g40 - g39)) * np.exp(1j * np.pi / 4)
    return out


def G_tube(alpha):
    return np.abs(T_tube(alpha))


def alpha_of(r, hr):
    return r * np.sqrt(2 * np.pi * hr / 60 * SA.RHO / SA.MU)
