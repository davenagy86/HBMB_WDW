#!/usr/bin/env python3
"""
hbmb_overlap_scaling.py

Numerical checks for the Section 2 strip/channel construction.

Checks:
  1. The exact DFT tangential basis has N_t=2l+1 and its uniform packet
     combination equals the m=0 harmonic.
  2. A normalized centered Gaussian strip of width delta_l~1/(l+1/2)
     obeys |V_l|~l^{-1/2} on the surviving even-l subsequence and has
     vanishing odd-l overlap by equatorial parity.
  3. A slightly displaced strip restores both parities while preserving the
     same asymptotic envelope.
  4. Bounded nonuniform channel weights preserve sigma_l^2~l^{-1} and hence
     K_l~l^{-3} for lambda_l=l(l+1).

This script is an auxiliary numerical robustness check. It does not attempt
to derive a full microscopic Einstein-matter interaction operator.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

_NQUAD = 600
_X, _W = leggauss(_NQUAD)
_THETA = np.arccos(_X)
_Y00 = np.full_like(_THETA, 1.0 / np.sqrt(4.0 * np.pi))

def Yl0(l: int):
    return np.sqrt((2*l+1)/(4.0*np.pi)) * eval_legendre(l, _X)


def tangential_dft_diagnostics(l: int = 7):
    n = 2 * l + 1
    ms = np.arange(-l, l + 1)
    js = np.arange(n)
    U = np.exp(-2j * np.pi * np.outer(js, ms) / n) / np.sqrt(n)
    unitary_error = np.linalg.norm(U @ U.conj().T - np.eye(n), ord=np.inf)
    uniform_coeff = U.sum(axis=0) / np.sqrt(n)
    target = np.zeros(n, dtype=complex)
    target[np.where(ms == 0)[0][0]] = 1.0
    uniform_error = np.linalg.norm(uniform_coeff - target)
    return unitary_error, uniform_error


def strip_overlap(l: int, c_width: float = 1.0, offset_widths: float = 0.0) -> float:
    """Normalized Gaussian-strip overlap with Y_00 and Y_l0."""
    delta = c_width / (l + 0.5)
    center = np.pi / 2 + offset_widths * delta
    profile = np.exp(-0.5 * ((_THETA - center) / delta) ** 2)
    norm2 = 2 * np.pi * np.sum(_W * profile**2)
    b = profile / np.sqrt(norm2)
    return float(2 * np.pi * np.sum(_W * _Y00 * b * Yl0(l)))


def power_fit(ls, vals):
    ls = np.asarray(ls, dtype=float)
    vals = np.asarray(vals, dtype=float)
    mask = vals > 1e-14
    p, logA = np.polyfit(np.log(ls[mask]), np.log(vals[mask]), 1)
    return p, np.exp(logA)


def centered_and_displaced_fits(lmin: int = 20, lmax: int = 90):
    even_ls = np.arange(lmin + (lmin % 2), lmax + 1, 2, dtype=int)
    even_vals = np.array([abs(strip_overlap(int(l), offset_widths=0.0)) for l in even_ls])
    p_center, A_center = power_fit(even_ls, even_vals)

    odd_ls = np.arange(max(21, lmin | 1), lmax + 1, 2, dtype=int)
    odd_vals = np.array([abs(strip_overlap(int(l), offset_widths=0.0)) for l in odd_ls])

    all_ls = np.arange(lmin, lmax + 1, dtype=int)
    displaced_vals = np.array([abs(strip_overlap(int(l), offset_widths=0.35)) for l in all_ls])
    p_disp, A_disp = power_fit(all_ls, displaced_vals)

    return {
        "center_power": p_center,
        "center_prefactor": A_center,
        "max_center_odd": float(odd_vals.max()),
        "displaced_power": p_disp,
        "displaced_prefactor": A_disp,
    }


def nonuniform_channel_fits(lmin: int = 20, lmax: int = 90):
    """
    Construct bounded deterministic nonuniform weights.

    bar_g_l is fixed from the centered/displaced strip envelope as
        bar_g_l = |V_l|/sqrt(N_t).
    We use the displaced strip so both parities are present. The perturbation
    delta_lj has zero mean and O(1) mean-square norm independent of l.
    """
    ls = np.arange(lmin, lmax + 1, dtype=int)
    sigma2 = []
    kernels = []
    mean_square_factors = []

    for l in ls:
        Nt = 2 * l + 1
        V = abs(strip_overlap(int(l), offset_widths=0.35))
        bar_g = V / np.sqrt(Nt)
        j = np.arange(Nt, dtype=float)
        delta = 0.25 * np.cos(2 * np.pi * j / Nt) + 0.10 * np.sin(4 * np.pi * j / Nt)
        delta -= delta.mean()
        weights = 1.0 + delta
        g = bar_g * weights
        s2 = np.sum(np.abs(g)**2)
        lam = l * (l + 1)
        sigma2.append(s2)
        kernels.append(s2 / lam)
        mean_square_factors.append(np.mean(np.abs(weights)**2))

    p_sigma2, _ = power_fit(ls, sigma2)
    p_kernel, _ = power_fit(ls, kernels)
    return {
        "sigma2_power": p_sigma2,
        "kernel_power": p_kernel,
        "ms_min": float(np.min(mean_square_factors)),
        "ms_max": float(np.max(mean_square_factors)),
    }


def main():
    print("HBMB strip/channel diagnostics")
    print("=" * 64)

    ue, u0e = tangential_dft_diagnostics(7)
    print("\n1) Tangential DFT basis")
    print(f"   unitarity error             = {ue:.3e}")
    print(f"   uniform packet -> m=0 error = {u0e:.3e}")

    fits = centered_and_displaced_fits()
    print("\n2) Strip overlap and parity")
    print(f"   centered even-l power       = {fits['center_power']:.6f}")
    print(f"   max centered odd overlap    = {fits['max_center_odd']:.3e}")
    print(f"   displaced all-l power       = {fits['displaced_power']:.6f}")

    nf = nonuniform_channel_fits()
    print("\n3) Bounded nonuniform channel weights")
    print(f"   mean-square factor range    = [{nf['ms_min']:.6f}, {nf['ms_max']:.6f}]")
    print(f"   sigma_l^2 power             = {nf['sigma2_power']:.6f}")
    print(f"   K_l power                   = {nf['kernel_power']:.6f}")

    print("\nExpected asymptotic targets: -1/2 for |V_l|, -1 for sigma_l^2, -3 for K_l.")


if __name__ == "__main__":
    main()
