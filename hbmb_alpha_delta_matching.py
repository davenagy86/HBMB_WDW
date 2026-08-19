#!/usr/bin/env python3
"""Finite-L matching and plateau coefficients for the HBMB QR revision.

Numerical precision note
------------------------
At the minimal N_*=55 pivot, L_* = exp(55 + Delta_match^(G)) ~= 1.78e24.
Direct evaluation of the Barnes-G representation there subtracts terms of
order L_*^2 log L_* (~1e50) to recover a residual of order 1e1. This loses
roughly 49-50 decimal digits. Standard floating-point precision is therefore
unsafe. For CMB-pivot evaluation use the asymptotic residual running of the
manuscript's Eq. (80), as done in hbmb_plateau_benchmark.py, or use at least
~60 decimal digits for a direct Barnes-G cross-check. Endpoint matching near
L ~ 2.3 is not affected by this large-L cancellation.
"""
from __future__ import annotations
import mpmath as mp

mp.mp.dps = 60
B_1SP = mp.mpf(2)/3
ZETA_PRIME_MINUS_ONE = mp.diff(lambda s: mp.zeta(s), -1)
C_A = 2 + 4*(mp.mpf(1)/12 - ZETA_PRIME_MINUS_ONE)


def S_barnes(L):
    L = mp.mpf(L)
    return 4*(L*mp.log(mp.gamma(L+1))-mp.log(mp.barnesg(L+1))) + (2*L+1)*mp.log(L+1)


def bulk_interface(L):
    L = mp.mpf(L)
    return 2*L**2*mp.log(L)-L**2+4*L*mp.log(L)


def g_1sp(L):
    return mp.mpf('0.5')*(S_barnes(L)-bulk_interface(L)-C_A)


def D_1sp(L):
    y = mp.log(L)
    return mp.diff(lambda yy: g_1sp(mp.e**yy), y)


def epsilon_full(N, delta, cchi=1, cL=1, gR=0, L0=1):
    N, delta = mp.mpf(N), mp.mpf(delta)
    cchi, cL, gR, L0 = map(mp.mpf, (cchi,cL,gR,L0))
    L = L0*mp.e**(cL*(N+delta))
    g = gR + g_1sp(L)
    return cchi*cL/2 * D_1sp(L)/g**2


def solve_delta(cchi=1, cL=1, guess=0.84):
    f = lambda d: epsilon_full(0,d,cchi,cL)-1
    return mp.findroot(f, guess)


def main():
    cchi = mp.mpf(1); cL = mp.mpf(1)
    alpha = cchi/(2*B_1SP*cL)
    delta_asy = mp.sqrt(alpha)
    delta_G = solve_delta(cchi,cL)
    L_end = mp.e**(cL*delta_G)
    eps_plateau_at_end = alpha/delta_G**2

    print('HBMB alpha/Delta matching')
    print('='*72)
    print(f'b_1sp                 = {mp.nstr(B_1SP, 16)}')
    print(f'c_chi                  = {mp.nstr(cchi, 16)}')
    print(f'c_L                    = {mp.nstr(cL, 16)}')
    print(f'alpha_eff              = {mp.nstr(alpha, 16)}')
    print(f'Delta_asy              = {mp.nstr(delta_asy, 16)}')
    print(f'Delta_match^(G)        = {mp.nstr(delta_G, 16)}')
    print(f'L_end                  = {mp.nstr(L_end, 16)}')
    print(f'epsilon_full(0)        = {mp.nstr(epsilon_full(0,delta_G), 16)}')
    print(f'epsilon_plateau(0)     = {mp.nstr(eps_plateau_at_end, 16)}')
    print('\nFinite matching depends separately on c_chi and c_L:')
    for cchi_i,cL_i in ((0.5,1),(1,1),(2,1),(1,0.5),(4,2)):
        alpha_i = mp.mpf(cchi_i)/(2*B_1SP*mp.mpf(cL_i))
        try:
            d = solve_delta(cchi_i,cL_i,guess=0.8)
            print(f'  cchi={cchi_i:3.1f}, cL={cL_i:3.1f}, alpha={float(alpha_i):.6f}, Delta_G={float(d):.9f}')
        except Exception as exc:
            print(f'  cchi={cchi_i}, cL={cL_i}: no root from chosen guess ({exc})')


if __name__ == '__main__':
    main()
