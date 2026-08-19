#!/usr/bin/env python3
"""HBMB single-species omitted-sector determinant checks for the QR revision.

Checks:
  * exact integer partial sum
  * standard Barnes-G continuation of that integer identity
  * corrected large-L expansion including L^-3 and L^-4 terms
  * convergence of the positive residual running g_1sp(L)

Convention:
  Gamma_Q^ren(L) = Gamma_0 - g_1sp(L)
with
  g_1sp(L) = (1/2) [S_G(L) - B(L) - C_A]
  B(L) = 2 L^2 ln L - L^2 + 4 L ln L.
"""
from __future__ import annotations
import mpmath as mp

mp.mp.dps = 60

B = mp.mpf(2) / 3
C2 = mp.mpf(31) / 360
C3 = -mp.mpf(1) / 12
C4 = mp.mpf(47) / 630

ZETA_PRIME_MINUS_ONE = mp.diff(lambda s: mp.zeta(s), -1)
LN_A_G = mp.mpf(1) / 12 - ZETA_PRIME_MINUS_ONE
C_A = 2 + 4 * LN_A_G


def S_integer(L: int) -> mp.mpf:
    return mp.fsum((2*l + 1) * mp.log(l*(l+1)) for l in range(1, L+1))


def S_barnes(L: mp.mpf) -> mp.mpf:
    L = mp.mpf(L)
    return (4 * (L * mp.log(mp.gamma(L+1)) - mp.log(mp.barnesg(L+1)))
            + (2*L + 1) * mp.log(L+1))


def bulk_interface(L: mp.mpf) -> mp.mpf:
    L = mp.mpf(L)
    return 2*L**2*mp.log(L) - L**2 + 4*L*mp.log(L)


def g_barnes(L: mp.mpf) -> mp.mpf:
    return mp.mpf('0.5') * (S_barnes(L) - bulk_interface(L) - C_A)


def g_asym(L: mp.mpf, order: int = 4) -> mp.mpf:
    L = mp.mpf(L)
    out = B * mp.log(L)
    if order >= 2:
        out += C2/L**2
    if order >= 3:
        out += C3/L**3
    if order >= 4:
        out += C4/L**4
    return out


def main() -> None:
    print('HBMB determinant-running check')
    print('='*72)
    print(f'C_A = {mp.nstr(C_A, 20)}')
    print('\nExact integer identity vs Barnes-G continuation:')
    for L in (1,2,3,5,10,20):
        err = abs(S_integer(L)-S_barnes(L))
        print(f'  L={L:2d}: |S_int-S_G| = {mp.nstr(err, 6)}')

    print('\nResidual-running relative errors (%)')
    print('      L          L^-2          L^-3          L^-4')
    for L in (2,3,5,10):
        ge = g_barnes(L)
        vals = [100*abs(g_asym(L,o)-ge)/abs(ge) for o in (2,3,4)]
        print(f'  {L:5d}  ' + '  '.join(f'{float(v):12.6g}' for v in vals))

    print('\nCorrected asymptotic coefficients:')
    print('  S(L):   +31/(180 L^2) -1/(6 L^3) +47/(315 L^4) + ...')
    print('  Gamma:  -31/(360 L^2) +1/(12 L^3) -47/(630 L^4) + ...')
    print('  g_1sp:  +31/(360 L^2) -1/(12 L^3) +47/(630 L^4) + ...')


if __name__ == '__main__':
    main()
