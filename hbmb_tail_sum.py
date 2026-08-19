#!/usr/bin/env python3
"""Exact representative HBMB tail identity used in Section 2."""
from __future__ import annotations

import math


def kernel(l: int) -> float:
    return (2*l + 1) / (l*l*(l+1)*(l+1))


def exact_tail(L: int) -> float:
    return 1.0 / (L + 1)**2


def partial_tail(L: int, lmax: int = 2_000_000) -> float:
    return sum(kernel(l) for l in range(L + 1, lmax + 1))


def main() -> None:
    print('HBMB exact representative tail identity')
    print('='*72)
    print('K_l=(2l+1)/[l^2(l+1)^2]=1/l^2-1/(l+1)^2')
    print('Sum_{l=L+1}^infinity K_l = 1/(L+1)^2')
    print('\nNumerical partial-sum checks:')
    for L in (1, 2, 5, 10, 50):
        # The finite-lmax remainder is exactly 1/(lmax+1)^2.
        lmax = 200_000
        numeric = partial_tail(L, lmax)
        corrected = numeric + 1.0/(lmax+1)**2
        exact = exact_tail(L)
        err = abs(corrected-exact)
        print(f'  L={L:2d}: exact={exact:.12e}, corrected sum={corrected:.12e}, |err|={err:.3e}')
        if err > 5e-14:
            raise RuntimeError(f'tail identity check failed for L={L}')
    print('\nChecks passed.')


if __name__ == '__main__':
    main()
