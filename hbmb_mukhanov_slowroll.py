#!/usr/bin/env python3
"""Leading slow-roll scalar/tensor benchmark for the HBMB manuscript."""
from __future__ import annotations

import math

ALPHA_EFF = 0.75
DELTA_G = 0.8366497586
NSTAR = 55.0
AS = 2.1e-9


def hubble_flow(N: float):
    u = N + DELTA_G
    eps1 = ALPHA_EFF/u**2
    eps2 = 2.0/u
    eps3 = 1.0/u
    return eps1, eps2, eps3


def observables(N: float = NSTAR):
    eps1, eps2, eps3 = hubble_flow(N)
    ns = 1.0 - 2.0*eps1 - eps2
    r = 16.0*eps1
    nt = -2.0*eps1
    alpha_s = -2.0*eps1*eps2 - eps2*eps3
    H_over_MP = math.sqrt(8.0*math.pi**2*AS*eps1)
    return eps1, eps2, eps3, ns, r, nt, alpha_s, H_over_MP


def main() -> None:
    eps1, eps2, eps3, ns, r, nt, alpha_s, H = observables()
    print('HBMB conditional minimal slow-roll benchmark')
    print('='*72)
    print(f'N_*             = {NSTAR:.1f}')
    print(f'Delta_G         = {DELTA_G:.12f}')
    print(f'alpha_eff       = {ALPHA_EFF:.12f}')
    print(f'epsilon_1*      = {eps1:.12e}')
    print(f'epsilon_2*      = {eps2:.12e}')
    print(f'epsilon_3*      = {eps3:.12e}')
    print(f'n_s             = {ns:.12f}')
    print(f'r               = {r:.12e}')
    print(f'n_t             = {nt:.12e}')
    print(f'alpha_s         = {alpha_s:.12e}')
    print(f'H_*/M_P         = {H:.12e}')

    targets = {
        'ns': (ns, 0.963700113013),
        'r': (r, 3.848952409255e-3),
        'nt': (nt, -4.811190511569e-4),
        'alpha_s': (alpha_s, -6.587251598519e-4),
    }
    for name, (value, target) in targets.items():
        if not math.isclose(value, target, rel_tol=3e-12, abs_tol=3e-12):
            raise RuntimeError(f'{name} reproducibility check failed: {value} != {target}')
    print('\nChecks passed.')


if __name__ == '__main__':
    main()
