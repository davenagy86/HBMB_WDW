#!/usr/bin/env python3
"""Numerical check of the illustrative adiabatic reheating source of Section 6."""
import numpy as np

NU = 0.2
N_C = 3.0
XI = 0.5
H = 1.0  # arbitrary units; Q then has rho/time units


def n_acc(N):
    # Monotone illustrative capacity history for a source-sign check only.
    return np.exp(2.0*N)


def rho_tail_ad(N):
    n = n_acc(N)
    return 1.0 / ((n + NU) * (1.0 + n/N_C))


def d_rho_dN(N, h=1e-6):
    return (rho_tail_ad(N+h)-rho_tail_ad(N-h))/(2*h)


def q_rh_ad(N):
    return XI * H * max(0.0, -d_rho_dN(N))


def main():
    print('Illustrative adiabatic tail-decoupling source check')
    print('N        rho_tail^ad       d rho/dN          Q_rh^ad')
    for N in np.linspace(0, 2, 6):
        print(f'{N:4.1f}  {rho_tail_ad(N): .8e}  {d_rho_dN(N): .8e}  {q_rh_ad(N): .8e}')
    assert all(d_rho_dN(N) < 0 for N in np.linspace(0, 2, 21))
    assert all(q_rh_ad(N) >= 0 for N in np.linspace(0, 2, 21))
    print('\nChecks passed: the illustrative reservoir decreases and the source is non-negative.')
    print('The factor H converts a derivative per e-fold into a rate per cosmic time.')


if __name__ == '__main__':
    main()
