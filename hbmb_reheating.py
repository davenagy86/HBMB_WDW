import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

BASE = os.path.dirname(os.path.dirname(__file__))
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

MPL_GEV = 2.435e18
A_s = 2.1e-9
NSTAR = 55.0
ALPHA = 0.75
DELTA = 1.0
GSTAR = 106.75


def epsilon1(Nrem, alpha=ALPHA, Delta=DELTA):
    return alpha / (Nrem + Delta) ** 2


def H_of_Nrem(Nrem, Nstar=NSTAR, alpha=ALPHA, Delta=DELTA, A_s=A_s):
    eps_star = epsilon1(Nstar, alpha, Delta)
    Hstar = math.sqrt(8.0 * math.pi**2 * A_s * eps_star)
    return Hstar * math.exp(alpha * (1.0 / (Nstar + Delta) - 1.0 / (Nrem + Delta)))


def reheating_run(gamma_ratio):
    Hend = H_of_Nrem(0.0)
    rho_end = 3.0 * Hend ** 2
    Gamma = gamma_ratio * Hend

    def rhs(t, y):
        rho_phi, rho_r, a = y
        H = math.sqrt(max(rho_phi + rho_r, 0.0) / 3.0)
        return [
            -3.0 * H * rho_phi - Gamma * rho_phi,
            -4.0 * H * rho_r + Gamma * rho_phi,
            H * a,
        ]

    def eq_event(t, y):
        return y[1] - y[0]

    eq_event.terminal = True
    eq_event.direction = 1

    y0 = [rho_end, 0.0, 1.0]
    sol = solve_ivp(rhs, [0.0, 1.0e12], y0, events=eq_event, max_step=1.0e4, rtol=1e-8, atol=1e-12)

    teq = float(sol.t_events[0][0])
    rho_phi_eq, rho_r_eq, a_eq = sol.y_events[0][0]
    Nreh = math.log(a_eq)
    Treh = MPL_GEV * ((30.0 * rho_r_eq) / (math.pi**2 * GSTAR)) ** 0.25

    return {
        'gamma_ratio': gamma_ratio,
        'Gamma_over_Hend': gamma_ratio,
        't_eq_Mpl_inv': teq,
        'N_reh': Nreh,
        'T_reh_GeV': Treh,
        'rho_eq_Mpl4': rho_r_eq,
        'sol': sol,
    }


def Nk_from_reheating(Nre, wre=0.0):
    Hstar = H_of_Nrem(NSTAR)
    Hend = H_of_Nrem(0.0)
    Vend = 3.0 * Hend ** 2
    return 61.6 - math.log(Vend ** 0.25 / Hstar) - 0.25 * (1.0 - 3.0 * wre) * Nre


if __name__ == "__main__":
    runs = [reheating_run(g) for g in [1e-3, 1e-2, 1e-1]]
    print("Gamma/H_end    N_reh       T_reh [GeV]        N_k consistency")
    for run in runs:
        Nk = Nk_from_reheating(run['N_reh'])
        print(f"{run['gamma_ratio']: .1e}     {run['N_reh']: .6f}    {run['T_reh_GeV']: .6e}    {Nk: .6f}")

    plt.figure(figsize=(8.5, 5.2))
    for run in runs:
        sol = run['sol']
        Nvals = np.log(sol.y[2])
        plt.semilogy(Nvals, sol.y[0], label=fr'$\rho_\phi$, $\Gamma/H_{{end}}={run["gamma_ratio"]:.0e}$')
        plt.semilogy(Nvals, sol.y[1], linestyle='--')
    plt.xlabel('Post-inflation e-folds')
    plt.ylabel(r'Energy density [$M_{Pl}^4$]')
    plt.title('Perturbative HBMB reheating benchmark')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'reheating_evolution.png'), dpi=200)
    plt.close()
