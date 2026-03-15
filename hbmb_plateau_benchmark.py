
import math
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(__file__))
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

A_s = 2.1e-9
MPL_GEV = 2.435e18
ALPHA = 0.75
DELTA = 0.8188863387
NSTAR = 55.0

def epsilon1(Nrem, alpha=ALPHA, Delta=DELTA):
    return alpha / (Nrem + Delta) ** 2

def epsilon2(Nrem, alpha=ALPHA, Delta=DELTA):
    return 2.0 / (Nrem + Delta)

def H_of_Nrem(Nrem, Nstar=NSTAR, alpha=ALPHA, Delta=DELTA, A_s=A_s):
    eps_star = epsilon1(Nstar, alpha, Delta)
    Hstar = math.sqrt(8.0 * math.pi**2 * A_s * eps_star)
    return Hstar * math.exp(alpha * (1.0 / (Nstar + Delta) - 1.0 / (Nrem + Delta)))

def observables(Nstar=NSTAR, alpha=ALPHA, Delta=DELTA, A_s=A_s):
    eps1 = epsilon1(Nstar, alpha, Delta)
    eps2 = epsilon2(Nstar, alpha, Delta)
    ns = 1.0 - 2.0 * eps1 - eps2
    r = 16.0 * eps1
    nt = -2.0 * eps1
    eps3 = 1.0 / (Nstar + Delta)
    alpha_s = -2.0 * eps1 * eps2 - eps2 * eps3
    Hstar = math.sqrt(8.0 * math.pi**2 * A_s * eps1)
    Hend = H_of_Nrem(0.0, Nstar, alpha, Delta, A_s)
    return {
        'alpha': alpha,
        'Delta': Delta,
        'epsilon1_star': eps1,
        'epsilon2_star': eps2,
        'n_s': ns,
        'r': r,
        'n_t': nt,
        'alpha_s': alpha_s,
        'H_star_Mpl': Hstar,
        'H_end_Mpl': Hend,
        'H_star_GeV': Hstar * MPL_GEV,
        'H_end_GeV': Hend * MPL_GEV,
        'V_star_quarter_GeV': (3.0 * Hstar**2) ** 0.25 * MPL_GEV,
        'V_end_quarter_GeV': (3.0 * Hend**2) ** 0.25 * MPL_GEV,
    }

if __name__ == "__main__":
    pars = observables()
    for k, v in pars.items():
        print(f"{k:20s} = {v}")

    Nrem = np.linspace(0.0, 70.0, 1200)
    e1 = np.array([epsilon1(n) for n in Nrem])
    e2 = np.array([epsilon2(n) for n in Nrem])
    H = np.array([H_of_Nrem(n) for n in Nrem])

    plt.figure(figsize=(8.5, 5.2))
    plt.semilogy(Nrem, e1, label=r'$\epsilon_1$')
    plt.semilogy(Nrem, e2, label=r'$\epsilon_2$')
    plt.axhline(1.0, linestyle='--', label='End of inflation')
    plt.xlabel('N_remaining')
    plt.ylabel('Hubble-flow parameters')
    plt.title('HBMB plateau benchmark with determinant-fixed alpha and matched Delta')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'plateau_slowroll.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(Nrem, H * MPL_GEV / 1e13)
    plt.xlabel('N_remaining')
    plt.ylabel(r'$H\;[10^{13}\,\mathrm{GeV}]$')
    plt.title('HBMB Hubble scale for the matched plateau benchmark')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'plateau_hubble.png'), dpi=200)
    plt.close()
