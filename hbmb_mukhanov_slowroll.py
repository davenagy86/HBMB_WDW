
import math
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(__file__))
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

A_s = 2.1e-9
k_pivot = 0.05
ALPHA = 0.75
DELTA = 0.8188863387
NSTAR = 55.0

def epsilon1(Nrem, alpha=ALPHA, Delta=DELTA):
    return alpha / (Nrem + Delta) ** 2

def epsilon2(Nrem, alpha=ALPHA, Delta=DELTA):
    return 2.0 / (Nrem + Delta)

def spectra(Nstar=NSTAR, alpha=ALPHA, Delta=DELTA, A_s=A_s, k_pivot=k_pivot):
    eps1 = epsilon1(Nstar, alpha, Delta)
    eps2 = epsilon2(Nstar, alpha, Delta)
    ns = 1.0 - 2.0 * eps1 - eps2
    nt = -2.0 * eps1
    r = 16.0 * eps1
    eps3 = 1.0 / (Nstar + Delta)
    alpha_s = -2.0 * eps1 * eps2 - eps2 * eps3
    ks = np.logspace(-4, 1, 800)
    lnkk = np.log(ks / k_pivot)
    Ps = A_s * np.exp((ns - 1.0) * lnkk + 0.5 * alpha_s * lnkk**2)
    Pt = r * A_s * np.exp(nt * lnkk)
    return ks, Ps, Pt, ns, r, nt, alpha_s

if __name__ == "__main__":
    ks, Ps, Pt, ns, r, nt, alpha_s = spectra()
    print(f"alpha = {ALPHA:.10f}")
    print(f"Delta = {DELTA:.10f}")
    print(f"n_s = {ns:.9f}")
    print(f"r   = {r:.9f}")
    print(f"n_t = {nt:.9f}")
    print(f"alpha_s = {alpha_s:.9f}")

    plt.figure(figsize=(8.5, 5.2))
    plt.loglog(ks, Ps, label=r'$\mathcal{P}_{\mathcal{R}}(k)$')
    plt.loglog(ks, Pt, label=r'$\mathcal{P}_{T}(k)$')
    plt.axvline(0.05, linestyle='--', label='pivot')
    plt.xlabel(r'$k\,[\mathrm{Mpc}^{-1}]$')
    plt.ylabel('dimensionless power')
    plt.title('HBMB primordial spectra for the matched plateau benchmark')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'primordial_spectra.png'), dpi=200)
    plt.close()
