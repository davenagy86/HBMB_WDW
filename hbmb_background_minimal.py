#!/usr/bin/env python3
"""Reproduce the illustrative Section 3 effective-fluid benchmark and Figures 2-3."""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from PIL import Image

NU = 0.2
A_F = 0.7
N_C = 3.0
OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)


def save_rgb_png(fig, path):
    """Save a 600-dpi publication PNG and normalize it to 8-bit RGB."""
    path = Path(path)
    fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white', transparent=False)
    with Image.open(path) as im:
        im.convert('RGB').save(path, dpi=(600, 600), optimize=True)


def n_acc(abar):
    abar = np.asarray(abar, dtype=float)
    return np.pi * abar**4 / (abar**2 + A_F**2)


def dln_nacc_dln_abar(abar):
    abar = np.asarray(abar, dtype=float)
    return 4.0 - 2.0 * abar**2 / (abar**2 + A_F**2)


def w_min(abar):
    n = n_acc(abar)
    return -1.0 + (dln_nacc_dln_abar(abar)/3.0) * n/(n+NU)


def w_exit(abar):
    n = n_acc(abar)
    return -1.0 + (dln_nacc_dln_abar(abar)/3.0) * (
        n/(n+NU) + n/(n+N_C)
    )


def eps1_exit(abar):
    return 1.5 * (1.0 + w_exit(abar))


def main():
    a_end = brentq(lambda z: float(eps1_exit(z) - 1.0), 0.3, 0.7)
    print(f'a_bar_end = {a_end:.12f}')
    print('a_bar       w_exit          epsilon_1')
    for a in [0.05,0.10,0.20,0.30,0.40,a_end,1.0]:
        print(f'{a:10.6f}  {float(w_exit(a)): .10f}  {float(eps1_exit(a)): .10f}')

    aa = np.logspace(-3, 1, 1200)

    fig, ax = plt.subplots(figsize=(5.6, 4.25), dpi=600)
    ax.plot(aa, w_min(aa), label='Minimal effective-fluid closure')
    ax.plot(aa, w_exit(aa), label='Capacity-suppressed exit closure')
    ax.axhline(-1/3, linestyle='--', label='Acceleration threshold')
    ax.axvline(a_end, linestyle=':', label=rf'$\bar a_{{\rm end}}\simeq {a_end:.3f}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\bar a=a/a_0$', fontsize=13)
    ax.set_ylabel(r'$w(\bar a)$', fontsize=13)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10.5, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)
    fig.subplots_adjust(top=0.78, left=0.14, right=0.98, bottom=0.15)
    save_rgb_png(fig, OUT/'minimal_background_w.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.15), dpi=600)
    ax.plot(aa, eps1_exit(aa), label=r'$\epsilon_1(\bar a)$')
    ax.axhline(1.0, linestyle='--', label=r'End of acceleration: $\epsilon_1=1$')
    ax.axvline(a_end, linestyle=':', label=rf'$\bar a_{{\rm end}}\simeq {a_end:.3f}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\bar a=a/a_0$', fontsize=13)
    ax.set_ylabel(r'$\epsilon_1(\bar a)$', fontsize=13)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10.5, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)
    fig.subplots_adjust(top=0.78, left=0.14, right=0.98, bottom=0.15)
    save_rgb_png(fig, OUT/'minimal_background_epsilon.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
