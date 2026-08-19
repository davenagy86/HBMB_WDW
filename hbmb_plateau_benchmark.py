#!/usr/bin/env python3
"""HBMB plateau benchmark and Figure 4/5 generation.

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

This script automatically switches from the exact Barnes-G residual near the
finite-L endpoint to the asymptotic residual for y=ln L > 8.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from PIL import Image

mp.mp.dps = 60
OUTDIR = Path(__file__).resolve().parent / 'figures'
OUTDIR.mkdir(parents=True, exist_ok=True)

B = mp.mpf(2)/3
C2 = mp.mpf(31)/360
C3 = -mp.mpf(1)/12
C4 = mp.mpf(47)/630
ZP = mp.diff(lambda s: mp.zeta(s), -1)
CA = 2 + 4*(mp.mpf(1)/12-ZP)

CCHI = 1.0
CL = 1.0
DELTA = 0.8366497586
ALPHA = CCHI/(2*float(B)*CL)
NSTAR = 55.0
AS = 2.1e-9
MP_GEV = 2.435e18


def save_rgb_png(fig, path):
    """Save a 600-dpi publication PNG and normalize it to 8-bit RGB."""
    path = Path(path)
    fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white', transparent=False)
    with Image.open(path) as im:
        im.convert('RGB').save(path, dpi=(600, 600), optimize=True)


def S_barnes(L):
    L=mp.mpf(L)
    return 4*(L*mp.log(mp.gamma(L+1))-mp.log(mp.barnesg(L+1)))+(2*L+1)*mp.log(L+1)


def bulk(L):
    L=mp.mpf(L)
    return 2*L**2*mp.log(L)-L**2+4*L*mp.log(L)


def g_exact_L(L):
    return mp.mpf('0.5')*(S_barnes(L)-bulk(L)-CA)


def D_exact_L(L):
    y=mp.log(L)
    return mp.diff(lambda yy: g_exact_L(mp.e**yy), y)


def g_asym_y(y):
    y=mp.mpf(y)
    return B*y + C2*mp.e**(-2*y) + C3*mp.e**(-3*y) + C4*mp.e**(-4*y)


def D_asym_y(y):
    y=mp.mpf(y)
    return B - 2*C2*mp.e**(-2*y) - 3*C3*mp.e**(-3*y) - 4*C4*mp.e**(-4*y)


def g_full_N(N):
    y=CL*(mp.mpf(N)+DELTA)
    # Barnes-G near the finite-L endpoint. At CMB-pivot L (~1.78e24),
    # direct Barnes-G subtraction is catastrophically cancellation-prone;
    # use the corrected asymptotic residual where y=ln L > 8.
    if y <= 8:
        return g_exact_L(mp.e**y)
    return g_asym_y(y)


def D_full_N(N):
    y=CL*(mp.mpf(N)+DELTA)
    if y <= 8:
        return D_exact_L(mp.e**y)
    return D_asym_y(y)


def eps_full(N):
    g=g_full_N(N)
    return float(mp.mpf(CCHI)*mp.mpf(CL)/2 * D_full_N(N)/g**2)


def eps_plateau(N):
    u=float(N)+DELTA
    return ALPHA/u**2


def H_ratio_full(N, Nref=NSTAR):
    g=g_full_N(N); gr=g_full_N(Nref)
    return float(mp.e**(-mp.mpf(CCHI)/(2*g)+mp.mpf(CCHI)/(2*gr)))


def H_ratio_plateau(N, Nref=NSTAR):
    u=float(N)+DELTA; ur=float(Nref)+DELTA
    return math.exp(ALPHA*(1/ur-1/u))


def observables(N):
    u=N+DELTA
    e1=ALPHA/u**2
    e2=2/u
    e3=1/u
    ns=1-2*e1-e2
    r=16*e1
    nt=-2*e1
    alphas=-2*e1*e2-e2*e3
    return e1,e2,ns,r,nt,alphas


def main():
    e1,e2,ns,r,nt,alphas=observables(NSTAR)
    # Use the full finite-L corrected epsilon at the pivot for amplitude normalization.
    e1_full=eps_full(NSTAR)
    Hstar=math.sqrt(8*math.pi**2*AS*e1_full)
    Hend=Hstar*H_ratio_full(0)
    rho_star_quarter=(3*(MP_GEV**2)*(Hstar*MP_GEV)**2)**0.25
    rho_end_quarter=(3*(MP_GEV**2)*(Hend*MP_GEV)**2)**0.25

    print('HBMB plateau benchmark')
    print('='*72)
    print(f'Delta_G             = {DELTA:.12f}')
    print(f'alpha_eff           = {ALPHA:.12f}')
    print(f'epsilon1*_plateau   = {e1:.12e}')
    print(f'epsilon1*_full      = {e1_full:.12e}')
    print(f'n_s                 = {ns:.12f}')
    print(f'r                   = {r:.12e}')
    print(f'n_t                 = {nt:.12e}')
    print(f'alpha_s             = {alphas:.12e}')
    print(f'H_* / M_P           = {Hstar:.12e}')
    print(f'H_end / M_P         = {Hend:.12e}')
    print(f'H_* [GeV]           = {Hstar*MP_GEV:.12e}')
    print(f'H_end [GeV]         = {Hend*MP_GEV:.12e}')
    print(f'rho_*^(1/4) [GeV]   = {rho_star_quarter:.12e}')
    print(f'rho_end^(1/4) [GeV] = {rho_end_quarter:.12e}')
    print('\nN_* sensitivity (minimal benchmark):')
    print(' N*        n_s             r')
    for N in (50,55,60):
        vals=observables(float(N))
        print(f'{N:3d}  {vals[2]:.9f}  {vals[3]:.9e}')

    Ns=np.linspace(0,70,420)
    ef=np.array([eps_full(float(N)) for N in Ns])
    ep=np.array([eps_plateau(float(N)) for N in Ns])

    # Endpoint-resolving grid for the lower diagnostic panels.
    Nz=np.linspace(0,6,360)
    efz=np.array([eps_full(float(N)) for N in Nz])
    epz=np.array([eps_plateau(float(N)) for N in Nz])
    eps_rel=np.abs(efz/epz-1.0)

    fig,(ax,axr)=plt.subplots(
        2,1,figsize=(5.8,5.4),dpi=600,
        gridspec_kw={'height_ratios':[2.25,1.0], 'hspace':0.12},
        sharex=False,
    )
    ax.plot(Ns,ef,label=r'$\epsilon_1^{\rm full}$')
    ax.plot(Ns,ep,'--',label=r'$\epsilon_1^{\rm pl}$')
    ax.axhline(1.0,linestyle=':',label='End of inflation')
    ax.set_yscale('log')
    ax.set_xlim(0,70)
    ax.set_ylabel(r'$\epsilon_1$',fontsize=12)
    ax.tick_params(labelsize=10.5)
    ax.legend(fontsize=9.5, loc='upper right')
    ax.grid(alpha=0.2)

    axr.plot(Nz,eps_rel)
    axr.set_yscale('log')
    axr.set_xlim(0,6)
    axr.set_xlabel(r'$N_{\rm rem}$',fontsize=12)
    axr.set_ylabel(r'$|\epsilon_1^{\rm full}/\epsilon_1^{\rm pl}-1|$',fontsize=12, labelpad=7)
    axr.tick_params(labelsize=10.5)
    axr.grid(alpha=0.2)
    fig.subplots_adjust(left=0.22,right=0.98,top=0.98,bottom=0.11)
    save_rgb_png(fig, OUTDIR/'plateau_slowroll.png')
    plt.close(fig)

    Hf=np.array([Hstar*H_ratio_full(float(N))*MP_GEV/1e13 for N in Ns])
    Hp=np.array([Hstar*H_ratio_plateau(float(N))*MP_GEV/1e13 for N in Ns])
    Hfz=np.array([Hstar*H_ratio_full(float(N))*MP_GEV/1e13 for N in Nz])
    Hpz=np.array([Hstar*H_ratio_plateau(float(N))*MP_GEV/1e13 for N in Nz])
    H_rel=np.abs(Hfz/Hpz-1.0)

    fig,(ax,axr)=plt.subplots(
        2,1,figsize=(5.8,5.4),dpi=600,
        gridspec_kw={'height_ratios':[2.25,1.0], 'hspace':0.12},
        sharex=False,
    )
    ax.plot(Ns,Hf,label='finite-$L$ corrected')
    ax.plot(Ns,Hp,'--',label='plateau approximation')
    ax.set_xlim(0,70)
    ax.set_ylabel(r'$H\;(10^{13}\,{\rm GeV})$',fontsize=12)
    ax.tick_params(labelsize=10.5)
    ax.legend(fontsize=9.5, loc='lower right')
    ax.grid(alpha=0.2)

    axr.plot(Nz,H_rel)
    axr.set_yscale('log')
    axr.set_xlim(0,6)
    axr.set_xlabel(r'$N_{\rm rem}$',fontsize=12)
    axr.set_ylabel(r'$|H_{\rm full}/H_{\rm pl}-1|$',fontsize=12, labelpad=7)
    axr.tick_params(labelsize=10.5)
    axr.grid(alpha=0.2)
    fig.subplots_adjust(left=0.22,right=0.98,top=0.98,bottom=0.11)
    save_rgb_png(fig, OUTDIR/'plateau_hubble.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
