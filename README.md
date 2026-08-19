# HBMB Wheeler-DeWitt cosmology — revised numerical package

This repository accompanies the revised manuscript **“Holographic Bit–Mode Balance Regularization of Wheeler–DeWitt Cosmology: Determinant Running and Inflationary Closure.”**

The code reflects the major-revision version. In particular, it does **not** treat the inflationary coefficient as a parameter-free output of the determinant alone. The normalized uncoupled single-effective-species reference determinant fixes

`b_1sp = 2/3`,

while the leading plateau coefficient is

`alpha_eff = c_chi / (2 b_1sp c_L) = (3/4)(c_chi/c_L)`.

For the minimal benchmark `c_chi = c_L = 1`, the standard Barnes-G continuation of the exact integer determinant sum gives

`Delta_match^(G) = 0.8366497586`,

and for `N_* = 55` the benchmark values are

`n_s = 0.963700113`,  `r = 3.8489524e-3`.

These are benchmark values of the conditional reference realization, not parameter-free predictions of complete HBMB microphysics.

## Files

- `hbmb_overlap_scaling.py` — tangential DFT basis, centered/displaced-strip parity, overlap scaling, and bounded nonuniform-channel robustness.
- `hbmb_tail_sum.py` — exact telescoping representative tail identity.
- `hbmb_background_minimal.py` — illustrative effective-fluid minisuperspace closure; regenerates the Section 3 figures.
- `hbmb_determinant_running.py` — exact integer determinant sum, Barnes-G identity check, corrected `L^-3`/`L^-4` asymptotics, and convergence diagnostics.
- `hbmb_alpha_delta_matching.py` — symbolic `c_chi,c_L` dependence and finite-L Barnes-G matching.
- `hbmb_plateau_benchmark.py` — revised benchmark observables, `N_*` sensitivity, and the finite-L versus plateau figures.
- `hbmb_mukhanov_slowroll.py` — scalar/tensor leading slow-roll benchmark.
- `hbmb_reheating.py` — checks the sign and dimensional role of the illustrative adiabatic reheating source.
- `run_all_checks.py` — executes the complete validation suite.
- `requirements.txt` — package versions used in the revision audit.

## Reproduce

With Python 3.13 or a compatible recent Python version:

```bash
python -m pip install -r requirements.txt
python run_all_checks.py
```

The figure-generating scripts write the manuscript PNGs to `figures/` at 600 dpi and normalize the raster output to non-transparent 8-bit RGB for publication-oriented reproducibility.

## Scope

The numerical determinant in this package is the normalized **uncoupled single-effective-species angular reference determinant**. It is not a computation of the full gauge-fixed Einstein–matter fluctuation determinant. The response parameter `c_chi`, the cutoff–e-fold mapping parameter `c_L`, and microscopic reheating remain effective/open inputs in the present work.
