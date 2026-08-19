#!/usr/bin/env python3
"""Run the complete numerical validation suite for the manuscript."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    'hbmb_overlap_scaling.py',
    'hbmb_tail_sum.py',
    'hbmb_background_minimal.py',
    'hbmb_determinant_running.py',
    'hbmb_alpha_delta_matching.py',
    'hbmb_plateau_benchmark.py',
    'hbmb_mukhanov_slowroll.py',
    'hbmb_reheating.py',
]


def main() -> None:
    failures = []
    print('HBMB WDW reproducibility suite')
    print('='*72)
    for script in SCRIPTS:
        print(f'\n>>> {script}')
        proc = subprocess.run([sys.executable, str(ROOT/script)], cwd=ROOT)
        if proc.returncode:
            failures.append((script, proc.returncode))
    if failures:
        print('\nFAILED:')
        for script, rc in failures:
            print(f'  {script}: return code {rc}')
        raise SystemExit(1)
    print('\n' + '='*72)
    print('All numerical validation scripts completed successfully.')


if __name__ == '__main__':
    main()
