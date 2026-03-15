import math
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(__file__))
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def N_acc(x, xF=0.7):
    return np.pi * x**4 / (x**2 + xF**2)


def dlnN_dlnx(x, xF=0.7):
    return 4.0 - 2.0 * x**2 / (x**2 + xF**2)


def w_minimal(x, nu=0.2, xF=0.7):
    N = N_acc(x, xF)
    return -1.0 + (1.0 / 3.0) * (N / (N + nu)) * dlnN_dlnx(x, xF)


def w_exit(x, nu=0.2, xF=0.7, Nc=3.0):
    N = N_acc(x, xF)
    bracket = (N / (N + nu)) + (N / (N + Nc))
    return -1.0 + (1.0 / 3.0) * dlnN_dlnx(x, xF) * bracket


def epsilon_H(w):
    return 1.5 * (1.0 + w)


def find_x_end(nu=0.2, xF=0.7, Nc=3.0):
    grid = np.logspace(-3, 1, 20000)
    vals = np.array([w_exit(x, nu, xF, Nc) + 1.0 / 3.0 for x in grid])
    idx = np.where(np.diff(np.sign(vals)))[0][0]
    x1, x2 = grid[idx], grid[idx + 1]
    y1, y2 = vals[idx], vals[idx + 1]
    return x1 - y1 * (x2 - x1) / (y2 - y1)


if __name__ == "__main__":
    xs = np.logspace(-3, 1, 3000)
    w1 = np.array([w_minimal(x) for x in xs])
    w2 = np.array([w_exit(x) for x in xs])
    e2 = epsilon_H(w2)
    x_end = find_x_end()

    plt.figure(figsize=(8.5, 5.2))
    plt.semilogx(xs, w1, label='Minimal closure')
    plt.semilogx(xs, w2, label='Exit closure')
    plt.axhline(-1.0/3.0, linestyle='--', label='Acceleration threshold')
    plt.axvline(x_end, linestyle=':', label=f'x_end ≈ {x_end:.3f}')
    plt.xlabel('x')
    plt.ylabel('w(x)')
    plt.title('HBMB background equation of state')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'minimal_background_w.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.semilogx(xs, e2)
    plt.axhline(1.0, linestyle='--', label='End of inflation')
    plt.axvline(x_end, linestyle=':', label=f'x_end ≈ {x_end:.3f}')
    plt.xlabel('x')
    plt.ylabel('epsilon_H(x)')
    plt.title('HBMB background slow-roll parameter')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'minimal_background_epsilon.png'), dpi=200)
    plt.close()

    print(f"x_end ≈ {x_end:.6f}")
    for x in [0.05, 0.1, 0.2, 0.3, 0.4, x_end, 1.0]:
        w = w_exit(x)
        e = epsilon_H(w)
        print(f"x={x:.6f}  w={w:.8f}  epsilon_H={e:.8f}")
