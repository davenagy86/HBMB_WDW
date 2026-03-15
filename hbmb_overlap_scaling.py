import numpy as np
import math
from scipy.special import eval_legendre
from scipy.integrate import simpson


def overlap_grid(l, shape='gaussian', c=1.0, n=12001):
    th = np.linspace(0, math.pi, n)
    delta = c / (l + 0.5)
    x = (th - math.pi / 2) / delta

    if shape == 'gaussian':
        w = np.exp(-0.5 * x * x)
    elif shape == 'exponential':
        w = np.exp(-np.abs(x))
    elif shape == 'tophat':
        w = (np.abs(x) <= 1).astype(float)
    else:
        raise ValueError('Unknown shape')

    norm = np.sqrt(2 * math.pi * simpson(w * w * np.sin(th), th))
    Yl0 = np.sqrt((2 * l + 1) / (4 * math.pi)) * eval_legendre(l, np.cos(th))
    integrand = (1 / np.sqrt(4 * math.pi)) * (w / norm) * Yl0 * np.sin(th)
    val = 2 * math.pi * simpson(integrand, th)
    return abs(val)


def main():
    ls = np.array([10, 20, 40, 80, 160, 320], dtype=float)
    print('shape, raw_slope, eff_slope, raw_kernel_slope, eff_kernel_slope')
    for shape in ['gaussian', 'exponential', 'tophat']:
        vals = np.array([overlap_grid(int(l), shape, n=12001) for l in ls])
        eff = vals / np.sqrt(2 * ls + 1)
        Kraw = (2 * ls + 1) * vals**2 / (ls * (ls + 1))
        Keff = (2 * ls + 1) * eff**2 / (ls * (ls + 1))
        p_raw = np.polyfit(np.log(ls), np.log(vals), 1)[0]
        p_eff = np.polyfit(np.log(ls), np.log(eff), 1)[0]
        q_raw = np.polyfit(np.log(ls), np.log(Kraw), 1)[0]
        q_eff = np.polyfit(np.log(ls), np.log(Keff), 1)[0]
        print(shape, f'{p_raw:.6f}', f'{p_eff:.6f}', f'{q_raw:.6f}', f'{q_eff:.6f}')
        print('sqrt(l)*V_raw:', vals * np.sqrt(ls))
        print('l*V_eff:', eff * ls)


if __name__ == '__main__':
    main()
