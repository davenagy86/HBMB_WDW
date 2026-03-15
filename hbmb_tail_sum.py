import numpy as np


def tail_sum_numeric(L: int, nmax: int = 2_000_000) -> float:
    ell = np.arange(L + 1, nmax + 1, dtype=np.float64)
    return float(np.sum((2.0 * ell + 1.0) / (ell**2 * (ell + 1.0) ** 2)))


def tail_sum_exact(L: int) -> float:
    return 1.0 / (L + 1.0) ** 2


if __name__ == "__main__":
    print("L    numerical tail             exact tail                 abs. error")
    for L in [0, 1, 2, 5, 10, 100]:
        num = tail_sum_numeric(L)
        exact = tail_sum_exact(L)
        print(f"{L:3d}  {num: .15e}   {exact: .15e}   {abs(num-exact): .3e}")
