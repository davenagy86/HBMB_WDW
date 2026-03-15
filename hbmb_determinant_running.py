import mpmath as mp

mp.mp.dps = 50
A_G = mp.qfrom('1.28242712910062263687534256886979172776768892732500119206374')


def S(L):
    return mp.nsum(lambda k: (2 * k + 1) * mp.log(k * (k + 1)), [1, L])


def asymptotic_S(L):
    return (
        2 * L**2 * mp.log(L)
        - L**2
        + 4 * L * mp.log(L)
        + (mp.mpf(4) / 3) * mp.log(L)
        + 2 + 4 * mp.log(A_G)
        + mp.mpf(31) / (180 * L**2)
    )


def residuals(L):
    exact = S(L)
    bulk = 2 * L**2 * mp.log(L) - L**2
    edge = 4 * L * mp.log(L)
    logpart = (mp.mpf(4) / 3) * mp.log(L)
    return {
        'exact': exact,
        'after_bulk': exact - bulk,
        'after_bulk_edge': exact - bulk - edge,
        'after_bulk_edge_log': exact - bulk - edge - logpart,
    }


def main():
    print('L, exact, asymptotic, error')
    for L in [10, 20, 50, 100, 200]:
        exact = S(L)
        approx = asymptotic_S(L)
        print(L, exact, approx, exact - approx)
    print('\nResidual hierarchy')
    for L in [10, 20, 50, 100]:
        r = residuals(L)
        print('\nL =', L)
        for k, v in r.items():
            print(k, v)
    print('\nDominance of log over L^-2')
    for L in [5, 10, 20, 50, 100]:
        logterm = (mp.mpf(4) / 3) * mp.log(L)
        correction = mp.mpf(31) / (180 * L**2)
        print(L, logterm, correction, abs(logterm / correction))


if __name__ == '__main__':
    main()
