
import mpmath as mp

mp.mp.dps = 50

b = mp.mpf(2) / 3
c2 = mp.mpf(31) / 360
cchi = mp.mpf(1)

alpha = cchi / (2 * b)

def epsilon_full(u, b=b, c2=c2, cchi=cchi):
    ginv = b * u + c2 * mp.e**(-2*u)
    dginv = b - 2 * c2 * mp.e**(-2*u)
    return cchi * dginv / (2 * ginv**2)

delta_asy = mp.sqrt(alpha)
delta_match = mp.findroot(lambda u: epsilon_full(u) - 1, delta_asy)

u0 = delta_asy
s0 = c2 * mp.e**(-2*u0)
delta_corr = - s0 * (2*b*u0 + 1) / (2 * b**2 * u0)
delta_approx = u0 + delta_corr

print(f"b = {b}")
print(f"c2 = {c2}")
print(f"alpha = {alpha}")
print(f"Delta_asymptotic = {delta_asy}")
print(f"Delta_match = {delta_match}")
print(f"Delta_first_order = {delta_approx}")
print(f"epsilon_full(Delta_match) = {epsilon_full(delta_match)}")
