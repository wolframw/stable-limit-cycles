import numpy as np
from scipy.io.wavfile import write
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import itertools
from math import isclose
import scipy.interpolate
import solver as solver

def lienard_even(t, state, mu):
    x, dx = state
    ddx = mu * (1 + x - x**2) * dx - (np.exp(x) - 1)
    return [dx, ddx]

periods_to_skip = 4
mus=[]
nspl=[]
gmin=[]
gmax=[]

for mu in np.linspace(0, 5, 101):
    mus.append(mu)

    t_span = (0, (1 + periods_to_skip) * 10 * int(10*mu+7))
    dt = 1/100
    init = [-1.21, 0.1]

    solution = solve_ivp(
        lambda t, state: lienard_even(t, state, mu),
        t_span, init, method=solver.RK2Ralston, max_step=dt,
    )

    half = 0            # half-wave
    period_samples = 1  # samples in the period
    skipped = 0
    extra_frac = 0
    gmin.append(np.min(solution.y[0]))
    gmax.append(np.max(solution.y[0]))
    for step in range(1, len(solution.y[0])):
        if solution.y[1][step - 1] > 0:
            half = 1
            period_samples += 1
        else:
            if half == 1:
                if skipped < periods_to_skip:
                    period_samples = 0
                    half = 0
                    skipped += 1
                    extra_frac = -1.21 - solution.y[0][step - 1]
                else:
                    if isclose(solution.y[1][step - 1], 0, abs_tol=0.00000001):
                        nspl.append((period_samples + extra_frac) * dt)
                        skipped = 0
                    else:
                        t = solution.y[0][step - 1] / (1.21 * solution.y[0][step - 2])
                        interp = (1 - t) * period_samples + t * (period_samples + 1)
                        nspl.append((interp + extra_frac) * dt)
                        skipped = 0
                    break
            period_samples += 1
    else:
        print(f"t_span not large enough {mu}")

print('[' + ', '.join([str(n) for n in nspl]) + ']')
print('[' + ', '.join([str(n) for n in gmin]) + ']')
print('[' + ', '.join([str(n) for n in gmax]) + ']')

plt.figure(figsize=(12, 7))
plt.plot(mus, nspl, 'bo')
plt.yscale('symlog')
plt.xlabel('µ')
plt.ylabel('Period in samples')
plt.title('µ-period relationship')
plt.legend()
plt.grid(True)
plt.show()
