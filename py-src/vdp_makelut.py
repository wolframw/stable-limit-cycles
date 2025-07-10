import numpy as np
from scipy.io.wavfile import write
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import itertools
from math import isclose
import scipy.interpolate
import solver as solver

# Van der Pol equation
def vdp(t, state, mu):
    x, dx = state
    ddx = mu * (1 - x**2) * dx - x
    return [dx, ddx]

periods_to_skip = 1
mus=[]
nspl=[]

for mu in np.linspace(0, 10, 101):
    mus.append(mu)

    t_span = (0, (1 + periods_to_skip) * int(1.5*mu+7)) # very crude linear approximation on upper bound to keep computation time acceptable... don't mess with this!
    dt = 1/1000
    init = [2, 0.0]

    solution = solve_ivp(
        lambda t, state: vdp(t, state, mu),
        t_span, init, method=solver.RK2Ralston, max_step=dt,# atol=1, rtol=1 #ddx=ddx
    )

    half = 0            # half-wave
    period_samples = 1  # samples in the period
    skipped = 0
    extra_frac = 0
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
                    extra_frac = 2 - solution.y[0][step - 1]
                else:
                    # low-effort rootfinding
                    if isclose(solution.y[1][step - 1], 0, abs_tol=0.00000001):
                        nspl.append((period_samples + extra_frac) * dt)
                        skipped = 0
                    else:
                        t = solution.y[0][step - 1] / (2 * solution.y[0][step - 2])
                        interp = (1 - t) * period_samples + t * (period_samples + 1)
                        nspl.append((interp + extra_frac) * dt)
                        skipped = 0
                
                    break
            period_samples += 1
    else:
        print("t_span not large enough")

print('[' + ', '.join([str(n) for n in nspl]) + ']')

plt.figure(figsize=(12, 7))
plt.plot(mus, nspl, 'bo')
plt.yscale('symlog')
plt.xlabel('µ')
plt.ylabel('Period in samples')
plt.title('µ-period relationship')
plt.legend()
plt.grid(True)
plt.show()
