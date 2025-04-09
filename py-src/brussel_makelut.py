from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.io.wavfile import write


def lerp(a, b, t):
    return (1 - t) * a + t * b

def brusselator(state, a, b):
    x, y = state

    dx = a + x**2 * y - b * x - x
    dy = b * x - x**2 * y

    return np.array([dx, dy])

def solve(state, dt, a, b):
    k1 = brusselator(state, a, b)
    k2 = brusselator(state + (2 / 3.0) * dt * k1, a, b)

    state = state + dt * (k1 + 3.0 * k2) / 4.0
    return state

freq = 2000
dt = 1 / freq

a_steps = 13*10-9
b_steps = 101

period_lut = np.zeros(a_steps * b_steps)
gmin_lut   = np.zeros(a_steps * b_steps)
gmax_lut   = np.zeros(a_steps * b_steps)

a_index = 0
b_index = 0

for a in np.linspace(1, 2.2, a_steps):
    for b in np.linspace(6, 8, b_steps):
        print(f'{a_index} / {a_steps - 1} ({a})     {b_index} / {b_steps - 1} ({b})')
        state = np.array([1.8, 2.8])
        warmup = 0
        max_warmup = 2
        sample_count = 0
        gmin = float('inf')
        gmax = float('-inf')
        last_value = -99999

        while True:
            state = solve(state, dt, a, b)
            value = state[0] - a

            if last_value == -99999:
                last_value = value

            if warmup >= max_warmup:
                sample_count += 1
                gmin = np.min([gmin, state[0]])
                gmax = np.max([gmax, state[0]])

            if last_value <= 0 and value >= 0:
                if warmup < max_warmup:
                    warmup += 1
                else:
                    break
            last_value = value

        index = a_index * b_steps + b_index
        period_lut[index] = sample_count / freq
        gmin_lut[index] = gmin
        gmax_lut[index] = gmax
        b_index += 1
    a_index += 1
    b_index = 0

np.set_printoptions(legacy='1.25')
np.set_printoptions(threshold=sys.maxsize, linewidth=np.nan)
f = open("brussel_lut.py", "w")
f.write('period_lut = ' + np.array2string(period_lut, separator=", ") + ';\n')
f.write('gmin_lut   = ' + np.array2string(gmin_lut, separator=", ") + ';\n')
f.write('gmax_lut   = ' + np.array2string(gmax_lut, separator=", ") + ';\n')
f.close()

f = open("brussel_lut.c", "w")
f.write(f'#define A_STEPS {a_steps}\n')
f.write(f'#define B_STEPS {b_steps}\n')
f.write('float period_lut[] = ' + np.array2string(period_lut, separator=", ").replace("[", "{").replace("]", "}") + ';\n')
f.write('float gmin_lut[]   = ' + np.array2string(gmin_lut, separator=", ").replace("[", "{").replace("]", "}") + ';\n')
f.write('float gmax_lut[]   = ' + np.array2string(gmax_lut, separator=", ").replace("[", "{").replace("]", "}") + ';\n')

f.close()
