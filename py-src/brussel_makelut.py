from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.io.wavfile import write
from numba import njit
from tqdm.contrib.concurrent import process_map
from itertools import product


def lerp(a, b, t):
    return (1 - t) * a + t * b

@njit
def brusselator(state, a, b):
    x, y = state

    dx = a + x**2 * y - b * x - x
    dy = b * x - x**2 * y

    return np.array([dx, dy])

@njit
def solve(state, dt, a, b):
    k1 = brusselator(state, a, b)
    k2 = brusselator(state + (2 / 3.0) * dt * k1, a, b)

    state = state + dt * (k1 + 3.0 * k2) / 4.0
    return state

a_steps = 86
b_steps = 101

period_lut = np.zeros(a_steps * b_steps)
gmin_lut   = np.zeros(a_steps * b_steps)
gmax_lut   = np.zeros(a_steps * b_steps)

def process(ab):
    a, b = ab
    a_index = 0
    b_index = 0

    freq = 2000
    if a >= 2.1 and b <= 6.5:
        freq = 2000
    
    dt = 1 / freq
    state = np.array([2.49, 2.0])
    warmup = 0
    max_warmup = 2
    if a >= 2.1 and b <= 6.5:
        max_warmup = 6

    sample_count = 0
    gmin = float('inf')
    gmax = float('-inf')
    last_value = 9999999

    while True:
        state = solve(state, dt, a, b)
        value = state[0] - 2.5

        if last_value == 9999999:
            last_value = value

        if warmup >= max_warmup:
            sample_count += 1
            gmin = np.min([gmin, state[0]])
            gmax = np.max([gmax, state[0]])

        if last_value >= 0 and value < 0:
            if warmup < max_warmup:
                warmup += 1
            else:
                break
        last_value = value

    a_index_float = (a - 1.35) / 0.01
    b_index_float = (b - 6) / 0.02
    a_index       = int(np.round(a_index_float))
    b_index       = int(np.round(b_index_float))
    index         = a_index * b_steps + b_index

    return {
        "gmin": gmin,
        "gmax": gmax,
        "period": sample_count / freq,
        "index": index
    }


if __name__ == "__main__":
    a_vals = np.linspace(1.35, 2.2, a_steps)
    b_vals = np.linspace(6.0, 8.0, b_steps)

    results = process_map(process, list(product(a_vals, b_vals)), max_workers=None, chunksize=10)

    for entry in results:
        index             = entry['index']
        gmin_lut[index]   = entry['gmin']
        gmax_lut[index]   = entry['gmax']
        period_lut[index] = entry['period']

    np.set_printoptions(legacy='1.25')
    np.set_printoptions(threshold=sys.maxsize, linewidth=np.nan)
    f = open("brussel_lut-PARALLEL_2000flat-2-4warmup.py", "w")
    f.write('period_lut = ' + np.array2string(period_lut, separator=", ") + ';\n')
    f.write('gmin_lut   = ' + np.array2string(gmin_lut, separator=", ") + ';\n')
    f.write('gmax_lut   = ' + np.array2string(gmax_lut, separator=", ") + ';\n')
    f.close()

    f = open("brussel_lut-PARALLEL_2000flat-2-4warmup.c", "w")
    f.write(f'#define A_STEPS {a_steps}\n')
    f.write(f'#define B_STEPS {b_steps}\n')
    f.write('float brussel_period_lut[] = ' + np.array2string(period_lut, separator=", ").replace("[", "{").replace("]", "}") + ';\n')
    f.write('float brussel_gmin_lut[]   = ' + np.array2string(gmin_lut, separator=", ").replace("[", "{").replace("]", "}") + ';\n')
    f.write('float brussel_gmax_lut[]   = ' + np.array2string(gmax_lut, separator=", ").replace("[", "{").replace("]", "}") + ';\n')

    f.close()
