import numpy as np
from scipy.integrate import OdeSolver

class RK2Ralston(OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, vectorized=False, **options):
        self.h = np.sign(t_bound - t0) * options.pop("max_step", abs(t_bound - t0))
        self.ddx = options.pop("ddx", None);
        super().__init__(fun, t0, y0, t_bound, vectorized=vectorized, **options)
    
    @property
    def dense_output(self):
        return None
    
    def _step_impl(self):
        k1 = self.fun(self.t, self.y)
        k2 = self.fun(self.t + self.h*(2/3.0), self.y + (2/3.0)*self.h*k1)

        if self.ddx is not None:
            self.ddx.append((k1[1] + 3.0*k2[1])/4.0)

        self.y = self.y + self.h * (k1 + 3.0*k2)/4.0
        self.t += self.h

        return True, None

class RK2Heun(OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, vectorized=False, **options):
        self.h = np.sign(t_bound - t0) * options.pop("max_step", abs(t_bound - t0))
        self.ddx = options.pop("ddx", None);
        super().__init__(fun, t0, y0, t_bound, vectorized=vectorized, **options)
    
    @property
    def dense_output(self):
        return None
    
    def _step_impl(self):
        k1 = self.fun(self.t, self.y)
        k2 = self.fun(self.t + self.h, self.y)

        if self.ddx is not None:
            self.ddx.append((k1 + k2)/2.0)

        self.y = self.y + self.h * (k1 + k2)/2.0
        self.t += self.h

        return True, None

class SymplecticEuler(OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, vectorized=False, **options):
        self.h = np.sign(t_bound - t0) * options.pop("max_step", abs(t_bound - t0))
        self.ddx = options.pop("ddx", None);
        super().__init__(fun, t0, y0, t_bound, vectorized=vectorized, **options)
    
    @property
    def dense_output(self):
        return None
    
    def _step_impl(self):
        ddx = self.fun(self.t, self.y)
        self.y = self.y + self.h * ddx
        self.t += self.h

        if self.ddx is not None:
            self.ddx.append(ddx)

        return True, None
