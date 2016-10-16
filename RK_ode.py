import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import ode


class Differentiate:

    def __init__(self, f, z0, start, t_max, h, p):
        self.solver = ode(f)
        self.solver.set_integrator('dopri5')
        self.solver.set_f_params(*p)
        self.solver.set_initial_value(z0, start)
        self.time = np.linspace(start, t_max, (t_max - start) / h)
        self.end = t_max
        self.sol = np.empty((int((self.end - start) / h), 2))
        self.sol[0] = z0



    def differentiate(self):
        k = 1
        while self.solver.successful() and self.solver.t < self.end:
            self.solver.integrate(self.time[k])
            self.sol[k] = self.solver.y
            k += 1

    def plot(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.time, self.sol[:, 0], label='x')
        plt.plot(self.time, self.sol[:, 1], label='y')
        plt.xlabel('t')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(self.sol[:, 0], self.sol[:, 1])
        plt.show()


alfa = 0.1
mi = 1.
d = 1.
e = 1.
params = alfa, mi, d, e

x0 = 0.1
y0 = 0.1
z0 = [x0, y0]
t0 = 0.

t_max = 200.
h = 0.01


def fun(t, z, alfa, mi, d, e):
    '''
    :param alfa:
    :param mi:
    :param y: initial derrivative at x
    :param d: -saddle
    :param e: -node
    :return: array x and array d1
    Right hand side of the differential equations
      dx/dt = y
      dy/dt2 = -alfa(x^2 - mi)d1 - x(x+d)(x+e) / de = 0
    '''
    x, y = z
    return [y, -alfa * (x**2 - mi) * y - x * (x + d) * (x + e) / (d * e)]

d = Differentiate(fun, z0, t0, t_max, h, params)
d.differentiate()
d.plot()





'''
def simple_osc(t, z, k, m):
    x, y = z
    return [ y, -(k / m) * x]

params_so = [1., 1.]
d_so = Differentiate(simple_osc, z0, t0, t_max, h, params_so)
d_so.differentiate()
d_so.plot()
'''

