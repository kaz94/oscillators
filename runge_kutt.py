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


