from runge_kutt import Differentiate

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


def simple_osc(t, z, k, m):
    x, y = z
    return [ y, -(k / m) * x]

params_so = [1., 1.]
d_so = Differentiate(simple_osc, z0, t0, t_max, h, params_so)
d_so.differentiate()
d_so.plot()
