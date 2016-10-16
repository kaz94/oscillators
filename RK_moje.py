# d2x/dt2 + alfa(x^2-mi)dx/dt + x(x+d)(x+e)/de=0
# dx/dt = d1
# d2x/dt2 = -alfa(x^2 - mi)d1 - x(x+d)(x+e) / de = 0
import numpy as np
from matplotlib import pyplot as plt


def d2(x, alfa, mi, d1, d, e):
    '''
    :param x: initial coordinate
    :param alfa:
    :param mi:
    :param d1: initial derrivative at x
    :param d: -saddle
    :param e: -node
    :return: array x and array d1
    '''
    return -alfa * (x**2 - mi) * d1 - x * (x + d) * (x + e) / (d * e)


def runge_kutt(time, alfa, mi, d, e):
    x = [0.1]
    d1 = [0.1]
    h = 0.01
    for ii in range(0, len(time)-1):
        K1x = d1[ii]
        K1d1 = d2(x[ii], alfa, mi, d1[ii], d, e)

        K2x = d1[ii]
        K2d1 = d2(x[ii] + 0.5*h*K1x, alfa, mi, d1[ii] + 0.5*h*K1d1, d, e)

        K3x = d1[ii]
        K3d1 = d2(x[ii] + 0.5 * h * K2x, alfa, mi, d1[ii] + 0.5 * h * K2d1, d, e)

        K4x = d1[ii]
        K4d1 = d2(x[ii] + h * K3x, alfa, mi, d1[ii] + h * K3d1, d, e)

        x.append(x[ii] + 1./6. * (K1x + 2*K2x + 2*K3x + K4x) * h)
        d1.append(d1[ii] + 1./6. * (K1d1 + 2*K2d1 + 2*K3d1 + K4d1) * h)
    return [x, d1]
alfa = 0.1
mi = 1.
d = 1.
e = 1.
time = np.arange(0., 200., 0.01)
[x, d1] = runge_kutt(time, alfa, mi, d, e)
plt.subplot(2, 1, 1)
plt.xlabel("time, s")
plt.ylabel("x")
plt.plot(time, x)
plt.grid()
plt.subplot(2, 1, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,d1)
plt.show()
# synchrogramy oscylatorów
#macierz sąsiedztwa wczytywana z pliku