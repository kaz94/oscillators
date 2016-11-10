from matplotlib import pyplot as plt
from ReadAdjacencyMatrix import read_file
from scipy.integrate import odeint
from scipy.signal import hilbert
import numpy as np
from scipy import signal
import pandas as pd
from scipy.signal import argrelmax

couplings_gl = []


def load_params():
    w = []
    params = []
    adj_list = read_file()
    for i, osc in enumerate(adj_list):
        # Oscillator(x, y, alfa, mi, d, e)
        w.append(osc[0])  # x0
        w.append(osc[1])  # y0
        params.append(osc[2:])  # alfa, mi, d, e, coupling1, k1, coupl2, k2, ...
        couplings_gl.append([i, list(zip(osc[7::2], osc[8::2]))])
    return w, params


def vector_field(w, t, p):
    """
        w :  vector of the state variables: w = [x1,y1,x2,y2,...]
        p = [alfa1, mi1, d1, e1, k1, alfa2, mi2, d2, e2, k2, ...]
        y: initial derrivative at x
        d: -saddle
        e: -node
          dx/dt = y
          dy/dt2 = -alfa(x^2 - mi)y - f*x(x+d)(x+e) / de = 0
    """

    # Create equasions = (x1',y1',x2',y2'):
    equasions = []
    y = w[1::2]
    x = w[0::2]
    for o in range(0, int(len(w)/2)):
        equasions.append(y[o])
        params = {'alfa': p[o][0], 'mi': p[o][1], 'd': p[o][2], 'e': p[o][3], 'f': p[o][4]}
        couplings = p[o][5:]
        couplings = list(zip(couplings[0::2], couplings[1::2]))
        eq = -1 * params['alfa'] * (x[o] ** 2 - params['mi']) * \
             y[o] - params['f'] * x[o] * (x[o] + params['d']) * (x[o] + params['e'])
        for c in couplings:
            eq += c[1] * x[int(c[0])]
        equasions.append(eq)

    return equasions


w0, p = load_params()
n = int(len(w0)/2)
# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 150.0
numpoints = 3000

t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]


# Call the ODE solver.
wsol = odeint(vector_field, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

plot_params = {'figure.figsize': (12, 10),
             'axes.labelsize': 'x-small',
             'axes.titlesize':'x-small',
             'xtick.labelsize':'x-small',
             'ytick.labelsize':'x-small'}
plt.rcParams.update(plot_params)

synchrograms = []
phases = []

fig, axes = plt.subplots(nrows=n, ncols=3)
for i in range(0, n):
    plt.subplot(n, 3, 3*i+1)
    title = ["Osc", i, "alfa=", p[i][0], "mi=", p[i][1], "d=", p[i][2], "e=", p[i][3], "f=", p[i][4], "Coupled to:" ]
    if len(p[i]) > 5:
        coupl = list(zip(p[i][5::2], p[i][6::2]))
        print(coupl)
        for c in coupl:
            title.append("osc")
            title.append(int(c[0]))
            title.append("k:")
            title.append(c[1])
    plt.title(' '.join(str(t) for t in title))
    plt.plot(t, wsol[:,2*i], label='x')
    plt.plot(t, wsol[:,2*i+1], label='y')
    plt.xlabel('t')
    plt.grid(True)
    plt.legend(prop={'size': 30/n})

    analytic_signal = hilbert(wsol[:,2*i])
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    phase = np.angle(analytic_signal)

    plt.subplot(n, 3, 3*i+2)
    plt.title("Phase space")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(wsol[:,2*i], wsol[:,2*i+1])

    plt.subplot(n, 3, 3*i+3)

    plt.title("Phase")
    plt.xlabel("t")
    plt.ylabel("Phase")



    plt.plot(t, phase, label='phase')
    #plt.plot(t, instantaneous_phase, label='inst phase')
    plt.legend(prop={'size': 30/n})

    phases.append(phase)


fig.tight_layout()
plt.savefig("/home/kasia/Pulpit/inzynierka/wykres")
plt.show()

# synchrogramy

plot_params = {'figure.figsize': (10, 5),
             'axes.labelsize': 'x-small',
             'axes.titlesize':'x-small',
             'xtick.labelsize':'x-small',
             'ytick.labelsize':'x-small'}
plt.rcParams.update(plot_params)

t = np.asarray(t)
t = t[50:]
for i, p in enumerate(phases):
    phases[i] = p[50:]  # delete first few points

i_subplots = 0
for i, osc in enumerate(couplings_gl):
    if len(osc[1]) > 0:
        i_subplots += 1

idx = 0
for i, osc in enumerate(couplings_gl):
    if len(osc[1]) > 0:
        print(osc[1])
        idx += 1
        plt.subplot(i_subplots,1,idx)
        print(couplings_gl)

        peakind = argrelmax(phases[i])
        #print(peakind)
        plt.scatter(t[peakind], phases[i][peakind], label=str(i))

        for c in osc[1]:
            # peakind = signal.find_peaks_cwt(phases[i], np.arange(1, 10))
            plt.scatter(t[peakind], phases[int(c[0])][peakind], label=str(int(c[0])), color="red")
            plt.plot(t, phases[i])
            plt.plot(t, phases[int(c[0])])

            plt.title("Synchrogram")
            plt.xlabel("t")
            plt.ylabel("Phase")
            plt.legend(prop={'size': 50 / n})


plt.show()


'''
for i in range(0, n):
    title = ["Phase space, ", i, "alfa=", p[i][0], "mi=", p[i][1], "d=", p[i][2], "e=", p[i][3], "f=", p[i][4], "Coupled to:" ]
    plt.plot(wsol[:,2*i], wsol[:,2*i+1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(prop={'size': 30/n})
    plt.savefig("".join(["/home/kasia/Pulpit/inzynierka/mi",str(i)]))
    plt.clf()


'''


'''
znajdowanie max lok sygnału
rzutowanie na drugi sygnał
wykres maximow pierwszego i odpowiadających punktów drugiego
fazowa - zgodność
czestotliwosciowa
miara synchronizacji

dla zmiennego k

x' = 0, y' = 0. -> p stałe. jakobian
ślad, wyznacznik i wartosci wlasne w p. stałych
p stabilne,



'''