from matplotlib import pyplot as plt
from scipy.integrate import odeint
from functions import timeSeries, plot_synchrograms, load_params, vector_field, get_phases, plot_trisurf_synchronization
import numpy as np
from scipy.signal import argrelmax


def dynamics():
    plot = not automate
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    # Call the ODE solver.
    wsol = odeint(vector_field, w0, t, args=(p,),
                  atol=abserr, rtol=relerr)

    # cut unstable points
    t = t[300:]
    wsol = wsol[300:, :]

    plot_params = {'figure.figsize': (12, 10),
                   'axes.labelsize': 'x-small',
                   'axes.titlesize': 'x-small',
                   'xtick.labelsize': 'x-small',
                   'ytick.labelsize': 'x-small'}
    plt.rcParams.update(plot_params)

    # dynamika i wykresy przebiegow czasowych
    phases = get_phases(wsol, n)
    '''if plot:
        timeSeries(t, wsol, n, p, phases)'''

    # synchrogramy
    any_coupling = False
    for c in couplings_gl:
        if len(c[1]) > 0:
            any_coupling = True

    if any_coupling:
        f_drive, f_driven = plot_synchrograms(t, phases, couplings_gl, n, quantif, idx, plot)
        freq_drive.append(f_drive)
        freq_driven.append(f_driven)


    '''if not automate:
        fit_f_parameters = fit_f(t, phases, p)
        fit_freq_parameters2 = fit_freq(t, phases, p)'''

w0, p, couplings_gl = load_params() # load params from file

n = int(len(w0)/2)

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 120.0
numpoints = int(20*stoptime)

idx = 0
automate = True # False - read a file and generate synchrograms
if automate:
    couplings_gl = []
    #     x,   y
    w0 = [0.1, 0.1,
          0.1, 0.1]
    #    alfa, mi,  d,    e,   f,   osc, k
    p = [[1.0, 1.0, 1.0, -1.0, 1.0],
         [1.0, 1.0, 1.0, -1.0, 3, 0.0, 0.0]]
    for i, osc in enumerate(p):
        couplings_gl.append([i, list(zip(osc[5::2], osc[6::2]))])
    n = int(len(w0)/2)
    quantif = []
    freq_drive = []
    freq_driven = []
    k_range = np.arange(0., 0.8, 0.04)
    delta_range = np.arange(-1.2, 1.2, 0.1)

    p[0][4] = 1.
    for k in k_range:
        p[1][6] = k
        for delta in delta_range:
            p[1][4] = p[0][4] + delta
            dynamics()
            idx+=1
    plot_trisurf_synchronization(k_range, delta_range, freq_drive, freq_driven, p, quantif)

else:
    freq_drive=[]
    freq_driven=[]
    quantif=[]
    dynamics()

plt.show()

