from matplotlib import pyplot as plt
from scipy.integrate import odeint
from functions import timeSeries, plot_synchrograms, load_params, vector_field, get_phases, \
    plot_hist_synchronization, plot_trisurf_synchronization, frequency, fit_f, \
    freq_fit_function, fit_freq
import numpy as np
from scipy.signal import argrelmax


def dynamics():
    plot = not automate
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    # Call the ODE solver.
    wsol = odeint(vector_field, w0, t, args=(p,),
                  atol=abserr, rtol=relerr)

    # cut unstable points
    t = t[200:]
    wsol = wsol[200:, :]

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
        f_drive, f_driven = plot_synchrograms(t, phases, couplings_gl, n, quantif, plot)
        freq_drive.append(f_drive)
        freq_driven.append(f_driven)

    if not automate:
        fit_f_parameters = fit_f(t, phases, p)
        fit_freq_parameters2 = fit_freq(t, phases, p)

w0, p, couplings_gl = load_params() # load params from file

n = int(len(w0)/2)

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 100.0
numpoints = int(20*stoptime)

automate = False
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
    k_range = np.arange(0., 1., 0.2)
    freq_ratio = np.arange(0.11, 1., 0.2)

    #fit_parameters = [ 0.51226277, 0.11745076, 0.66762324, 0.28301999] # for frequency = ...
    f_parameters = [ 29.33256013,   1.99857513,   0.0785717 ] # for f = ...
    freq_parameters = [ 0.18666117,  0.98518362, -0.05311368, -0.00331252]
    p[0][4] = freq_fit_function(p[0][4], *freq_parameters)
    for k in k_range:
        p[1][6] = k
        for f_r in freq_ratio:
            p[1][4] = f_r * p[0][4]
            dynamics()

    plot_trisurf_synchronization(k_range, freq_ratio, freq_drive, freq_driven, p, quantif)
    plot_hist_synchronization(k_range, freq_ratio, freq_drive, freq_driven, p, quantif)

else:
    freq_drive=[]
    freq_driven=[]
    quantif=[]
    dynamics()

plt.show()

