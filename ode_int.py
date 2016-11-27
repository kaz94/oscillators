from matplotlib import pyplot as plt
from scipy.integrate import odeint
from functions import timeSeries, plot_synchrograms, load_params, vector_field, save_p_s, quantification
import numpy as np

couplings_gl = []

# w0, p = load_params(couplings_gl) # load params from file
#     x,   y
w0 = [0.1, 0.1,
      0.1, 0.1]
#    alfa, mi,  d,    e,   f,   osc, k
p = [[1.0, 1.0, 1.0, -1.0, 1.0],
     [1.0, 1.0, 1.0, -1.0, 0.8, 0.0, 0.0]]
for i, osc in enumerate(p):
    couplings_gl.append([i, list(zip(osc[5::2], osc[6::2]))])

n = int(len(w0)/2)

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 100.0
numpoints = int(20*stoptime)

#k_range = np.arange(0,1,0.1)
#freq_ratio = np.arange(0.1,1,0.1)
k_range = np.arange(0, 1)
freq_ratio = np.arange(1, 2)

for k in k_range:
    for f_r in freq_ratio:
        t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
        # Call the ODE solver.
        wsol = odeint(vector_field, w0, t, args=(p,),
                      atol=abserr, rtol=relerr)
        # cut unstable points
        t = t[100:]
        wsol = wsol[100:, :]

        plot_params = {'figure.figsize': (12, 10),
                     'axes.labelsize': 'x-small',
                     'axes.titlesize':'x-small',
                     'xtick.labelsize':'x-small',
                     'ytick.labelsize':'x-small'}
        plt.rcParams.update(plot_params)


        synchrograms = []
        phases = []

        # dynamika i wykresy przebiegow czasowych
        conn_no_dupl = timeSeries(t, wsol, n, p, phases)
        # save phase-space charts for different parameters values
        # save_p_s(n, wsol,p)

        # synchrogramy
        any_coupling = False
        for c in couplings_gl:
            if len(c[1]) > 0:
                any_coupling = True

        if any_coupling:
            plot_synchrograms(t, phases, couplings_gl, n)



'''


'''