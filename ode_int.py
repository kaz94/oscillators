from matplotlib import pyplot as plt
from scipy.integrate import odeint
from functions import plot_synchrograms, load_params, vector_field, get_phases
from plotting import timeSeries, arnold_tongue
import numpy as np

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 120.0
numpoints = int(20*stoptime)


def synchrograms_from_file():
    w0, p, couplings_gl = load_params() # load params from file
    n = int(len(w0)/2)
    quantif=[]
    dynamics(True, n, w0, p, couplings_gl, quantif)


def dynamics(plot, n, w0, p, couplings_gl, quantif, N=1, M=1):
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
    if plot:
        timeSeries(t, wsol, n, p, phases)

    # synchrogramy
    any_coupling = False
    for c in couplings_gl:
        if len(c[1]) > 0:
            any_coupling = True

    if any_coupling:
        plot_synchrograms(t, phases, couplings_gl, quantif, N, M, plot)


def automate(N, M):
    couplings_gl = []
    #     x,   y
    w0 = [0.1, 0.1,
          0.1, 0.1]
    n = int(len(w0) / 2)
    #    alfa, mi,  d,    e,   freq, osc, k
    p = [[1.0, 1.0, 1.0, -1.0, 1.],
         [1.0, 1.0, 1.0, -1.0, 1., 0.0, 0.0]]

    for i, osc in enumerate(p):
        couplings_gl.append([i, list(zip(osc[5::2], osc[6::2]))])
    quantif = []
    # must have equal lengths:
    k_range = np.linspace(0., 1.0, 40)
    delta_range = np.linspace(-0.15, .15, 40)

    with open("out"+str(N)+"_"+str(M)+".dat", 'w') as f:
        # N - drive (0), M - driven (1)
        base_freq = 0.2
        ii = 0
        p[0][4] = N*base_freq
        for k in k_range:
            p[1][6] = k
            for delta in delta_range:
                # freq = 0.2*sqrt(f)
                p[1][4] = M*base_freq + delta
                dynamics(False, n, w0, p, couplings_gl, quantif, N, M)
                f.write(" ".join(map(str, [k, p[1][4], quantif[ii]])) + "\n")
                ii += 1


#automate(1, 1)
#automate(1, 2)
#automate(1, 3)
arnold_tongue(1, 3)
arnold_tongue(1, 2)
arnold_tongue(1, 1)
plt.show()
plt.savefig("1_2_3.png")
#synchrograms_from_file()