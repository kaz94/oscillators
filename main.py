from matplotlib import pyplot as plt
from scipy.integrate import odeint
from functions import  load_params, vector_field, get_phases, synchronization, poincare_protophases
from plotting import plot_synchrograms,timeSeries, arnold_tongue
import numpy as np

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 40.0
numpoints = int(20*stoptime)


def synchrograms_from_file():
    w0, p, couplings_gl = load_params() # load params from file
    n = int(len(w0)/2)
    q = dynamics(False, n, w0, p, couplings_gl, N=1, M=1)
    print("synchro:", q)


def dynamics(plot, n, w0, p, couplings_gl, N=1, M=1):
    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    # Call the ODE solver.
    wsol = odeint(vector_field, w0, t, args=(p,),
                  atol=abserr, rtol=relerr)

    # cut unstable points
    t = t[400:]
    wsol = wsol[400:, :]

    plot_params = {'figure.figsize': (12, 10),
                   'axes.labelsize': 'x-small',
                   'axes.titlesize': 'x-small',
                   'xtick.labelsize': 'x-small',
                   'ytick.labelsize': 'x-small'}
    plt.rcParams.update(plot_params)

    # dynamika i wykresy przebiegow czasowych
    phases = get_phases(wsol, n)
    protophases = poincare_protophases(t, wsol, phases, p)
    if plot:
        timeSeries(t, wsol, n, p, phases)

    # synchrogramy
    any_coupling = False
    for c in couplings_gl:
        if len(c[1]) > 0:
            any_coupling = True
    if any_coupling:
        if plot:
            plot_synchrograms(t, phases, couplings_gl)
        qq = synchronization(phases, couplings_gl, N, M)
        return qq


def automate(N, M):
    points = 7
    w0, p, couplings_gl = load_params()
    '''couplings_gl = []
    #     x,   y
    w0 = [0.1, 0.1,
          0.1, 0.1]
    #    alfa, mi,  d,    e,   freq, osc, k
    p = [[1.0, 1.0, 1.0, -1.0, 1.],
         [1.0, 1.0, 1.0, -1.0, 1., 0.0, 0.0]]

    for i, osc in enumerate(p):
        couplings_gl.append([i, list(zip(osc[5::2], osc[6::2]))])'''

    n = int(len(w0) / 2)
    quantif = []
    print(couplings_gl)
    # must have equal lengths:
    k_range = np.linspace(0., 1.0, points)
    delta_range = np.linspace(-0.15, 0.15, points)
    print("k:   f:  q:")
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
                qq = dynamics(False, n, w0, p, couplings_gl, N, M)
                quantif += qq
                print(k, p[1][4], qq)
                f.write(" ".join(map(str, [k, p[1][4], *qq])) + "\n")
                #print(k, p[1][4], quantif[ii])
                #f.write(" ".join(map(str, [k, p[1][4], quantif[ii]])) + "\n")
                ii += 1

'''automate(1, 3)
arnold_tongue(1, 3)
automate(1, 2)
arnold_tongue(1, 2)
automate(1, 1)
arnold_tongue(1, 1)
plt.savefig("1_2_3.png")
plt.show()'''



synchrograms_from_file()




'''(d^2x/dt^2 )+a*((x)^2-b)*(dx(t)/dt)+x*(x(t)+d)*(x(t)+e)*(1/(d*e))=0'''