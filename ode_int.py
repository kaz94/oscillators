from matplotlib import pyplot as plt
from scipy.integrate import odeint
from functions import timeSeries, plot_synchrograms, load_params, vector_field, save_p_s, quantification

couplings_gl = []

w0, p = load_params(couplings_gl)
n = int(len(w0)/2)

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 100.0
numpoints = int(20*stoptime)

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

# dynamika i wykresy przebiegow czasowych
conn_no_dupl = timeSeries(t, wsol, n, p, phases)
# save phase-space charts for different parameters values
# save_p_s(n, wsol,p)

# synchrogramy
if len(couplings_gl) > 0:
    plot_synchrograms(t, phases, couplings_gl, n)

    print(couplings_gl)


'''
x' = 0, y' = 0. -> p stałe. jakobian
ślad, wyznacznik i wartosci wlasne w p. stałych
p stabilne,

'''