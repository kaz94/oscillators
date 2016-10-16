from matplotlib import pyplot as plt
from ReadAdjacencyMatrix import read_file
from scipy.integrate import odeint
from oscillator import Oscillator

couplings_gl = []
def load_params():
    w = []
    params = []
    adj_list = read_file()
    for i, osc in enumerate(adj_list):
        # Oscillator(x, y, alfa, mi, d, e)
        w.append(osc[0]) #x0
        w.append(osc[1]) #y0
        params.append(osc[2:]) # alfa, mi, d, e, coupling1, k1, coupl2, k2, ...
        couplings_gl.append([i, list(zip(osc[6::2], osc[7::2]))])
    return w, params


def vector_field(w, t, p):
    """
        w :  vector of the state variables: w = [x1,y1,x2,y2,...]
        p = [alfa1, mi1, d1, e1, k1, alfa2, mi2, d2, e2, k2, ...]
        y: initial derrivative at x
        d: -saddle
        e: -node
          dx/dt = y
          dy/dt2 = -alfa(x^2 - mi)d1 - x(x+d)(x+e) / de = 0
    """

    # Create f = (x1',y1',x2',y2'):
    f = []
    y = w[1::2]
    x = w[0::2]
    for o in range(0, int(len(w)/2)):
        f.append(y[o])
        params = {'alfa' : p[o][0], 'mi' : p[o][1], 'd' : p[o][2], 'e' : p[o][3]}
        couplings = p[o][4:]
        couplings = list(zip(couplings[0::2], couplings[1::2]))
        eq = -1 * params['alfa'] * (x[o] ** 2 - params['mi']) * \
             y[o] - x[o] * (x[o] + params['d']) * (x[o] + params['e']) / (params['d'] * params['e'])
        for c in couplings:
            eq += c[1] * x[int(c[0])]
        f.append(eq)

    return f


w0, p = load_params()
n = int(len(w0)/2)
# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 50.0
numpoints = 1500

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

fig, axes = plt.subplots(nrows=n, ncols=2)
for i in range(0, n):
    plt.subplot(n, 2, 2*i+1)
    title = ["Osc", i,"alfa=", p[i][0],"mi=", p[i][1],"d=", p[i][2],"e=", p[i][3], "Coupled to:" ]
    if len(p[i]) > 4:
        coupl = list(zip(p[i][4::2], p[i][5::2]))
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
    plt.legend(prop={'size':50/n})
    plt.subplot(n, 2, 2*i+2)
    plt.title("Phase space")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(wsol[:,2*i], wsol[:,2*i+1])

fig.tight_layout()
plt.savefig("/home/kasia/Pulpit/inzynierka/wykres")
plt.show()
