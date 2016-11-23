from matplotlib import pyplot as plt
from scipy.signal import hilbert
import numpy as np
from scipy.signal import argrelmax
from ReadAdjacencyMatrix import read_file


def load_params(couplings_gl):
    w = []
    params = []
    adj_list = read_file()
    for i, osc in enumerate(adj_list):
        # Oscillator(x, y, alfa, mi, d, e)
        w.append(osc[0])  # x0
        w.append(osc[1])  # y0
        params.append(osc[2:])  # alfa, mi, d, e, coupling1, k1, coupl2, k2, ...
        #if len(list(zip(osc[7::2], osc[8::2]))) > 0:
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


def timeSeries(t, wsol, n, p, phases):
    fig, axes = plt.subplots(nrows=n, ncols=3)
    for i in range(0, n):
        plt.subplot(n, 3, 3 * i + 1)
        title = ["Osc", i, "alfa=", p[i][0], "mi=", p[i][1], "d=", p[i][2], "e=", p[i][3], "f=", p[i][4], "Coupled to:"]
        if len(p[i]) > 5:
            coupl = list(zip(p[i][5::2], p[i][6::2]))
            # connections.append((i, int(p[i][5])))
            for c in coupl:
                title.append("osc")
                title.append(int(c[0]))
                title.append("k:")
                title.append(c[1])
        plt.title(' '.join(str(t) for t in title))
        plt.plot(t, wsol[:, 2 * i], label='x')
        plt.plot(t, wsol[:, 2 * i + 1], label='y')
        plt.xlabel('t')
        plt.grid(True)
        plt.legend(prop={'size': 30 / n})

        analytic_signal = hilbert(wsol[:, 2 * i])
        # amplitude_envelope = np.abs(analytic_signal)
        # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
        phase = np.angle(analytic_signal)

        plt.subplot(n, 3, 3 * i + 2)
        plt.title("Phase space")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(wsol[:, 2 * i], wsol[:, 2 * i + 1])

        plt.subplot(n, 3, 3 * i + 3)

        plt.title("Phase")
        plt.xlabel("t")
        plt.ylabel("Phase")

        plt.plot(t, phase, label='phase')
        # plt.plot(t, instantaneous_phase, label='inst phase')
        plt.legend(prop={'size': 30 / n})

        phases.append(phase)

    fig.tight_layout()
    plt.savefig("/home/kasia/Pulpit/inzynierka/wykres")
    plt.show()
'''
    conn_no_dupl = connections.copy()
    for i, c1 in enumerate(conn_no_dupl):
        for j, c2 in enumerate(conn_no_dupl):
            if c1 == c2 or (c1[0] == c2[1] and c1[1] == c2[0]):
                conn_no_dupl.remove(conn_no_dupl[i])
    return conn_no_dupl
'''


def plot_synchrograms(t, phases, couplings_gl, n):

    t = np.asarray(t)
    # t = t[50:]
    # for i, p in enumerate(phases):
    #    phases[i] = p[50:]  # delete first few points

    i_subplots = 0
    for i, coupl in enumerate(couplings_gl):
        if len(coupl[1]) > 0:
            i_subplots += 1

    plot_params = {'figure.figsize': (10, 3*i_subplots)}
    plt.rcParams.update(plot_params)

    idx = 0
    print(couplings_gl)
    for i, coupl in enumerate(couplings_gl):
        if len(coupl[1]) > 0:

            c = coupl[1][0][0]
            print(coupl)
            print(coupl[1])

            idx += 1
            plt.subplot(i_subplots, 1, idx)

            # for c in coupl[1]:
            peakind = argrelmax(phases[int(c)])
            plt.scatter(t[peakind], phases[int(c)][peakind], label=' '.join(["osc",str(int(c))]))
            # peakind = signal.find_peaks_cwt(phases[i], np.arange(1, 10))
            plt.scatter(t[peakind], phases[i][peakind], label=' '.join(["osc",str(i)]), color="red")
            plt.plot(t, phases[i])
            plt.plot(t, phases[int(c)])

            plt.title(' '.join(["Synchrogram, k=", str(coupl[1][0][1])]))
            plt.xlabel("t")
            plt.ylabel("Phase")
            plt.legend(prop={'size': 50 / n})


    plt.tight_layout(h_pad=1)
    plt.savefig("/home/kasia/Pulpit/inzynierka/synchro")
    plt.show()


def save_p_s(n, wsol, p):
    for i in range(0, n):
        title = ["Phase space, ", i, "alfa=", p[i][0], "mi=", p[i][1], "d=", p[i][2], "e=", p[i][3], "f=", p[i][4],
                 "Coupled to:"]
        plt.plot(wsol[:, 2 * i], wsol[:, 2 * i + 1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(prop={'size': 30 / n})
        plt.savefig("".join(["/home/kasia/Pulpit/inzynierka/mi", str(i)]))
        plt.clf()


def quantification(phases1, phases2):
    phases_diff = phases1-phases2
    j_phases_diff = np.array([complex(imag=p) for p in phases_diff])
    return abs(np.sum(np.exp(j_phases_diff)/len(phases_diff)))

