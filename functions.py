from matplotlib import pyplot as plt
from scipy.signal import hilbert
import numpy as np
from scipy.signal import argrelmax
from ReadAdjacencyMatrix import read_file
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit


def load_params():
    w = []
    params = []
    couplings_gl = []
    adj_list = read_file()
    for i, osc in enumerate(adj_list):
        # Oscillator(x, y, alfa, mi, d, e, f)
        w.append(osc[0])  # x0
        w.append(osc[1])  # y0
        params.append(osc[2:])  # alfa, mi, d, e, f, coupling1, k1, coupling2, k2, ...
        couplings_gl.append([i, list(zip(osc[7::2], osc[8::2]))])
    return w, params, couplings_gl


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

        plt.subplot(n, 3, 3 * i + 2)
        plt.title("Phase space")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(wsol[:, 2 * i], wsol[:, 2 * i + 1])

        plt.subplot(n, 3, 3 * i + 3)

        plt.title("Phase")
        plt.xlabel("t")
        plt.ylabel("Phase")

        plt.plot(t, phases[i], label='phase')
        # plt.plot(t, instantaneous_phase, label='inst phase')
        plt.legend(prop={'size': 30 / n})

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


def get_phases(wsol, n):
    phases = []
    for i in range(0, n):
        analytic_signal = hilbert(wsol[:, 2 * i])
        # amplitude_envelope = np.abs(analytic_signal)
        # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
        phase = np.angle(analytic_signal)
        phases.append(phase)
    return phases


def coherence(phases1, phases2):
    phases_diff = phases1-phases2
    j_phases_diff = np.array([complex(imag=p) for p in phases_diff])
    return abs(np.sum(np.exp(j_phases_diff)/len(phases_diff)))


def frequency(series_maksima, t):
    return len(series_maksima[0]) / (t[len(t)-1] - t[0])


def plot_synchrograms(t, phases, couplings_gl, n, quantif, freq_drive, freq_driven, plot=True):

    t = np.asarray(t)
    if plot:
        subplots = 0
        for i, coupl in enumerate(couplings_gl):
            if len(coupl[1]) > 0:
                subplots += 1

        plot_params = {'figure.figsize': (10, 3*subplots)}
        plt.rcParams.update(plot_params)
        active_subplot = 0

    for i, coupl in enumerate(couplings_gl):
        osc_i_couplings = coupl[1]
        if len(osc_i_couplings) > 0:

            strength = osc_i_couplings[0][1]
            from_osc = osc_i_couplings[0][0]

            # find peaks in the drive signal
            peak_indexes = argrelmax(phases[int(from_osc)])

            drive = phases[int(from_osc)][peak_indexes]
            driven = phases[i][peak_indexes]

            if plot:
                active_subplot += 1
                plt.subplot(subplots, 1, active_subplot)

                plt.scatter(t[peak_indexes], drive, label=' '.join(["osc", str(int(from_osc))]))
                plt.plot(t, phases[int(from_osc)])

                plt.scatter(t[peak_indexes], driven, label=' '.join(["osc", str(i)]), color="red")
                plt.plot(t, phases[i])

                plt.title(' '.join(["Synchrogram, k=", str(strength)]))
                plt.xlabel("t")
                plt.ylabel("Phase")
                plt.legend(prop={'size': 50 / n})

            # coherence:

            f_drive = frequency(argrelmax(phases[int(from_osc)]), t)
            f_driven = frequency(argrelmax(phases[int(i)]), t)
            freq_drive.append(f_drive)
            freq_driven.append(f_driven)

            ratio = f_drive / f_driven

            # 1:1
            q11 = coherence(drive, driven)


            quantif.append(q11)

    if plot:
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


def plot_hist_synchronization(k_range, freq_ratio, freq_drive, freq_driven, p, quantif):
    s_pos = []
    k_axis, freq_axis = np.meshgrid(k_range, freq_ratio)
    k_axis = k_axis.flatten()
    freq_axis = freq_axis.flatten()
    for k in k_range:
        p[1][6] = k
        for f_r in freq_ratio:
            s_pos.append(0)
            p[1][4] = f_r * p[0][4]

    f_pos = (np.array(freq_drive) / np.array(freq_driven)).tolist()
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.bar3d(k_axis, f_pos, s_pos, 0.07, 0.01, quantif, alpha=0.4)
    ax.set_xlabel('k')
    ax.set_ylabel('f1:f2')
    ax.set_zlabel('synchronization')
    plt.ylim([0, 1.5])
    plt.savefig("/home/kasia/Pulpit/inzynierka/quantification_hist.png")
    plt.show(block=False)


def plot_trisurf_synchronization(k_range, freq_ratio, freq_drive, freq_driven, p, quantif):
    k_axis = []
    for k in k_range:
        p[1][6] = k
        for f_r in freq_ratio:
            p[1][4] = f_r * p[0][4]
            k_axis.append(k)

    f_axis = (np.array(freq_drive) / np.array(freq_driven)).tolist()

    plot_params = {'figure.figsize': (8, 6)}
    plt.rcParams.update(plot_params)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(k_axis, f_axis, quantif, cmap=cm.jet)
    ax.set_xlabel('k')
    ax.set_ylabel('f1:f2')
    ax.set_zlabel('synchronization')
    plt.ylim([0, 1.5])
    plt.savefig("/home/kasia/Pulpit/inzynierka/coherence.png")
    plt.show(block=False)


def function(x, a, b, c, d):
    return a * np.log(b * x + c) + d


def fit_params(t, phases, p):
    # f and frequency ratio
    frequencies = []
    f_list = []
    t = np.array(t)
    tt = t[100: len(t) - 100]
    for i, phase in enumerate(phases):
        phase = np.array(phase[100:len(phase) - 100])
        f_list.append(p[i][4])
        peaks_1 = argrelmax(phase)[0]
        t_period = tt[peaks_1[len(peaks_1) - 1]] - tt[peaks_1[0]]
        frequencies.append((len(phase[peaks_1]) - 1) / t_period)

    popt, pcov = curve_fit(function, np.array(f_list), np.array(frequencies))
    print("dopasowanie:", popt)
    plt.plot(f_list, frequencies, 'ko', label="Original Data")
    fit = function(np.array(f_list), *popt)
    plt.plot(f_list, fit, 'r-', label="Fitted Curve")
    plt.text(1,1.3,"frequency = a * log(b * f + c) + d")
    plt.text(1,1.2,str(popt))
    plt.xlabel("f - parameter")
    plt.ylabel("frequency")
    plt.savefig("/home/kasia/Pulpit/inzynierka/przelicznik_dop.png")
    plt.show()

#np.diff(faza)/delta t
