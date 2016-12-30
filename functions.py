from matplotlib import pyplot as plt
from scipy.signal import hilbert
import numpy as np
from scipy.signal import argrelmax
from ReadAdjacencyMatrix import read_file
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

        # passed frequency has to be calculated to parameter f:
        params['f'] = 29.3*params['f']**2
        couplings = p[o][5:]
        couplings = list(zip(couplings[0::2], couplings[1::2]))
        eq = -1 * params['alfa'] * (x[o] ** 2 - params['mi']) * \
             y[o] - params['f'] * x[o] * (x[o] + params['d']) * (x[o] + params['e'])
        for c in couplings:
            eq += c[1] * x[int(c[0])]
        equasions.append(eq)
    return equasions


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


def coherence(freq1, n, freq2, m):
    f1, f2 = np.copy(freq1), np.copy(freq2)
    f1 += np.pi
    f2 += np.pi
    f1 = np.unwrap(f1) % (2 * n * np.pi) / n
    f2 = np.unwrap(f2) % (2 * m * np.pi) / m

    phases_diff = f1-f2
    j_phases_diff = np.array([complex(imag=p) for p in phases_diff])
    return abs(np.sum(np.exp(j_phases_diff)/len(phases_diff)))


def frequency(series_maksima, t):
    return len(series_maksima[0]) / (t[len(t)-1] - t[0])


def plot_synchrograms(t, phases, couplings_gl, quantif, N, M, plot=True):
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
            peak_indexes = argrelmax(phases[int(from_osc)], order=5)

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
                plt.legend(prop={'size': 50 / len(couplings_gl)})

            '''freq_drive = frequency(argrelmax(phases[int(from_osc)], order=5), t)
            freq_driven = frequency(argrelmax(phases[int(i)], order=5), t)'''

            # coherence:
            drive = phases[int(from_osc)]
            driven = phases[i]
            q = coherence(drive, N, driven, M)
            quantif.append(q)

            # testy
            print(q)
            '''with open("test.dat", "w") as f:
                f.write(" ".join(map(str, t)) + "\n")
                f.write(" ".join(map(str, phases[int(from_osc)])) + "\n")
                f.write(" ".join(map(str, phases[i])) + "\n")'''


    if plot:
        plt.tight_layout(h_pad=1)
        plt.savefig("synchro")
        plt.show()


def freq_fit_function(x, a, b, c, d):
    return a * np.sqrt(b * x + c) + d


def f_fit_function(x, a, b, c):
    return a * x**b + c


def fit_f(t, phases, p):
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
    popt, pcov = curve_fit(f_fit_function, np.array(frequencies), np.array(f_list))
    fit = f_fit_function(np.array(frequencies), *popt)

    plt.plot(frequencies, f_list, 'ko', label="Original Data")
    plt.plot(frequencies, fit, 'r-', label="Fitted Curve")
    plt.text(0.01,40,"f = a * frequency^b + c")
    plt.text(0.01,38,str(popt))
    plt.ylabel("f - parameter")
    plt.xlabel("frequency")
    plt.legend()
    plt.savefig("przelicznik_f.png")
    plt.show()
    plt.plot(frequencies, np.sqrt(f_list), 'ko', label="Original Data")
    plt.ylabel("sqrt(f)")
    plt.xlabel("frequency")
    plt.legend()
    plt.savefig("przelicznik_f.png")

    plt.show()
    return popt


def fit_freq(t, phases, p):
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
    popt, pcov = curve_fit(freq_fit_function, np.array(f_list), np.array(frequencies))
    fit = freq_fit_function(np.array(f_list), *popt)

    plt.plot(f_list, frequencies, 'ko', label="Original Data")
    plt.plot(f_list, fit, 'r-', label="Fitted Curve")
    plt.text(1,1.3,"frequency = a * sqrt(b * f + c) + d")
    plt.text(1,1.1,str(popt))
    plt.xlabel("f - parameter")
    plt.ylabel("frequency")
    plt.legend()
    plt.savefig("przelicznik_freq.png")
    plt.show()
    return popt


