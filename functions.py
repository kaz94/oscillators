from matplotlib import pyplot as plt
from scipy.signal import hilbert
import numpy as np
from scipy.signal import argrelmax
from ReadAdjacencyMatrix import read_file
from scipy.optimize import curve_fit
from itertools import product
from collections import Iterable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
          dy/dt = -alfa(x^2 - mi)y - f*x(x+d)(x+e) = 0
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
            eq += c[1] * (x[int(c[0])] + y[int(c[0])])
        equasions.append(eq)
    return equasions


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


def synchronization(phases, couplings_gl, N, M):
    quantif = []

    for i, coupl in enumerate(couplings_gl):
        osc_i_couplings = coupl[1]
        if len(osc_i_couplings) > 0:
            from_osc = osc_i_couplings[0][0]

            # coherence:
            drive = phases[int(from_osc)]
            driven = phases[i]
            q = coherence(drive, N, driven, M)
            quantif.append(q)

    return quantif


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


if __name__ == '__main__':
    x = np.loadtxt("phi3vdp0.5K.txt")
    N = int(x.shape[1] / 2)
    '''phi = x[:, :N]
    dphi = x[:, N:]'''

    '''phi = x[:, :2]
    dphi = x[:, 3:5]

    #qc1py = fourier_coefficients(phi[:, 0], phi[:, 1], dphi[:, 0], order=1)
    qc1py, q = fourier_coeff(phi, dphi, order=10)

    norm, omega = coeff_norm(qc1py)'''

    #print("qcoeff: ", qc1py)
    #print("q: ", q)

    # porownanie trzech funkcji liczacych protof->faza
    '''phi = x[:, 0]
    phase1 = protophase2phase(phi)
    phase2 = true_phases_ZLE(phi)
    phase3, dph3 = true_phases(phi)

    print("1:",phase1, phase1.shape)
    print("2:",phase2[0], phase2.shape)
    print("3:",phase3[0], phase3.shape)

    nnn=300
    plt.plot(phase1[:nnn], color="r", label="prom")
    plt.plot(phase2[0][:nnn]+7, color="g", label="moje")
    plt.plot(phase3[0][:nnn]+14, color="b", label="rosenblum")
    plt.legend()
    plt.show()'''


