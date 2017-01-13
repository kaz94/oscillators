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
          dy/dt2 = -alfa(x^2 - mi)y - f*x(x+d)(x+e) = 0
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


def poincare_protophases(t,wsol, phases, p):
    x=wsol[:,0]
    y=wsol[:,1]
    t=np.array(t)
    poincare_x_idx = [i for i, j in enumerate(x) if j > (x.max() - x.min()) *1.2/3.]
    poincare_y_idx = argrelmax(-np.abs(y), order=5)[0]
    plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.scatter(t[poincare_y_idx], y[poincare_y_idx])
    plt.show()
    poincare_points_idx = [i for i in poincare_x_idx if i in poincare_y_idx]
    plt.scatter(x[poincare_points_idx], y[poincare_points_idx], color="r")
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    L = 0.
    L_i = 0.
    poincare_protoph = []
    for i, point in enumerate(poincare_points_idx[:-1]):
        start = poincare_points_idx[i]
        stop = poincare_points_idx[i+1]+1
        trajectory_parts = np.sqrt(np.diff(x[start:stop])**2 + np.diff(y[start:stop])**2)
        delta_L_i = np.sum(trajectory_parts)

        L += np.cumsum(trajectory_parts)
        poincare_protoph += list(2.*np.pi*(L - L_i)/delta_L_i + 2.*np.pi*i)

        L_i += delta_L_i
        L = L[-1]

    plt.plot(t, phases[0], linestyle="--", label="hilbert")
    poincare_protoph = np.array(poincare_protoph)%(2.*np.pi) - np.pi

    plt.plot(t[poincare_points_idx[0]: poincare_points_idx[-1]], poincare_protoph, linestyle="-.", color="r", label="poincare")

    true_ph_poincare = true_phases(poincare_protoph)
    plt.plot(t[poincare_points_idx[0]: poincare_points_idx[-1]], true_ph_poincare, color="g", label="poincare->true")
    true_ph_hilbert = true_phases(get_phases(wsol,1)[0])
    plt.plot(t, true_ph_hilbert, color="black", label="hilbert->true")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("proto(phase)")
    #plt.savefig("hilbert_poincare_k0.05.png")
    plt.show()

    return poincare_protoph


def true_phases(protophases):
    protophases = np.array(protophases)
    N = len(protophases)
    n_range = np.arange(-10,10)
    n_range = np.delete(n_range, 10) # remove 0
    S =  1./N *np.array([ np.sum(np.exp(-1j*n*protophases)) for n in n_range])
    phases = protophases + np.array([np.sum(S/(1j*n_range) * np.exp(1j*n_range*p-1.)) for p in protophases])

    return phases.real


'''def matlab_proto(y, NV):

    x = []
    S = y.shape
    if S[0] > S[1]:
        y = np.transpose(y)
    y[0,:] = y[0,:] / np.std(y[0,:]);
    y[1,:] = y[1,:] / np.std(y[1,:]);
    Pro = np.zeros(len(y))
    Se = np.zeros(len(y));
    dd = np.zeros(len(y));
    theta = np.zeros(len(y));

    for n in range(1, len(y)+1):
        Pro[n] = np.transpose(NV)*y[:,n]

    IN = 1;
    for n = 2:length(Pro);
    if ((Pro(n) > 0) & & (Pro(n - 1) < 0)) ; % Intersection with Poincare plane
    Se(n) = 1;
    V(IN) = Pro(n) / (Pro(n) - Pro(n - 1));
    IN = IN + 1;
    else
    Se(n) = 0;
    end;
    end;
    dy = gradient(y); % Computing
    the
    covered
    distance in the
    state
    space
    for n= 1: length(y);
    dd(n) = norm(dy(:, n));
    end;

    Dis = cumsum(dd); % Covered
    distance
    along
    the
    trajectory

    Pmin = find(Se == 1); % Indices
    of
    the
    beginning
    of
    the
    cycles

    for i= 1: length(Pmin) - 1; %
    Computing
    protophase
    theta
    for j= Pmin(i): Pmin(i + 1) - 1;
    R1 = V(i) * (Dis(Pmin(i)) - Dis(Pmin(i) - 1));
    R2 = (1 - V(i + 1)) * (Dis(Pmin(i + 1)) - Dis(Pmin(i + 1) - 1));
    theta(j) = 2 * pi * (Dis(j) - (Dis(Pmin(i)) - R1)) / ((Dis(Pmin(i + 1) - 1) + R2) - (Dis(Pmin(i)) - R1));
    end;
    end;
    theta = unwrap(theta);
    Start = Pmin(1);
    Stop = Pmin(end) - 1;
    end'''




