import numpy as np
from itertools import product
from scipy.signal import hilbert
from scipy.signal import argrelmax
from matplotlib import pyplot as plt


def get_pphases(t, wsol, cut):

    wsol1=wsol[:, 0::2]
    analytic_signal = hilbert(np.transpose(wsol1))

    # cut unstable points due to Hilbert transform
    start = cut
    end = len(t) - cut
    analytic_signal = analytic_signal[:, start:end]
    wsol = wsol[start:end, :]
    t = t[start:end]

    analytic_signal = analytic_signal - np.transpose([np.mean(analytic_signal, axis=1)])
    phases = np.angle(analytic_signal)
    phases = np.mod(phases, 2*np.pi)

    return t, wsol, phases


def poincare_protophases(t, wsol, phases, p):
    x=wsol[:,0]
    y=wsol[:,1]
    t=np.array(t)
    poincare_x_idx = [i for i, j in enumerate(x) if j > (x.max() - x.min()) *1.2/3.]
    poincare_y_idx = argrelmax(-np.abs(y), order=5)[0]
    poincare_points_idx = [i for i in poincare_x_idx if i in poincare_y_idx]
    '''plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.scatter(t[poincare_y_idx], y[poincare_y_idx])
    plt.show()

    plt.plot(x, y, label="trajektoria")
    plt.scatter(x[poincare_points_idx], y[poincare_points_idx], label="punkty Poincare", color="r")
    plt.xlabel("x", size="large")
    plt.ylabel("y", size="large")
    plt.legend()
    plt.title("Przekrój Poincare", size="large")
    plt.savefig("poincare.png")
    plt.show()'''

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

    poincare_protoph_wra = np.array(poincare_protoph)%(2.*np.pi) - np.pi
    true_ph_poincare, noth = true_phases(np.array(poincare_protoph))
    true_ph_hilbert, noth = true_phases(get_pphases(wsol, 1)[0])

    omega_0 = []
    for ph in phases:
        omega_0.append(natural_freq(t, ph))

    idx_poincare = range(poincare_points_idx[0], poincare_points_idx[-1])
    t_poincare = t[idx_poincare]
    plt.plot(t, np.unwrap(phases[0])-omega_0[0]*(t-t[0]), label="protofaza - Hilbert")
    plt.plot(t_poincare, poincare_protoph-omega_0[0]*(t_poincare-t_poincare[0]), label="protofaza - Poincare")
    plt.plot(t, np.unwrap(true_ph_hilbert)[0]-omega_0[0]*(t-t[0]), label="faza - Hilbert")
    plt.plot(t_poincare, np.unwrap(true_ph_poincare[0])-omega_0[0]*(t_poincare-t_poincare[0]), label="faza Poincare")

    '''plt.plot(t, phases[0], linestyle="--", label="hilbert")
    plt.plot(t[poincare_points_idx[0]: poincare_points_idx[-1]], poincare_protoph_wra, linestyle="-.", color="r", label="poincare")
    plt.plot(t[poincare_points_idx[0]: poincare_points_idx[-1]], true_ph_poincare[0], color="g", label="poincare->true")
    plt.plot(t, true_ph_hilbert[0], color="black", label="hilbert->true")'''
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("proto(phase)")
    plt.show()



    return poincare_protoph_wra


def natural_freq(time, proto_phases):
    r, c = proto_phases.shape
    ax = 1
    if r > c:
        ax = 0
    proto_phases = np.unwrap(proto_phases, axis=ax)
    freq = (proto_phases[-1] - proto_phases[0])/(time[-1] - time[0])
    return freq # omega_0


def out_remv(t, x, ph, dph, nstd=5):
    n, N = ph.shape
    m, N2 = dph.shape
    if n != m:
        raise Exception("arrays have different size of items")
    if N != N2:
        raise Exception("Not consistent number of oscillators")

    out = np.array([], dtype=np.int)
    for i in range(N):
        # finding the indices of the outlying samples
        out = np.append(out, np.where(np.isnan(ph[:, i]))[0])
        out = np.append(out, np.where(np.isnan(dph[:, i]))[0])
        outInds = np.where(np.logical_or(
            dph[:, i] < np.mean(dph[:, i]) - nstd * np.std(dph[:, i]),
            dph[:, i] > np.mean(dph[:, i]) + nstd * np.std(dph[:, i])))

        out = np.append(out, outInds)

    if len(out) != 0:
        print("removed !")
        out = np.array(list(set(out)))
        print(out)

    ph = np.delete(ph, out, axis=0)
    dph = np.delete(dph, out, axis=0)
    t = np.delete(t, out)
    x = np.delete(x, out, axis=0)

    return t, x, np.transpose(ph), np.transpose(dph), out


def true_phases(theta):
    if len(theta.shape) == 1:
        N = 1
        n_points = len(theta)
    else:
        N, n_points = theta.shape
    if N > 3:
        raise Exception("Number of osc > 3, maybe transpose theta array?")
    phi = []

    if len(theta.shape) == 1:
        theta = np.array([theta])
    for theta_ in theta:
        theta_ = np.array(theta_)
        nfft = 100
        Spl = np.zeros(nfft, dtype=complex)
        Hl = np.zeros(nfft)

        tmp = np.diff(np.mod(theta_, 2 * np.pi))
        IN = [i for i, j in enumerate(tmp) if j < 0]  # poprawka na pełne okresy
        npt = len(theta_[IN[0]: IN[-1]])

        S = 0
        c = float(npt + 2) / float(npt)
        for k in range(nfft):
            Spl[k] = np.sum(np.exp(-1j * (k+1) * theta_[IN[0]: IN[-1]+1])) / (npt+1)
            S = S + Spl[k] * Spl[k].conjugate() - 1. / float(npt+1)
            Hl[k] = np.real((k+1) / (npt+1) - c * S)
        indopt = np.argmin(Hl)

        phi_ = np.copy(theta_)
        for k in range(indopt+1):
            phi_ = phi_ + 2. * np.imag(Spl[k] * (np.exp(1j * (k+1) * theta_)-1) / (k+1))
        phi.append(phi_)

    return np.array(phi)


def phi_dot(phi, fs):
    norder = 5  # order of the fitting polynomial
    sl = 12  # window semi-length
    wl = 2 * sl + 1  # window length
    g = np.loadtxt("IO/golay_coeff.txt")
    phi = np.unwrap(phi, axis=1)

    dphi = []
    for ph_ in phi:
        dphi.append(np.convolve(ph_, g[:, 1], "same"))
    dphi = np.array(dphi)

    dphi = dphi[:, sl:(dphi.shape[1] - sl)]*fs
    phi = phi[:, sl:phi.shape[1]-sl]

    return dphi, phi


def max_tri_sync(theta, order=5):
    m_sync_in = np.zeros((order, order, order))
    maxind = 0
    for n in range(0,order+1):
        for m in range(-order, order+1):
            for l in range(-order, order+1):
                index = np.abs(np.mean(np.exp(1j * (n*theta[0] + m*theta[1] + l*theta[2]) )))
                if (index > maxind) and (n != 0 or m != 0 or l != 0):
                    maxind = index
                    n_th1 = n
                    m_th2 = m
                    l_th3 = l
                # m_sync_in[n, m+order, l+order] = index

    return m_sync_in, maxind, n_th1, m_th2, l_th3


def fourier_coeff(phi, dphi, order=10):

    nn, N = phi.shape
    if N != dphi.shape[1]:
        raise Exception("Not consistent number of oscillators")
    if N > 3:
        raise Exception("Number of oscillators is greater than 3")

    order1 = order + 1
    ncf = 2 * order
    ncf1 = ncf + 1
    ncf2 = ncf1 * ncf1  # number of coefficients in each dimension

    phi = np.unwrap(phi, axis=0)
    # This matrix contains the coefficients
    # A[n + p, m + q, k + r]
    # for the linear system of equations to
    # obtain the coefficients Qcoef[n, m, k]
    A = np.zeros(([4 * order + 1] * N), dtype=np.complex)
    nmk = list(range(-ncf, ncf1))

    for nmk_ in product(*([nmk, ] * N)):
        nmk_ = np.array(nmk_)

        idx1 = tuple(nmk_ + ncf)
        idx2 = tuple(-nmk_ + ncf)

        A[idx1] = np.mean(np.exp(1j * np.sum(nmk_ * phi, axis=1)))
        A[idx2] = np.conj(A[idx1])

    B = np.zeros((ncf1 ** N, N), dtype=np.complex)
    C = np.zeros((ncf1 ** N, ncf1 ** N), dtype=np.complex)

    idx = 0
    rsq = list(range(-order, order1))
    for rsq_ in product(*([rsq, ] * N)):
        rsq_ = np.array(rsq_)
        exp = np.exp(-1j * np.sum(rsq_ * phi, axis=1)).reshape(nn, 1)
        B[idx, :] = np.mean(dphi * exp, axis=0)
        del exp
        idx += 1
        for nmk_ in product(*([rsq, ] * N)):
            nmk_ = np.array(nmk_)
            in_idx1 = (rsq_ + order)
            in_idx2 = (nmk_ + order)
            if N == 2:
                in_idx1 = in_idx1[0] * ncf1 + in_idx1[1]
                in_idx2 = in_idx2[0] * ncf1 + in_idx2[1]
            if N == 3:
                in_idx1 = in_idx1[0] * ncf2 + in_idx1[1] * ncf1 + in_idx1[2]
                in_idx2 = in_idx2[0] * ncf2 + in_idx2[1] * ncf1 + in_idx2[2]
            out_idx = tuple((nmk_ - rsq_) + ncf)
            C[in_idx1, in_idx2] = A[out_idx]

    print("wyznacznik:", np.linalg.det(C))

    coeff = []
    qcoeff = []

    for i in range(N):
        coeff.append(np.dot(np.linalg.inv(C), B[:, i]))
        qcoeff.append(np.zeros((ncf1,)*N, dtype=np.complex))
    coeff = np.array(coeff)

    for n in range(ncf1):
        for m in range(ncf1):
            if N == 3:
                for k in range(ncf1):
                    idx = n * ncf2 + m * ncf1 + k
                    qcoeff[0][k, n, m] = coeff[0][idx]
                    qcoeff[1][n, m, k] = coeff[1][idx]
                    qcoeff[2][m, k, n] = coeff[2][idx]
            elif N == 2:
                idx = n * ncf1 + m
                qcoeff[0][n, m] = coeff[0][idx]
                qcoeff[1][m, n] = coeff[1][idx]

    qcoeff = np.array(qcoeff)

    return coeff, qcoeff


def co_3to2(qcoef_, N, thresh):

    thresh = np.max(np.abs(qcoef_)) * thresh / 100.
    ind = np.abs(qcoef_) < thresh
    qcoef_[ind] = 0
    or1 = N + 1
    or21 = 2 * N + 1
    qc12 = np.zeros((or21, or21), dtype=np.complex)
    qc13 = np.zeros((or21, or21), dtype=np.complex)
    C123 = 0.
    part12 = 0.
    part13 = 0.
    tot = 0.

    for n in range(0, or21):
        for m in range(0, or21):
            qc12[n, m] = qcoef_[or1 - 1, n, m]
            qc13[n, m] = qcoef_[m, n, or1 - 1]
            for k in range(0, or21):
                qcoef_pow2_abs = np.abs(qcoef_[k, n, m] ** 2)
                if ((m+1) != or1) and ((k+1) != or1):
                    C123 += qcoef_pow2_abs
                part12 += ((m+1) - or1)**2 * qcoef_pow2_abs
                part13 += ((k+1) - or1)**2 * qcoef_pow2_abs
                tot += qcoef_pow2_abs

    C12 = np.sqrt(np.sum(np.sum(np.abs(qc12) ** 2)))
    C13 = np.sqrt(np.sum(np.sum(np.abs(qc13) ** 2)))
    C123 = np.sqrt(C123)
    tot = np.sqrt(tot)
    part12 = np.sqrt(part12)
    part13 = np.sqrt(part13)

    return C12, C13, C123, part12, part13, tot


def q_norms(qcoeff):

    ## method = 1
    thresh = 2
    N = qcoeff[0].shape
    N = int((N[0] - 1) / 2)

    COUP = np.zeros((3, 3))
    NORM = np.zeros(3)

    omega = np.zeros(len(qcoeff))

    for n in range(len(qcoeff)):
        omega[n] = np.abs(qcoeff[n][(N,)*len(qcoeff)])
        qcoeff[n][(N,)*len(qcoeff)] = 0.

    COUP[0, 1], COUP[0, 2], COUP[0, 0], _, _, NORM[0] = co_3to2(qcoeff[0], N, thresh)
    COUP[1, 2], COUP[1, 0], COUP[1, 1], _, _, NORM[1] = co_3to2(qcoeff[1], N, thresh)
    COUP[2, 0], COUP[2, 1], COUP[2, 2], _, _, NORM[2] = co_3to2(qcoeff[2], N, thresh)

    return COUP, NORM, omega


if __name__ == '__main__':
    data = np.loadtxt('IO/signal.txt')
    t = data[:, 0]
    wsol = data[:, 1:]
    fs = 1 / (t[1] - t[0])

    X = wsol[:, 0::2]
    Y = wsol[:, 1::2]
    N = X.shape[1]

    # normal
    theta = -np.arctan2(Y, X)

    # noisy
    # theta = -np.arctan2(Y - 0.2 + np.random.uniform(-0.05, 0.05), X - 0.2 + np.random.uniform(-0.05, 0.05))
    # np.savetxt('IO/theta.txt', theta)

    omega = natural_freq(t, theta)
    print(omega)

    ph = true_phases(theta.T)


    '''# check maximal synchronization:
    m_sync_in, maxind, n_th1, m_th2, l_th3 = max_tri_sync(ph, order=5)
    print("maxind: ", maxind, "\t\t", n_th1, m_th2, l_th3)

    dphi, phi = phi_dot(ph, fs)

    cf, qcf = fourier_coeff(phi.T, dphi.T, order=3)

    q, n, o = q_norms(qcf)
    print('Coupling:\n', q, '\nnorm:', n, '\nomega:', o)'''

    # theta and phi comparison, single oscillator
    '''plt.plot(t, np.unwrap(theta[:, 0]) - omega*(t-t[0]), linestyle='--', label=r"$\theta$")
    plt.plot(t, np.unwrap(ph[0]) - omega*(t-t[0]), label=r"$\phi$")
    plt.xlabel("czas", fontsize=20)
    plt.ylabel(r"$\phi, \theta - \omega_0 t$", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("IO/phi_theta.png")
    plt.show()'''

    # wrapped
    '''theta_mod = theta # np.mod(theta, 2*np.pi)
    phi_mod = phi

    # unwrapped
    theta = np.unwrap(theta)
    phi = np.unwrap(phi)

    end = 200
    N=1
    for osc in range(N):
        plt.plot(t[:end], theta[:end, osc], color="r", label="unwr")
        plt.plot(t[:end], theta_mod[:end, osc], color="g", label="wrapped")
    plt.title("theta")
    plt.legend()
    plt.show()


    sl = 12
    for osc in range(N):
        plt.plot(t[:end], theta[:end, osc], color="r", label="th_unwr")
        plt.plot(t[sl:len(t)-sl][:end], phi[osc, :end], color="g", label="phi_unwr")
    plt.legend()
    plt.show()

    for osc in range(N):
        plt.plot(t[:end], theta[:end, osc] - o[osc]*t[:end], color="r", label="th_unwr - omega*t")
        plt.plot(t[sl:len(t)-sl][:end], phi[osc, :end] - o[osc]*t[sl:len(t)-sl][:end], color="g", label="phi_unwr - omega*t")
    plt.legend()
    plt.show()

    omega = theta[-1, :] - theta[0, :]
    for osc in range(N):
        plt.plot(t[:end], theta[:end, osc] - omega[osc]*t[:end], color="r", label="th_unwr - omega2*t")
        plt.plot(t[sl:len(t)-sl][:end], phi[osc, :end] - omega[osc]*t[sl:len(t)-sl][:end], color="g", label="phi_unwr - omega2*t")
    plt.legend()
    plt.show()'''



