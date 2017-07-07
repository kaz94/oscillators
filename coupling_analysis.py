import numpy as np
from itertools import product


def true_phases(theta):
    N, n_points = theta.shape
    if N > 3:
        print("N: ", N)
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
    g = np.loadtxt("golay_coeff.txt")
    phi = np.unwrap(phi, axis=1)

    dphi = []
    for ph_ in phi:
        dphi.append(np.convolve(ph_, g[:, 1], "same"))
    dphi = np.array(dphi)

    dphi = dphi[:, sl:(dphi.shape[1] - sl)]*fs
    phi = phi[:, sl:phi.shape[1]-sl]

    return dphi, phi


def fourier_coeff(phi, dphi, order=10):

    nn, N = phi.shape
    if N != dphi.shape[1]:
        raise Exception("Not consistent number of oscillators")
    if N > 3:
        print(phi.shape)
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
    print(N)
    N = int((N[0] - 1) / 2)
    print(N)

    COUP = np.zeros((3, 3))
    NORM = np.zeros(3)

    omega = np.zeros(len(qcoeff))

    for n in range(len(qcoeff)):
        omega[n] = np.abs(qcoeff[n][N, N, N])
        qcoeff[n][N,N,N] = 0.

    COUP[0, 1], COUP[0, 2], COUP[0, 0], _, _, NORM[0] = co_3to2(qcoeff[0], N, thresh)
    COUP[1, 2], COUP[1, 0], COUP[1, 1], _, _, NORM[1] = co_3to2(qcoeff[1], N, thresh)
    COUP[2, 0], COUP[2, 1], COUP[2, 2], _, _, NORM[2] = co_3to2(qcoeff[2], N, thresh)

    return COUP, NORM, omega


if __name__ == '__main__':
    data = np.loadtxt('signal.txt')
    t = data[:, 0]
    wsol = data[:, 1:]
    fs = 1 / (t[1] - t[0])

    X = wsol[:, 0::2]
    Y = wsol[:, 1::2]
    N = X.shape[1]

    theta = -np.arctan2(Y, X)

    ph = true_phases(theta.T)

    dphi, phi = phi_dot(ph, fs)

    cf, qcf = fourier_coeff(phi.T, dphi.T, order=5)


    #q_norms(qcf)

    '''Q = np.loadtxt("/home/kasia/PycharmProjects/oscillators/Qtest.dat")
    Q = Q.reshape((20, 20, 20))
    Q = Q.transpose((0, 2, 1))
    #print(Q)
    #print(co_3to2(Q, 2, 1.1))

    qc = np.array([Q, Q, Q])'''

    q, n, o = q_norms(qcf)
    print('Coupling:\n', q, '\nnorm:', n, '\nomega:', o)




