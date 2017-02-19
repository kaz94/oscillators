from functions import *


def get_pphases(t, wsol, cut):
    phases = []
    n = int(wsol.shape[1] / 2)
    for i in range(0, n):
        analytic_signal = hilbert(wsol[:, 2 * i])
        # amplitude_envelope = np.abs(analytic_signal)
        # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
        phase = np.angle(analytic_signal)
        '''plt.plot(phase)
        plt.savefig(str(i)+"phase.png")
        plt.show()'''
        phases.append(phase)


    # cut start and end points with error due to hilbert transform
    start = cut
    end = len(t) - cut
    t = t[start:end]
    phases = np.array(phases)
    phases = phases[:, start:end]
    wsol = wsol[start:end, :]

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


def true_phases_moje(protophases):
    phases = []
    if not isinstance(protophases[0], Iterable):
        protophases = np.array([protophases])
    for protoph in protophases:
        protoph = np.array(protoph)
        N = len(protoph)
        n_range = np.arange(-10,10)
        n_range = np.delete(n_range, 10) # remove 0
        S =  1./N *np.array([ np.sum(np.exp(-1j*n*protoph)) for n in n_range])
        phases.append(protoph + \
                 np.array([np.sum(S/(1j*n_range) * np.exp(1j*n_range*p-1.)) for p in protoph]))


    return np.real(np.array(phases))


def true_phases(t, x, theta):
    phi = []
    ph_diff = []
    if len(theta.shape) == 1:
        theta = np.array([theta])
    for theta_ in theta:
        theta_ = np.array(theta_)
        nfft = 10
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
        ph_diff.append(np.append(np.diff(phi_) / np.diff(t), 0))

    t, x, phi, ph_dif = out_remv(t, x, np.array(phi), np.array(ph_diff))

    return t, x, phi, ph_dif


def protophase2phase(theta, order=10):
    ph = theta.astype(np.complex)
    for n in range(-order, order + 1):
        if n == 0:
            continue
        Sn = np.mean(np.exp(-1j * n * theta), axis=0)
        ph += (Sn / (1j * n)) * (np.exp(1j * n * theta) - 1)

    return np.real(ph)


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
        print(out)

    ph = np.delete(ph, out, axis=0)
    dph = np.delete(dph, out, axis=0)
    t = np.delete(t, out)
    x = np.delete(x, out, axis=0)

    return t, x, ph, dph


def natural_freq(time, proto_phases):
    proto_phases = np.unwrap(proto_phases)
    freq = (proto_phases[-1] - proto_phases[0])/(time[-1] - time[0])
    return freq # omega_0

# Ta funkcja działa OK!!!!!
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
                in_idx1 = in_idx1[0] * ncf1 + in_idx1[1] #* ncf1 + in_idx1[2]
                in_idx2 = in_idx2[0] * ncf1 + in_idx2[1] #* ncf1 + in_idx2[2]
            if N == 3:
                in_idx1 = in_idx1[0] * ncf2 + in_idx1[1] * ncf1 + in_idx1[2]
                in_idx2 = in_idx2[0] * ncf2 + in_idx2[1] * ncf1 + in_idx2[2]
            out_idx = tuple((nmk_ - rsq_) + ncf)
            C[in_idx1, in_idx2] = A[out_idx]

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
                    for ii, qc in enumerate(qcoeff):
                        qc[n, m, k] = coeff[ii][idx]
            elif N == 2:
                idx = n * ncf1 + m
                for ii, qc in enumerate(qcoeff):
                    qc[n, m] = coeff[ii][idx]
    return np.array(qcoeff)


def q_fourier(qcoeff, order=10):
    N = qcoeff.shape[0]
    ngrid = 5
    Y, X = np.meshgrid(2 * np.pi * (np.arange(0, ngrid) - 1) / (ngrid - 1),
                       2 * np.pi * (np.arange(0, ngrid) - 1) / (ngrid - 1))
    q = np.zeros([N, ngrid, ngrid], dtype=complex)
    for n in range(-order, order + 1):
        for m in range(-order, order + 1):
            tmp = np.exp(1j * n * X + 1j * m * Y)
            for i_q, q_ in enumerate(q):
                # TODO sumować ten wymiar czy dodać kolejny też do q?
                # dodac !
                # np.exp(1j * n * X + 1j * m * Y + 1j * k * Z)
                if isinstance(qcoeff[i_q][n + order, m + order], Iterable):
                    for qcoeff_n in qcoeff[i_q][n + order, m + order]:
                        q[i_q] = q[i_q] + qcoeff_n * tmp
                else:
                    q[i_q] = q[i_q] + qcoeff[i_q][n + order, m + order] * tmp
    q = np.real(q)
    #print(q)

    '''fig = plt.figure()
    for ii, q_ in enumerate(q):
        ax = fig.add_subplot(1, len(q), ii + 1, projection='3d')
        surf = ax.plot_surface(X, Y, q_, cmap=cm.coolwarm)
        ax.set_zlabel("q")
        fig.colorbar(surf)
    plt.show()'''


'''
def fourier_coefficients(p1, p2, dphi1, order=10):
    N_F = order
    N_F1 = N_F + 1
    A = np.zeros([4 * N_F + 1, 4 * N_F + 1], dtype=np.complex)

    or2 = 2 * N_F
    or21 = or2 + 1

    for n in range(-or2, or21):
        for m in range(-or2, n + 1):
            A[n + or2, m + or2] = np.mean(np.exp(1j * (n * p1 + m * p2)))
            A[-n + or2, -m + or2] = A[n + or2, m + or2].conjugate()

    B1 = np.zeros([or21**2], dtype=np.complex)
    C = np.zeros([or21**2, or21**2], dtype=np.complex)

    ind = 0
    for n in range(-N_F, N_F1):
        i1_1 = (n + N_F) * or21
        for m in range(-N_F, N_F1):
            i1 = i1_1 + m + N_F
            i4 = m + or2
            tmp = np.exp(-1j * (n * p1 + m * p2))
            B1[ind] = np.mean(dphi1 * tmp)
            ind += 1
            for r in range(-N_F, N_F1):
                i3 = (r + N_F) * or21 + N_F
                i2 = (n - r) + or2
                for s in range(-N_F, N_F1):
                    C[i1, i3 + s] = A[i2, i4 - s]

    qc1 = np.dot(np.linalg.inv(C.conjugate()), B1)

    return qc1
'''


def coeff_norm(qcoeff):
    for qc in qcoeff:
        S = np.array(qc).shape
        N = int((S[0] - 1.) / 2.)
        omega = np.real(qc[N, N])
        qc[N, N] = 0.
        nrmq = np.sqrt(np.trapz(np.trapz(np.abs(qcoeff) ** 2.)))

    return nrmq, omega


def estimate_coeff(phi, dphi, order=5):
    """
    Chooses the optimum coefficients and fits the coupled
    oscillator model to data.

    Coefficients are first estimated by integrating the angular frequency
    with the complex conjugate of exponentials with given n,m,k's.

    TODO: Then the
    coefficients are sorted and the ones with the largest power are
    summed until the ratio of its sum to that of all
    the coefficients is met.

    Argumen
    phi    = phase of oscillator 1, 2, and 3
    dphi   = phase velocity for oscillator 1, 2, and 3
    order  = order of the Fourier series


    Output
        qcoef1 = coefficients of the Fourier series for oscillator 1
        qcoef2 = coefficients of the Fourier series for oscillator 2
        qcoef3 = coefficients of the Fourier series for oscillator 3

    ..note:: dla dużych N ~ 10 to trwa wieczność :(
    """

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
    print(A.shape)
    nmk = list(range(-ncf, ncf1))

    for nmk_ in product(*([nmk, ] * N)):
        nmk_ = np.array(nmk_)

        idx1 = tuple(nmk_ + ncf)
        idx2 = tuple(-nmk_ + ncf)

        A[idx1] = np.mean(np.exp(1j * np.sum(nmk_ * phi, axis=1)))
        A[idx2] = np.conj(A[idx1])

    B = np.zeros((ncf1**N, N), dtype=np.complex)
    C = np.zeros((ncf1**N, ncf1**N), dtype=np.complex)

    # TODO: poprawić bo
    # dalej zadziała tylko dla 3 oscylatorów
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
            in_idx1 = in_idx1[0] * ncf2 + in_idx1[1] * ncf1 + in_idx1[2]
            in_idx2 = (nmk_ + order)
            in_idx2 = in_idx2[0] * ncf2 + in_idx2[1] * ncf1 + in_idx2[2]
            out_idx = tuple((nmk_ - rsq_) + ncf)
            C[in_idx1, in_idx2] = A[out_idx]

    coeff1 = np.dot(np.linalg.inv(C), B[:, 0])
    coeff2 = np.dot(np.linalg.inv(C), B[:, 1])
    coeff3 = np.dot(np.linalg.inv(C), B[:, 2])

    qcoeff1 = np.zeros((ncf1, ncf1, ncf1), dtype=np.complex)
    qcoeff2 = np.zeros((ncf1, ncf1, ncf1), dtype=np.complex)
    qcoeff3 = np.zeros((ncf1, ncf1, ncf1), dtype=np.complex)

    for n in range(ncf1):
        for m in range(ncf1):
            for k in range(ncf1):
                idx = n * ncf2 + m * ncf1 + k
                qcoeff1[n, m, k] = coeff1[idx]
                qcoeff2[m, k, n] = coeff2[idx]
                qcoeff3[k, n, m] = coeff3[idx]

    return qcoeff1, qcoeff2, qcoeff3

def fourier_coefficients(p1, p2, dphi1, order=10):
    N_F = order
    N_F1 = N_F + 1
    A = np.zeros([4 * N_F + 1, 4 * N_F + 1], dtype=np.complex)

    or2 = 2 * N_F
    or21 = or2 + 1

    for n in range(-or2, or21):
        for m in range(-or2, n + 1):
            A[n + or2, m + or2] = np.mean(np.exp(1j * (n * p1 + m * p2)))
            A[-n + or2, -m + or2] = A[n + or2, m + or2].conjugate()

    B = np.zeros([or21**2], dtype=np.complex)
    C = np.zeros([or21**2, or21**2], dtype=np.complex)

    ind = 0
    for n in range(-N_F, N_F1):
        i1_1 = (n + N_F) * or21
        for m in range(-N_F, N_F1):
            i1 = i1_1 + m + N_F
            i4 = m + or2
            tmp = np.exp(-1j * (n * p1 + m * p2))
            B[ind] = np.mean(dphi1 * tmp)
            ind += 1
            for r in range(-N_F, N_F1):
                i3 = (r + N_F) * or21 + N_F
                i2 = (n - r) + or2
                for s in range(-N_F, N_F1):
                    C[i1, i3 + s] = A[i2, i4 - s]

    qc1 = np.dot(np.linalg.inv(C.conjugate()), B)
    return qc1


if __name__ == '__main__':
    '''data = np.loadtxt('signal.txt')
    t = data[:, 0]
    wsol = data[:, 1:]

    point_dens = int(len(t) / (t[-1] - t[0]))

    # protophases
    t, wsol, pphases = get_pphases(t, wsol, cut=point_dens * 10)

    # true phases reconstructed
    t, wsol, ph, dph = true_phases(t, wsol, pphases)'''

    #ph, dph = np.transpose(ph), np.transpose(dph)
    #np.savetxt('true.txt', np.hstack((np.transpose([t]), ph, dph)), fmt='%.18g',
    #           delimiter=' ', newline='\n')

    #qcoeff = fourier_coeff(ph, dph, order=2)
    #print(qcoeff.shape)
    #print(np.real(qcoeff[0]))
    #for i, qcoeff_ in enumerate(qcoeff):
        #np.savetxt("qcoeff"+str(i), qcoeff_, fmt='%.18g', delimiter=' ', newline='\n')

    #q = q_fourier(qcoeff)
    #print(qcoeff.shape)


    x = np.loadtxt("phi3vdp0.5K.txt")
    N = int(x.shape[1] / 2)
    phi = x[:, :N]
    dphi = x[:, N:]

    qc1py = fourier_coeff(phi,dphi, 5) # OK!!!
    print(qc1py)

    '''a = np.array([[1, 2],[9,4]])
    b = np.array([1,2])
    c=np.dot(np.linalg.inv(a), b)
    print(c)'''



