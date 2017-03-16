from functions import *


def get_pphases(t, wsol, cut):

    wsol1=wsol[:, 0::2]
    analytic_signal = hilbert(np.transpose(wsol1))

    # cut unstable points due to Hilbert transform
    start = cut
    end = len(t) - cut
    analytic_signal = analytic_signal[:, start:end]
    wsol1 = wsol1[start:end, :]
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

    t, x, phi, ph_dif, out = out_remv(t, x, np.array(phi).T, np.array(ph_diff).T)

    return t, x, phi, ph_dif, out


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
        out = np.array(list(set(out)))
        print(out)

    ph = np.delete(ph, out, axis=0)
    dph = np.delete(dph, out, axis=0)
    t = np.delete(t, out)
    x = np.delete(x, out, axis=0)

    return t, x, np.transpose(ph), np.transpose(dph), out


def natural_freq(time, proto_phases):
    proto_phases = np.unwrap(proto_phases)
    freq = (proto_phases[-1] - proto_phases[0])/(time[-1] - time[0])
    return freq # omega_0


if __name__ == '__main__':
    data = np.loadtxt('signal.txt')
    t = data[:, 0]
    wsol = data[:, 1:]

    point_dens = int(len(t) / (t[-1] - t[0]))

    # protophases
    t, wsol, pphases = get_pphases(t, wsol, cut=point_dens * 10)

    # true phases reconstructed
    t, wsol, ph, dph, out = true_phases(t, wsol, pphases)


    pphases = np.transpose(pphases)
    np.savetxt('pphases.txt', pphases, fmt='%.18g', #np.transpose([t]),
               delimiter=' ', newline='\n')
    ph, dph = np.transpose(ph), np.transpose(dph)
    np.savetxt('phases.txt', np.hstack((ph, dph)), fmt='%.18g', #np.transpose([t]),
               delimiter=' ', newline='\n')

    ph_mat = np.loadtxt('matlab/phi_mat_moje.txt')
    ph_mat = np.delete(ph_mat, out, axis=0)
    print(np.allclose(ph, ph_mat, atol=0.05))

