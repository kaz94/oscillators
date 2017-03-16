import numpy as np
from itertools import product
from collections import Iterable


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
    C_mat = np.loadtxt("matlab/C.txt")

    print("wyznacznik:", np.linalg.det(C))
    print("C vs C_mat:", np.allclose(np.real(C), C_mat))

    coeff = []
    coeff2=[]
    qcoeff = []

    for i in range(N):
        coeff.append(np.dot(np.linalg.inv(C), B[:, i]))
        coeff2.append(np.linalg.lstsq(C, B[:, i])[0])
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

    return coeff, np.array(qcoeff), np.array(coeff2)


def q_fourier(qcoeff, order=10):
    N = qcoeff.shape[0]
    ngrid = 5
    Y, X = np.meshgrid(2 * np.pi * (np.arange(0, ngrid) - 1) / (ngrid - 1),
                       2 * np.pi * (np.arange(0, ngrid) - 1) / (ngrid - 1))
    q = np.zeros([N, ngrid, ngrid], dtype=complex)
    for n in range(-order, order + 1):
        for m in range(-order, order + 1):
            if N == 2:
                tmp = np.exp(1j * n * X + 1j * m * Y)
                for i_q, q_ in enumerate(q):
                    # TODO dodac kolejny wymiar
                    if isinstance(qcoeff[i_q][n + order, m + order], Iterable):
                        for qcoeff_n in qcoeff[i_q][n + order, m + order]:
                            q[i_q] = q[i_q] + qcoeff_n * tmp
                    else:
                        q[i_q] = q[i_q] + qcoeff[i_q][n + order, m + order] * tmp

            '''elif N == 3:
                for k in range(-order, order + 1):
                    tmp = np.exp(1j * n * X + 1j * m * Y + 1j * k * Z)'''
    q = np.real(q)
    #print(q)

    '''fig = plt.figure()
    for ii, q_ in enumerate(q):
        ax = fig.add_subplot(1, len(q), ii + 1, projection='3d')
        surf = ax.plot_surface(X, Y, q_, cmap=cm.coolwarm)
        ax.set_zlabel("q")
        fig.colorbar(surf)
    plt.show()'''


def coeff_norm(qcoeff):
    for qc in qcoeff:
        S = np.array(qc).shape
        N = int((S[0] - 1.) / 2.)
        omega = np.real(qc[N, N])
        qc[N, N] = 0.
        nrmq = np.sqrt(np.trapz(np.trapz(np.abs(qcoeff) ** 2.)))

    return nrmq, omega


if __name__ == '__main__':

    '''data = np.loadtxt('phi3vdp0.5K.txt')
    points, N = data.shape
    ph = data[:,:int(N/2)]
    dph = data[:,int(N/2):]
    coeff, qcoeff = fourier_coeff(ph, dph, order=2)

    cf_mat = np.loadtxt("matlab/coeff_real_mat.txt")
    #print(coeff.shape)
    print("coeff mat=py: ", np.allclose(cf_mat, np.real(np.transpose(coeff))))
    #print(np.real(qcoeff))'''


    data2 = np.loadtxt('phases.txt')
    points, N = data2.shape
    ph2 = data2[:,:int(N/2)]
    dph2 = data2[:,int(N/2):]
    coeff2, qcoeff2, coeff_lstsq = fourier_coeff(ph2, dph2, order=2)

    cf_mat2 = np.loadtxt("matlab/coeff_real_mat2.txt")
    print("coeff mat=py (moje dane): ", np.allclose(cf_mat2, np.real(np.transpose(coeff2))))
    #print(np.real(np.transpose(coeff2)))
    print(np.transpose(coeff_lstsq))
    print(np.transpose(coeff2))
    print(np.allclose(cf_mat2, np.real(np.transpose(coeff_lstsq)), rtol=0.5))








    #for i, qcoeff_ in enumerate(qcoeff):
        #np.savetxt("qcoeff"+str(i), qcoeff_, fmt='%.18g', delimiter=' ', newline='\n')

    #q = q_fourier(qcoeff)

    # qcoeff
    '''x = np.loadtxt("phi3vdp0.5K.txt")
    N = int(x.shape[1] / 2)
    phi = x[:, :N]
    dphi = x[:, N:]

    qc1py = fourier_coeff(phi,dphi, 5)
    print(qc1py)'''



