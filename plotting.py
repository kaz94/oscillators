from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.signal import argrelmax
import numpy as np

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


def plot_synchrograms(t, phases, couplings_gl):
    t = np.asarray(t)
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
            plot_params = {'figure.figsize': (12, 10),
                           'axes.labelsize': 'large',
                           'axes.titlesize': 'large',
                           'xtick.labelsize': 'large',
                           'ytick.labelsize': 'large'}
            plt.rcParams.update(plot_params)
            strength = osc_i_couplings[0][1]
            from_osc = osc_i_couplings[0][0]

            # find peaks in the drive signal
            peak_indexes = argrelmax(phases[int(from_osc)], order=5)

            drive = phases[int(from_osc)][peak_indexes]
            driven = phases[i][peak_indexes]

            active_subplot += 1
            plt.subplot(subplots, 1, active_subplot)

            #plt.scatter(t[peak_indexes], drive, label=' '.join([" f=1.0"]))
            #plt.plot(t, phases[int(from_osc)])

            plt.scatter(t[peak_indexes], driven, color="red")
            #plt.plot(t, phases[i])

            plt.title(' '.join(["k=", str(strength), " delta=0.2"]))
            plt.ylim([-4,3])
            plt.xlabel("t")
            plt.ylabel("$\phi$")
            plt.legend(prop={'size': 50 / len(couplings_gl)})

    plt.tight_layout(h_pad=1)
    plt.savefig("synchro")
    plt.show()


def arnold_tongue(N, M):
    data = np.loadtxt("out"+str(N)+"_"+str(M)+".dat")
    k_ax = data[:, 0]
    delta_ax = data[:, 1]
    quantif = data[:, 2:]
    len_k = len(k_ax[k_ax == k_ax[0]])

    len_delta = int(len(k_ax) / len_k)
    for col in range(quantif.shape[1]):
        plot_map(k_ax, delta_ax, quantif[:, col], len_k, len_delta)


def plot_map(k, delta, quantif, len_k, len_delta):
    qqqq = quantif
    #plt.clf()
    k_diff=np.diff(k)
    delta_diff=np.diff(delta)
    dx = k_diff[k_diff!=0][0]
    dy = delta_diff[delta_diff!=0][0]
    #delta_axis, k_axis = np.meshgrid(delta_range, k_range)
    #delta_axis = delta_range.reshape()
    quantif = quantif.reshape(len_delta, len_k)
    delta_axis = np.array(delta).reshape(len_k, len_delta)
    k_axis = np.array(k).reshape(len_delta, len_k)
    quantif = quantif[:-1, :-1]
    levels = MaxNLocator(nbins=20).tick_values(quantif.min(), quantif.max())
    cmap = plt.get_cmap('PiYG')

    plt.contourf(delta_axis[:-1, :-1] + dx / 2.,
                      k_axis[:-1, :-1] + dy / 2., quantif, levels=levels,cmap=cmap)
    plt.title('Mapa synchronizacji')
    plt.xlabel("$\Delta$")
    plt.ylabel("k")
    plt.tight_layout()

    #plt.savefig("coherence.png")
    #plt.show(block=False)

    '''fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(k, delta, qqqq, cmap=cm.jet)
    ax.set_xlabel('k')
    ax.set_ylabel('delta')
    ax.set_zlabel('synchronization')
    plt.savefig("coherence.png")
    plt.show(block=False)'''


def plot_hist_synchronization(k_range, freq_ratio, p, quantif):
    s_pos = []
    k_axis, freq_axis = np.meshgrid(k_range, freq_ratio)
    k_axis = k_axis.flatten()
    freq_axis = freq_axis.flatten()
    for k in k_range:
        p[1][6] = k
        for f_r in freq_ratio:
            s_pos.append(0)
            p[1][4] = f_r * p[0][4]

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.bar3d(k_axis, freq_axis, s_pos, 0.07, 0.01, quantif, alpha=0.4)
    ax.set_xlabel('k')
    ax.set_ylabel('f1:f2')
    ax.set_zlabel('synchronization')
    #plt.ylim([0, 1.5])
    plt.savefig("quantification_hist.png")
    plt.show(block=False)


def plot_trisurf_synchronization(k_axis, f_axis, quantif):
    plot_params = {'figure.figsize': (8, 6)}
    plt.rcParams.update(plot_params)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(k_axis, f_axis, quantif, cmap=cm.jet)
    ax.set_xlabel('k')
    ax.set_ylabel('delta')
    ax.set_zlabel('synchronization')
    plt.savefig("coherence.png")
    plt.show(block=False)


def save_p_s(n, wsol, p):
    for i in range(0, n):
        title = ["Phase space, ", i, "alfa=", p[i][0], "mi=", p[i][1], "d=", p[i][2], "e=", p[i][3], "f=", p[i][4],
                 "Coupled to:"]
        plt.plot(wsol[:, 2 * i], wsol[:, 2 * i + 1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(prop={'size': 30 / n})
        plt.savefig("".join(["mi", str(i)]))
        plt.clf()