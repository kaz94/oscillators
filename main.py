from scipy.integrate import odeint
from plotting import *
from functions import *
from coupling_analysis import get_pphases

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 250
point_dens = 40 # points per time unit
numpoints = int(point_dens * stoptime)
filename = 'macierz.txt'


def synchrograms_from_file():
    w0, p, couplings_gl = load_params(file='IO/'+filename) # load params from file
    n = int(len(w0)/2)
    q = dynamics(n, w0, p, couplings_gl, N=1, M=1, plot=True, save=True)
    print("synchro:", q)


def dynamics(n, w0, p, couplings_gl, N=1, M=1, plot=True, save=False):
    t = np.array([stoptime * float(i) / (numpoints - 1) for i in range(numpoints)])

    # Call the ODE solver.
    wsol, dict = odeint(vector_field, w0, t, args=(p,),
                  atol=abserr, rtol=relerr, full_output=True)
    # ttt=np.round(dict['hu'], 3)
    # print(np.round(list(set(ttt)), 3))

    # cut unstable points
    t = t[point_dens * 221:]
    wsol = wsol[point_dens * 221:, :]

    plot_params = {'figure.figsize': (12, 10),
                   'axes.labelsize': 'medium',
                   'axes.titlesize': 'medium',
                   'xtick.labelsize': 'xx-small',
                   'ytick.labelsize': 'xx-small'}
    plt.rcParams.update(plot_params)

    t, wsol, pphases = get_pphases(t, wsol, cut=point_dens * 10)

    if save:
        t = np.transpose([t])
        np.savetxt('IO/signal_tmp.txt', np.hstack((t,wsol)), fmt='%.18g', delimiter=' ', newline='\n')



    '''from estimate_phase import out_remv
    diff = np.diff(phases) / np.diff(t)
    phases = phases[:,:-1]
    t=t[:-1]
    ph, dph, out = out_remv(np.unwrap(phases), diff, nstd=5)
    plt.plot(t, phases[0], label="faza")
    plt.plot(t, diff[0], color="r", label="pochodna")
    #print(out.shape, ph.shape)
    #print(out)
    #plt.plot(t[out[0]], ph[out][0], color="g")
    #plt.plot(t[out[0]], dph[out][0], color="b")
    plt.legend()
    plt.savefig("ph_dph.png")
    plt.show()'''

    # protophases = poincare_protophases(t, wsol, phases, p)
    #t_phases, t_phases_diff = true_phases(phases)

    # phases vs protophases plot
    '''phases_unwr = np.unwrap(phases)
    natural_fr0 = natural_freq(t, phases[0])
    natural_fr1 = natural_freq(t, phases[1])
    #natural_fr2 = natural_freq(t, phases[2])

    plot_params = {'figure.figsize': (12, 10),
                   'axes.labelsize': 'large'}
    plt.rcParams.update(plot_params)

    plt.plot(t, phases_unwr[0]-natural_fr0*(t-t[0]), label="protofaza", color="red")
    plt.plot(t, np.unwrap(t_phases[0])-natural_fr0*(t-t[0]), label="faza", color="green")
    #plt.plot(t, phases_unwr[1], color="red")
    #plt.plot(t, np.unwrap(t_phases[1]), color="green")
    plt.plot(t, phases_unwr[1]-natural_fr1*(t-t[0]), color="red")
    plt.plot(t, np.unwrap(t_phases[1])-natural_fr1*(t-t[0]), color="green")
    #plt.plot(t, phases_unwr[2]-natural_fr2*(t-t[0])-1.5, label="protofaza f=0.2")
    #plt.plot(t, np.unwrap(t_phases[2])-natural_fr2*(t-t[0])-1.5, label="faza f=0.2")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel(r"$\phi - \omega_0$")
    plt.savefig("proto(phases).png")
    plt.show()'''

    #qcoeff1, qcoeff2 = fourier_coeff(phases, ph_diff)

    if plot:
        timeSeries(t, wsol, n, p, pphases)

    # synchrogramy
    any_coupling = False
    for c in couplings_gl:
        if len(c[1]) > 0:
            any_coupling = True
    if any_coupling:
        if plot:
            plot_synchrograms(t, pphases, couplings_gl)
        qq = synchronization(pphases, couplings_gl, N, M)
        return qq


def automate(N, M, file='IO/'+filename):
    w0, p, couplings_gl = load_params(file)
    '''couplings_gl = []
    #     x,   y
    w0 = [0.1, 0.1,
          0.1, 0.1]
    #    alfa, mi,  d,    e,   freq, osc, k
    p = [[1.0, 1.0, 1.0, -1.0, 1.],
         [1.0, 1.0, 1.0, -1.0, 1., 0.0, 0.0]]

    for i, osc in enumerate(p):
        couplings_gl.append([i, list(zip(osc[5::2], osc[6::2]))])'''

    n = int(len(w0) / 2)
    quantif = []
    print("couplings: ", couplings_gl)
    points = 20
    k_range = np.linspace(0., 1.0, points)
    delta_range = np.linspace(-0.15, 0.15, points)
    print("k:   f:  q:")
    with open("out"+str(N)+"_"+str(M)+".dat", 'w') as f:
        # N - drive (0), M - driven (1)
        base_freq = 0.2
        ii = 0
        p[0][4] = N*base_freq
        for k in k_range:
            p[1][6] = k
            for delta in delta_range:
                # freq = 0.2*sqrt(f)
                p[1][4] = M*base_freq + delta
                qq = dynamics(n, w0, p, couplings_gl, N, M, plot=False)
                quantif += qq
                print(k, p[1][4], qq)
                f.write(" ".join(map(str, [k, p[1][4], *qq])) + "\n")
                #print(k, p[1][4], quantif[ii])
                ii += 1

if __name__ == '__main__':

    synchrograms_from_file()
    '''automate(1, 3)
    automate(1, 2)
    automate(1, 1)
    plot_params = {'figure.figsize': (12, 10),
                   'axes.labelsize': 'large',
                   'axes.titlesize': 'large',
                   'xtick.labelsize': 'large',
                   'ytick.labelsize': 'large'}
    plt.rcParams.update(plot_params)
    arnold_tongue(1, 3)
    arnold_tongue(1, 2)
    arnold_tongue(1, 1)
    plt.colorbar()
    plt.savefig("IO/123arnold.png")
    plt.show()'''





'''(d^2x/dt^2 )+a*((x)^2-b)*(dx(t)/dt)+x*(x(t)+d)*(x(t)+e)*(1/(d*e))=0'''