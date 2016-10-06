from matplotlib import pyplot as plt

def vectorfield(w, t, p):
    """
    Defines diff. equasions
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2,...]
        t :  time
        p :  vector of the parameters:
                p = [alfa1, mi1, d1, e1, k1, alfa2, mi2, d2, e2, k2, ...]
                alfa:
                mi:
                y: initial derrivative at x
                d: -saddle
                e: -node
                Right hand side of the differential equations
                  dx/dt = y
                  dy/dt2 = -alfa(x^2 - mi)d1 - x(x+d)(x+e) / de = 0
    """
    x1, y1, x2, y2 = w
    alfa1, mi1, d1, e1, alfa2, mi2, d2, e2, k = p

    # Create f = (x1',y1',x2',y2'):
    f = [y1,
         -alfa1 * (x1 ** 2 - mi1) * y1 - x1 * (x1 + d1) * (x1 + e1) / (d1 * e1),
         y2,
         -alfa2 * (x2 ** 2 - mi2) * y2 - x2 * (x2 + d2) * (x2 + e2) / (d2* e2) + k * x1]
    return f


from scipy.integrate import odeint


# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
x1 = 0.1
y1 = 0.1
x2 = 0.1
y2 = 0.1

alfa1 = 0.1
mi1 = 1.
d1 = 1.
e1 = 1.
alfa2 = 0.1
mi2 = 1.
d2 = 1.
e2 = 1.
k = 0.1

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 100.0
numpoints = 2500

t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
p = [alfa1, mi1, d1, e1, alfa2, mi2, d2, e2, k]
w0 = [x1, y1, x2, y2]

# Call the ODE solver.
wsol = odeint(vectorfield, w0, t, args=(p,),
              atol=abserr, rtol=relerr)


plt.subplot(2, 1, 1)
plt.plot(t, wsol[:,0], label='x1')
plt.plot(t, wsol[:,1], label='y1')
plt.plot(t, wsol[:,2], label='x2')
plt.plot(t, wsol[:,3], label='y2')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.subplot(2, 1, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(wsol[:,0], wsol[:,1], label='osc 1')
plt.plot(wsol[:,2], wsol[:,3], label='osc 2')
plt.legend()
plt.show()


