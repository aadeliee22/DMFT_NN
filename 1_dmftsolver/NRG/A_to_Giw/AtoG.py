import numpy as np
from scipy.integrate import simps, trapz
from . import _cpp_module
#import matplotlib.pyplot as plt

D = 1.
T = 0.01
beta = int(1./T)
N = 1000
if T == 0.001:
    add_dir = '_0.001'
    U_c1, U_c2 = 2.2, 2.59
    U = np.array([1.50, 2.00, 2.20, 2.40, 2.58, 2.59, 2.60, 2.80, 3.00, 3.50])
elif T == 0.01:
    add_dir = ''
    U_c1, U_c2 = 2.2, 2.37
    U = np.array([1.50, 1.80, 2.00, 2.20, 2.36, 2.37, 2.40, 2.70, 3.00, 3.50])

directory = ''
w_len = len(np.loadtxt(f'.{directory}/1to4{add_dir}/Bethe-{2.00:.2f}_solution.dat', \
                       unpack = True, dtype = 'complex128')[0])
x = np.zeros((len(U), w_len), dtype = 'float64')
omega = np.pi/beta * (2*np.arange(N+1)+1) # w_n
tau = np.linspace(0, beta, N*5)

def G(x):
    N = len(omega)
    return np.array([simps(x/(1j*omega[i]-w.real), w.real) for i in range (N)])

G_omega = np.zeros((len(x), len(omega)), dtype = 'complex128')
G_tau = np.zeros((len(x), len(tau)), dtype = 'complex128')
for i, u in enumerate(U):
    w, A_w, G_w, S_w = np.loadtxt(f'.{directory}/1to4{add_dir}/Bethe-{u:.2f}_solution.dat', \
                                  unpack = True, dtype = 'complex128')
    x[i] = A_w.real.copy()
    G_omega[i] = G(x[i])
    G_tau[i] = GreenIF(G_omega, N*5, beta, (1,0,0))

np.savetxt(f'Gw.dat', np.concatenate([[omega.T], G_array]).T, header = f\"w   G_w\")
np.savetxt(f'Gt.dat', np.concatenate([[tau.T], G_tau]).T, header = f\"t   G_t\")
