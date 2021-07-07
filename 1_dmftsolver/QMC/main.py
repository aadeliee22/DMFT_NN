import numpy as np
from gfmod import *

# nyquist frequency
wn, Greal, Gimag = np.loadtxt('hello', unpack = True)
giwn = Greal + Gimag*1j
ntau = 10*len(wn)
beta = 100
gtau = gftools.GreenInverseFourier(giwn, ntau, beta, (1, 0, 0))

tau = np.linspace(0, beta, ntau)
np.savetxt('hello-tau', np.array([tau, gtau]).T)
"""
niwn = 60
ntau = 121
beta = 20
mu = 0
V = 1
D = 1
nmc = 3
nmeas = 100
Uarr = [1.0, 5.0]
seed = 0

giwn = gftools.SemiCircularGreen(niwn, beta, mu, V, D)
gtau = gftools.GreenInverseFourier(giwn, ntau, beta, (1, 0, 0))

import matplotlib.pyplot as plt
tau = np.linspace(0, beta, len(gtau))
for i, U in enumerate(Uarr):
    solver = qmc.hfqmc(gtau, beta, U, seed)
    gtau_up, err_up, gtau_dw, err_dw = solver.meas(nmc, nmeas)
    plt.errorbar(tau, (gtau_up + gtau_dw)/2.0, (err_up + err_dw)/2, marker = 'o',
      markersize = 5, linewidth = 1, capsize = 5, label = r'$U=%.2f$'%U)
plt.legend(loc = 'best', fontsize = 14, edgecolor = 'None', framealpha = 0.0)
plt.tick_params(which = 'both', direction = 'in', labelsize = 14)
plt.xlabel(r'$\tau$', fontsize = 14)
plt.ylabel(r'$G(\tau)$', fontsize = 14)
plt.show()
"""
