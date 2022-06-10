#!/usr/bin/env python3
import numpy as np
from gfmod import *
import matplotlib.pyplot as plt
import os
#os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=1
#os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=1
#os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=1
#os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=1
#os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=1

import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-U", help = "on-site interaction", type = float)
parser.add_argument("-beta", help = "1/temperature", type = float, default = 50)
parser.add_argument("-load", help = "previous data", type = str, default = None)
parser.add_argument("-way", help = "direction", type = int)
args = parser.parse_args()

def main():
    way = args.way
    beta = args.beta
    U = args.U

    niwn = 400
    ntau = 1000
    mu = 0
    #muu = U/2.0
    V = 1
    D = 1
    nmc = 3
    nmeas = 4
    seed = 0
    numLoop = 50
    delta_G = 5e-3

    # imaginary time & Matsubara frequency
    giwn = gftools.SemiCircularGreen(niwn, beta, mu, V, D)
    gtau = gftools.GreenInverseFourier(giwn, ntau, beta, (1,0,0))
    tau = np.linspace(0, beta, len(gtau))
    omega = np.array([1j*(2*n+1.)*np.pi/beta for n in range(len(giwn))], dtype = 'complex128')

    # HF-QMC solver
    distance = lambda G1, G2: np.average(np.abs(G1-G2))
    G_new = giwn.copy()
    if (args.load!=None):
        print ("---- LOAD DATA : ", args.load)
        w, Gr, Gi = np.loadtxt(args.load, unpack = True, dtype = 'float64')
        G_new = Gr + 1j*Gi

    print (f"---- START SELF-CONSISTENT LOOP: U = {U}.")
    for numiter in range(numLoop):
        G_old = G_new.copy()
        # Self-consistent equation G_bath = (iwn + mu - (D/2)^2*G)^-1
        G_bath = 1.0/(omega + mu - (D/2.0)**2*G_old)
        gtau = gftools.GreenInverseFourier(G_bath, ntau, beta, (1,0,0))

        # Solve
        solver = qmc.hfqmc(gtau, beta, U, seed)
        gtau_up, err_up, gtau_dw, err_dw = solver.meas(nmc, nmeas)
        gtau = (gtau_up + gtau_dw)/2.0
        G_new = gftools.GreenFourier(gtau, niwn, beta)

        np.savetxt(f'./Bethe_{way:.0f}_beta{beta:.0f}/Gw-{U:.2f}.dat', np.array([omega.imag, G_new.real, G_new.imag]).T)
        np.savetxt(f'./Bethe_{way:.0f}_beta{beta:.0f}/Gt-{U:.2f}.dat', np.array([tau, gtau, (err_up + err_dw)/2]).T)
        if (numiter>0):
            dist = distance(G_new, G_old)
            print(" * |G_new-G_old| : ", dist)
            if (dist < delta_G):
                print(">>>> CONVERGE")
                break
        sys.stdout.flush()

    #fig, ax = plt.subplots(1,2, figsize=(11,4))
    #ax[0].errorbar(tau, gtau, (err_up + err_dw)/2, marker = 'o',markersize = 3, linewidth = 1, capsize = 5)
    #ax[0].legend(loc = 'best', fontsize = 14, edgecolor = 'None', framealpha = 0.0)
    #ax[0].tick_params(which = 'both', direction = 'in', labelsize = 14)
    #ax[0].set_ylim(-0.55, 0.05)
    #ax[0].axhline(y=-0.5, linestyle='--', color='k', lw=1)
    #ax[0].set_xlabel(r'$\tau$', fontsize = 14)
    #ax[0].set_ylabel(r'$G(\tau)$', fontsize = 14)
    #ax[1].plot(omega.imag, G_new.imag, marker = '.', markersize = 5, linewidth=1)
    #ax[1].set_ylim(-1.5, 0.01)
    #plt.show()

main()
