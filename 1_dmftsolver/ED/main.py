#!/usr/bin/env python3
import sys
import numpy as np
import modules.ed_solver as ed
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-U", help = "on-site interaction", type = float)
parser.add_argument("-beta", help = "inverse temperature", type = float, default = 100)
parser.add_argument("-load", help = "file name of previous data to restart a DMFT loop", type = str, default = None)
args = parser.parse_args()

def main():
    # handling parameter
    beta = args.beta
    niwn = 5000
    nbath = 5
	# 2.3 ~ 2.9 phase coexistence region (metal & insulator)
    # U > 2.9 : insulator; U < 2.3 : metal
    U = args.U
    mu = U/2.0
    D = 1.0
    delta_G = 1e-4
    numLoop = 100
    prefix = './1to4_0.001/'
    # Matsubara frequency
    omega = np.array([1j*(2*n+1.)*np.pi/beta for n in range(niwn)], dtype = 'complex128')
    # ED solver
    solver = ed.solver(nbath = nbath, niwn = niwn, beta = beta)
    # distance measure between two Green's functions
    distance = lambda G1, G2 : np.sum(np.abs(G1 - G2))
    G_new = np.zeros_like(omega)
    G_old = np.zeros_like(omega)
    if (args.load!=None):
        print ("load data from ", args.load)
        w, Gr, Gi = np.loadtxt(args.load, unpack = True, dtype = 'float64')
        G_new = Gr + 1j*Gi

    print ("--- START A SELF-CONSISTENT LOOP!")
    for numIter in range(numLoop):
        print ("ITERATION: ", numIter+1)
        G_old = G_new.copy()
        # self-consistent equation (Bethe lattice): BathGreen = (iwn + mu - (D/2)^2*G)^-1
        BathGreen = 1.0/(omega + mu -(D/2.0)**2*G_old)
        G_new, density, double_occ = solver.solve({'U' : U, 'mu' : mu}, BathGreen)
        print (" - <N>: ", density)
        print (" - <N_up*N_dw>: ", double_occ)
        # save current states
        np.savetxt(prefix + f'checkpoint-{U:.2f}', solver.get_pvec().reshape([1, -1]))
        with open(prefix + f'result-{U:.2f}', 'w') as f:
            f.write('DENSITY: %f\n'%density)
            f.write('DOUBLE-OCC: %f\n'%double_occ)
            f.write('SUM-RULE: %f\n'%np.sum(solver.get_pvec()[nbath:]**2))
            f.flush()
        if (numIter > 0):
            dist = distance(G_new, G_old)
            print (" * |G_NEW - G_OLD| : ", dist)
            if (dist < delta_G):
                print (">>> CONVERGE!")
                break
        print ("\n")
        sys.stdout.flush()
    np.savetxt(prefix + f'Giw-{U:.2f}.dat', np.array([omega.imag, G_new.real, G_new.imag]).T)


if __name__ == "__main__":
    main()
