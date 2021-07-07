#!/usr/bin/python3
import argparse
from triqs.utility import mpi
import numpy as np
from nrgljubljana_interface import Solver, SemiCircular
import utility
from scipy import optimize
from copy import deepcopy

#================= Parameters =================
parser = argparse.ArgumentParser()
parser.add_argument("-U", help = "on-site interaction", type = float)
parser.add_argument("-T", help = "temperature", type = float, default = 1e-4)
parser.add_argument("-D", help = "half bandwidth", type = float, default = 1)
parser.add_argument("-alp", help = "alpha", type = float, default = 0.8)
parser.add_argument("-lamb", help = "lambda", type = float, default = 1.6)
parser.add_argument("-keepE", help = "keep energy", type = float, default = 10.0)
parser.add_argument("-niter", help = "# of iterations", type = int, default = 100)
parser.add_argument("-tol", help = "tolerance of the convergence criterion", type = float, default = 1e-3)
parser.add_argument("-load", help = "file name of previous data to restart a DMFT loop", type = str, default = None)
args = parser.parse_args()
D = args.D
U = args.U
T = args.T
alpha = args.alp
keepE = args.keepE
niter = args.niter
tol = args.tol
lamb = args.lamb
filename = f'./lamb{lamb:.1f}_14/Bethe-{U:.2f}.dat'
#==============================================
# particle-hole symmetric case
mu = U/2
e_f = -U/2.0
# Set up the Impurity Solver
imp_solver = Solver(model = "SIAM", symtype = "QS", mesh_max = 5.0, mesh_min = 1e-5, mesh_ratio = 1.01)
#imp_solver.set_verbosity(False)
# Solve Parameters
sp = { "T": 1e-4, "Lambda": lamb, "Nz": 4, "Tmin": 1e-6, \
        "keep": 3000, "keepenergy": keepE, "alpha": 0.4, "bandrescale": 1.0 }
# Model Parameters
mp = { "U1": U, "eps1": e_f }
sp["model_parameters"] = mp
# Set up the Lattice Solver
lattice_solver = utility.BetheLatticeSolver(imp_solver, D)
# Load previous data
lattice_solver.load_data(imp_solver, filename if args.load is None else args.load)
# real frequency domain
realFreq = np.array([w for w in imp_solver.Delta_w.mesh], dtype = 'complex128')
G_old = np.zeros_like(realFreq)
G_new = np.zeros_like(realFreq)
Delta_in = imp_solver.Delta_w.copy()

# Start with a DMFT self-consistent loop
for n in range(niter):
    if mpi.is_master_node():
        print ('# OF ITERATIONS:', (n+1))
    # Solve the impurity model
    imp_solver.Delta_w << Delta_in
    imp_solver.solve(**sp)
    G_new = imp_solver.G_w['imp'].data.reshape([-1])
    # Save results
    if mpi.is_master_node():
        np.savetxt(filename, np.array([realFreq.real, \
            imp_solver.A_w['imp'].data.reshape([-1]).real, \
            imp_solver.G_w['imp'].data.reshape([-1]).real, \
            imp_solver.G_w['imp'].data.reshape([-1]).imag, \
            imp_solver.Sigma_w['imp'].data.reshape([-1]).real, \
            imp_solver.Sigma_w['imp'].data.reshape([-1]).imag ]).T, \
            header = f"w   A_w   G_w   Sigma_w  (D: {D}, T: {T}, U: {U}, half-filling)")
    diff = np.mean(np.abs(G_new-G_old))
    if n > 0:
        if mpi.is_master_node():
            print ("|G_NEW - G_OLD|: %.7f"%diff)
        if (diff < tol):
            if mpi.is_master_node():
                print ("CONVERGE!")
            break
        else:
            if mpi.is_master_node():
                print ('\n')
    else:
        if mpi.is_master_node():
            print ('\n')
    G_old = G_new.copy()
    # Solve self-consistent equation
    lattice_solver.update_hybridization_function(imp_solver)
    # Linear mixing
    Delta_out = imp_solver.Delta_w.copy()
    newDelta = alpha*Delta_out + (1-alpha)*Delta_in
    Delta_in << newDelta
