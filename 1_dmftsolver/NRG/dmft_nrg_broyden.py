#!/usr/bin/python3
import argparse
from triqs.utility import mpi
import numpy as np
from nrgljubljana_interface import Solver, SemiCircular
import utility
from scipy import optimize
from scipy.integrate import simps, trapz
from copy import deepcopy
import sys
import subprocess as sub

#================= Parameters =================
parser = argparse.ArgumentParser()
parser.add_argument("-U", help = "on-site interaction", type = float)
parser.add_argument("-T", help = "temperature", type = float, default = 1e-4)
parser.add_argument("-D", help = "half bandwidth", type = float, default = 1)
parser.add_argument("-alp", help = "alpha", type = float, default = 0.8)
parser.add_argument("-lamb", help = "lambda", type = float, default = 2.4)
parser.add_argument("-keepE", help = "keep energy", type = float, default = 12.0)
parser.add_argument("-niter", help = "# of iterations", type = int, default = 100)
parser.add_argument("-tol", help = "tolerance of the convergence criterion", type = float, default = 1e-4)
parser.add_argument("-load", help = "file name of previous data to restart a DMFT loop", type = str, default = None)
parser.add_argument("-way", help = "direction", type = int)
args = parser.parse_args()
D = args.D
U = args.U
T = args.T
alpha = args.alp
keepE = args.keepE
niter = args.niter
tol = args.tol
lamb = args.lamb
way = args.way
filename = f'./B_{way:.0f}_2.4/Bethe-{U:.3f}.dat'
#==============================================
# particle-hole symmetric case
mu = U/2
e_f = -U/2.0
B = 0.0
# Set up the Impurity Solver
imp_solver = Solver(model = "SIAM", symtype = "QS", mesh_max = 5.0, mesh_min = 1e-5, mesh_ratio = 1.01)
#imp_solver.set_verbosity(False)
# Solve Parameters
sp = { "T": T, "Lambda": lamb, "Nz": 4, "Tmin": 1e-6, \
        "keep": 3000, "keepmin":1000, "keepenergy": keepE, "alpha": 0.4, "bandrescale": 1.0 }
# Model Parameters
mp = { "U1": U, "B1": B, "eps1": e_f }
sp["model_parameters"] = mp
ob = ["n_d"]
# Set up the Lattice Solver
lattice_solver = utility.BetheLatticeSolver(imp_solver, D)
# Load previous data
lattice_solver.load_data(imp_solver, filename if args.load is None else args.load)


newG = lambda : imp_solver.G_w.copy()
nr_blocks = lambda bgf : len([bl for bl in bgf.indices])
block_size = lambda bl : len(imp_solver.G_w[bl].indices[0])
identity = lambda bl : np.identity(block_size(bl))
def fix_hyb_function(Delta, Delta_min):
  Delta_fixed = Delta.copy()
  for bl in Delta.indices:
    for w in Delta.mesh:
      for n in range(block_size(bl)): # only diagonal parts
        r = Delta[bl][w][n,n].real
        i = Delta[bl][w][n,n].imag
        Delta_fixed[bl][w][n,n] = r + 1j*(i if i<-Delta_min else -Delta_min)
  # Possible improvement: re-adjust the real part so that the Kramers-Kronig relation is maintained
  return Delta_fixed
# Hilbert Transformation
ht1 = lambda z: 2*(z-1j*np.sign(z.imag)*np.sqrt(1-z**2))
ht0 = lambda z: ht1(z/D)
EPS = 1e-20
ht = lambda z: ht0(z.real+1j*(z.imag if z.imag>0.0 else EPS))
def calc_G(Sigma, mu):
  Gloc = newG()
  for bl in Gloc.indices:
    for w in Gloc.mesh:
      for i in range(block_size(bl)):
        for j in range(block_size(bl)): # assuming square matrix
          if i == j:
            Gloc[bl][w][i,i] = ht(w + mu - Sigma[bl][w][i,i]) # Hilbert-transform
          else:
            assert abs(Sigma[bl][w][i,j])<1e-10, "This implementation only supports diagonal self-energy"
            Gloc[bl][w][i,j] = 0.0
  Glocinv = Gloc.inverse()
  Delta = newG()
  for bl in Delta.indices:
    for w in Delta.mesh:
      Delta[bl][w] = (w+mu)*identity(bl) - Sigma[bl][w] - Glocinv[bl][w]
  return Gloc, Delta

# real frequency domain
realFreq = np.array([w for w in imp_solver.Delta_w.mesh], dtype = 'complex128')
A_w = np.zeros_like(realFreq)
G_old = newG()
G_new = newG()
Delta_in = imp_solver.Delta_w.copy()
Delta_min=1e-6
n = 0

class Converged(Exception):
  def __init__(self, message):
      self.message = message

def dmft_step(Delta_in):
    global n, G_old, G_new
    if mpi.is_master_node():
        print ('# OF ITERATIONS:', (n+1))
    # Solve the impurity model
    Delta_fixed = fix_hyb_function(Delta_in, Delta_min)
    imp_solver.Delta_w << Delta_fixed
    imp_solver.solve(**sp)
    G_new, _ = calc_G(imp_solver.Sigma_w, mu)
    lattice_solver.update_hybridization_function(imp_solver, (D/2)**2 * G_new['imp'].data)
    Delta_out = imp_solver.Delta_w.copy()

    diff = np.mean(np.abs(G_new['imp'].data.reshape([-1])-G_old['imp'].data.reshape([-1])))
    G_old = G_new.copy()
    nexpv = imp_solver.expv["n_d"]
    A_w = -1/np.pi*np.array(G_new['imp'].data.reshape([-1]).imag)

    n += 1
    # Save results
    if mpi.is_master_node():
        np.savetxt(filename, np.array([realFreq.real, \
            A_w, \
            G_new['imp'].data.reshape([-1]).real, \
            G_new['imp'].data.reshape([-1]).imag, \
            imp_solver.Sigma_w['imp'].data.reshape([-1]).real, \
            imp_solver.Sigma_w['imp'].data.reshape([-1]).imag, \
            imp_solver.chi_NN_w['imp'].data.reshape([-1]).real,\
            imp_solver.chi_NN_w['imp'].data.reshape([-1]).imag,\
            imp_solver.chi_SS_w['imp'].data.reshape([-1]).real,\
            imp_solver.chi_SS_w['imp'].data.reshape([-1]).imag ]).T, \
            header = f"w   G_w   Sigma_w    chi_NN_w    chi_SS_w  (D: {D}, T: {T}, U: {U}, half-filling)")

    if n > 0:
        if mpi.is_master_node():
            print ("|G_NEW - G_OLD|: %.7f"%diff)
            print ("<n>: %.7f"%nexpv)
        if (diff < tol):
            if mpi.is_master_node():
                print("Converge on G(w)")
                #raise Converged("Converge!!")
        else:
            if mpi.is_master_node():
                print ('\n')
    else:
        if mpi.is_master_node():
            print ('\n')

    return Delta_out

newG = lambda : imp_solver.G_w.copy()
nr_blocks = lambda bgf : len([bl for bl in bgf.indices])
def bgf_to_nparray(bgf):
    return np.hstack((bgf[bl].data.flatten() for bl in bgf.indices))
def nparray_to_gf(a, gf):
    b = a.reshape(gf.data.shape)
    gf.data[:,:,:] = b[:,:,:]
def nparray_to_bgf(a):
    G = newG()
    split = np.split(a, nr_blocks(G)) # Here we assume all blocks are equally large
    for i, bl in enumerate(G.indices):
        nparray_to_gf(split[i], G[bl])
    return G
F = lambda D: dmft_step(D) - D
npF = lambda x : bgf_to_nparray(F(nparray_to_bgf(x)))
def solve_mix_B(Delta, alpha):
    xin = bgf_to_nparray(Delta)
    sol = optimize.broyden1(npF, xin, alpha=alpha, \
    verbose=True, reduction_method="svd", max_rank = 8, f_tol = tol, maxiter = niter)

# Start with a DMFT self-consistent loop
solve_mix_B(imp_solver.Delta_w, alpha)
#try:
#    solve_mix_B(imp_solver.Delta_w, alpha)
#except Converged as c:
#    print(c.message)
mpi.barrier()
