from triqs.utility import mpi
import numpy as np
from nrgljubljana_interface import Solver, SemiCircular
import sys

class BaseLatticeSolver:
    """
    Here we assume that the Green's function of the quantum impurity model
    is under the paramagnetic phase (Spin up and down cases are identical.)
    """
    def __init__(self):
        pass

    def load_data(self, imp_solver, filename):
        # Load previous data
        Delta_w = imp_solver.Delta_w['imp'].data.copy()
        if mpi.is_master_node():
            try:
                Delta_w = self._load_hybridization_function(filename).reshape(Delta_w.shape)
                print ('READ THE PREVIOUS DATA:', filename)
            except IOError:
                print ('START WITH A DEFAULT SETTING')
                Delta_w = self._default_hybridization_function()
        Delta_w = mpi.bcast(Delta_w)
        self._set_hybridization_function(imp_solver, Delta_w)

    def _set_hybridization_function(self, imp_solver, data):
        # Matrix size of Green's functions in block 'bl'
        block_size = lambda bl : len(imp_solver.Delta_w[bl].indices[0])
        for i, bl in enumerate(imp_solver.Delta_w.indices):
            assert(imp_solver.Delta_w[bl].data.shape == data.shape)
            for j, w in enumerate(imp_solver.Delta_w.mesh):
                for n in range(block_size(bl)): # only diagonal parts
                    imp_solver.Delta_w[bl][w][n, n] = data[j, n, n]

    def _load_hybridization_function(self, filename):
        raise NotImplementedError

    def _default_hybridization_function(self):
        raise NotImplementedError


class BetheLatticeSolver(BaseLatticeSolver):
    """
    self-consistent equation: Delta_w = D^2/4*G_w,
    where Delta_w is a hybridization function and G_w full Green's function.
    """
    def __init__(self, imp_solver, D):
        super(BetheLatticeSolver, self).__init__()
        # Initialize hybridization function (Bethe lattice)
        imp_solver.Delta_w['imp'] << (D/2)**2*SemiCircular(D)
        self._D = D
        self._default = imp_solver.Delta_w['imp'].data.copy()

    def update_hybridization_function(self, imp_solver, data):
        """
        record a new hybridiazation function into 'imp_solver'.
        """
        self._set_hybridization_function(imp_solver, data)

    def _load_hybridization_function(self, filename):
        """
        return the hybridization function according to the self-consistent equation.
        """
        realFreq, A_w, G_w_r, G_w_i, _,_,_,_,_,_ = np.loadtxt(filename, unpack = True, dtype = 'complex128')
        return (self._D/2)**2*(G_w_r + 1j * G_w_i)

    def _default_hybridization_function(self):
        """
        return the hybridization function of the non-interacting Green's function
        """
        if mpi.is_master_node():
            print ('INITIALIZE A HYBRIDIZATION FUNCTION WITH A SEMICIRCULAR DOS.')
        return self._default
