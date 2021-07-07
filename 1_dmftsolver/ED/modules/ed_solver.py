import numpy as np
from . import _ed_solver

def get_value_from_dict(kwargs, key, default):
    try:
        value = kwargs[key]
    except KeyError:
        value = default
    return value

class solver:
    def __init__(self, **kwargs):
        """
          (handling parameter)
          nbath : # of bath prameters
          beta : inverse temperature
          niwn : # of Matsubara frequency

          (default parameter)
          dist : tolerance of the optimization of the Anderson impurity projection
          pvec : bath parameter [eps_0, eps_1,..., eps_{nbath-1}, V_0, V_1,...,V_{nbath-1}]
        """
        self._nbath = int(kwargs['nbath'])
        self._beta = float(kwargs['beta'])
        self._niwn = int(get_value_from_dict(kwargs, 'niwn', 1000))
        self._dist = float(get_value_from_dict(kwargs, 'dist', 1e-8))
        self._pvec = get_value_from_dict(kwargs, 'pvec', self._get_default_pvec())
        self._cpp_solver = _ed_solver.ed_solver(self._nbath, self._niwn, self._beta, 10, 1000)

    def solve_with_pvec(self, params, pvec):
        """
          * return full Green function, density, and double occupancy
          params: dictionary (chemical potential 'mu', onsite interaction 'U')
          pvec: bath parameter [eps_0, eps_1,..., eps_{nbath-1}, V_0, V_1,...,V_{nbath-1}]
        """
        assert(2*self._nbath == len(pvec))
        U = float(params['U'])
        mu = float(params['mu'])
        FullGreen = self._cpp_solver.run(pvec, U, mu)
        density = self._cpp_solver.get_density()
        double_occ = self._cpp_solver.get_double_occupancy()
        return [FullGreen, density, double_occ]

    def solve(self, params, BathGreen):
        """
          * return full Green function, density, and double occupancy.
          params: dictionary (chemical potential 'mu', onsite interaction 'U')
          BathGreen: bath Greeen function (complex type)
        """
        assert(self._niwn == len(BathGreen))
        BathGreen = np.array(BathGreen).astype('complex128')
        mu = float(params['mu'])
        self._pvec = self._cpp_solver.anderson_impurity_projection(self._pvec, BathGreen, mu, self._dist)
        return self.solve_with_pvec(params, self._pvec)

    def get_pvec(self):
        """
          * return bath parameter
        """
        return self._pvec.copy()

    def _get_default_pvec(self):
        """
          norm of pvec[self._nbath:] = 1/4 (hybridization sum-rule of Bethe lattice)
        """
        pvec = np.zeros([2*self._nbath], dtype = 'float64')
        # eps_i has a mirror symmetry against eps=0
        for i in range(self._nbath):
            pvec[i] = -1+i*2.0/(self._nbath-1)
        pvec[self._nbath:] = 0.5/np.sqrt(self._nbath)
        return pvec
