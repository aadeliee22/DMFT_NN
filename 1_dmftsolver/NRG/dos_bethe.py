from triqs.gf import *
from triqs.operators import *
from h5 import *
from triqs.utility import mpi
from nrgljubljana_interface import Solver, MeshReFreqPts, hilbert_transform_refreq
import math, os, warnings
import numpy as np
from scipy import interpolate, integrate, special, optimize
from collections import OrderedDict

table = np.loadtxt('dos_bethe.dat')
w, _ = np.loadtxt('dos_bethe.dat', unpack=True, dtype='float64')
global dosA
dosA = Gf(mesh=MeshReFreqPts(table[:,0]), target_shape=[])
for i, w in enumerate(dosA.mesh):
	dosA[w] = np.array([[ table[i,1] ]])
ht0 = lambda z: hilbert_transform_refreq(dosA, z)


np.savetxt('dos_bethe_result', np.array([w, ht0(w)]).T)
