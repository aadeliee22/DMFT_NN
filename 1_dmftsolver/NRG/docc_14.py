import subprocess as sub
import argparse
import numpy as np
from scipy.integrate import simps

parser = argparse.ArgumentParser()
parser.add_argument("-U", help = "on-site interaction", type = float)
parser.add_argument("-T", help = "temperature", type = float, default = 1e-4)
parser.add_argument("-lamb", help = "lambda", type = float, default = 1.6)
args = parser.parse_args()
U = args.U
T = args.T
lamb = args.lamb
filename = f'./B_14_{lamb:.1f}/Bethe-{U:.2f}.dat'
w, Aw, Gwr, Gwi, Swr, Swi = np.loadtxt(filename, unpack = True, dtype = 'complex128')


sub.run(['g++', 'docc_14.cpp', '-o', 'docc14', '-I/home/hyejin/fftw-3.3.9', '-L/opt/fftw/lib', '-lfftw3', '-std=gnu++11'])
beta = 1.0/T
N = 80000
omega = np.pi/beta * (2*np.arange(N+1)+1) # w_n
def realtoimag(x):
    N = len(omega)
    return np.array([simps(x/(1j*omega[i]-w.real), w.real) for i in range (N)])
G_iw = np.zeros(len(omega), dtype = 'complex128')
G_iw = realtoimag(Aw.real)
np.savetxt(f'./Giw_{lamb:.1f}/Giw_1to4_{U:.2f}.dat', \
            np.array([omega.real, G_iw.real, G_iw.imag]).T)
np.savetxt(f'./num/{lamb:.1f}_{U:.2f}', np.array([f'{U:.2f}\n{lamb:.1f}']), fmt='%s')
myint = open(f'./num/{lamb:.1f}_{U:.2f}')
p = sub.Popen('./docc14', stdin=myint)
p.wait()
