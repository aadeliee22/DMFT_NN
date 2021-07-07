import sys
import numpy as np
w, A_w, G_w, S_w = np.loadtxt(sys.argv[1], unpack = True, dtype = 'complex128')

np.savetxt('./result/A_w-'+sys.argv[1], np.array([w.real, A_w.real]).T)

np.savetxt('./result/G_w_real-'+sys.argv[1], np.array([w.real, G_w.real]).T)
np.savetxt('./result/G_w_imag-'+sys.argv[1], np.array([w.real, G_w.imag]).T)
np.savetxt('./result/S_w_real-'+sys.argv[1], np.array([w.real, S_w.real]).T)
np.savetxt('./result/S_w_imag-'+sys.argv[1], np.array([w.real, S_w.imag]).T)
