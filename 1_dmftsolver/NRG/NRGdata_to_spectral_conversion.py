import numpy as np
import os.path as path

D = 1
T = 1e-4
U = np.array([0.001*i for i in range(500, 5001)])
U14 = np.zeros(142, dtype = 'float64')
U41 = np.zeros(129, dtype = 'float64')

n = 0
for i, u in enumerate(U):
    if path.isfile(f'./B_14/Bethe-{u:.3f}.dat')==False: continue
    U14[n] = u
    n += 1
    
n = 0
for i, u in enumerate(U):
    if path.isfile(f'./B_41/Bethe-{u:.3f}.dat')==False: continue
    U41[n] = u
    n += 1
    
    
for i,u in enumerate(U14):
	w, A, _,_,_,_,_,_,_,_ = np.loadtxt(f'./B_14/Bethe-{u:.3f}.dat', unpack = True, dtype = 'float64')
	filename = f'./NRG_metal_insulator/Bethe-{u:.3f}.dat'
	np.savetxt(filename, np.array([w, A]).T, header = f"w   A_w  (D: {D}, T: {T}, U: {u:.3f}, half-filling)")
	
for i,u in enumerate(U41):
	w, A, _,_,_,_,_,_,_,_ = np.loadtxt(f'./B_41/Bethe-{u:.3f}.dat', unpack = True, dtype = 'float64')
	filename = f'./NRG_insulator_metal/Bethe-{u:.3f}.dat'
	np.savetxt(filename, np.array([w, A]).T, header = f"w   A_w  (D: {D}, T: {T}, U: {u:.3f}, half-filling)")
