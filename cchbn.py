'''

a script to generate quantum espresso input from ASE

'''

from ase.lattice.hexagonal import Graphene 
from ase.units import Bohr
import numpy as np
import os

n = 5
alat = 2.50
clat = 20

fn = 'pbe'
C = 'C.' + fn + '-n-kjpaw_psl.1.0.0.UPF'
B = 'B.' + fn + '-n-kjpaw_psl.1.0.0.UPF'
N = 'N.' + fn + '-n-kjpaw_psl.1.0.0.UPF'
#C = 'C.' + fn + '-n-rrkjus_psl.1.0.0.UPF'
#B = 'B.' + fn + '-n-rrkjus_psl.1.0.0.UPF'
#N = 'N.' + fn + '-n-rrkjus_psl.1.0.0.UPF'
atom_species = [12.011,10.813,24.753]
ecutwfc = 70 # Ry
ecutrho = 4 # ecutrho * ecutwfc, default 4
kpt = [1,2,3,4]

append = '_ecut%g_c%g'%(ecutwfc,clat)

folder_path = '/Users/blau/Code/MATLAB/FI/qe_input/'
header_path = os.path.normpath(folder_path + 'qe_header_ibrav1_hbn.header')

ase_atoms = Graphene(symbol= 'C', latticeconstant={'a':alat,'c':clat}, size=(n,n,1))
with_c2 = True

for k in kpt:
    if with_c2:
        # BN + C2
        ase_atoms.symbols = 'CC'+'BN'*(n*n-1)
        filename = 'hbn_%d%d1_k%d'%(n,n,k) + append + '.in'
    else:
        # Pure BN
        ase_atoms.symbols = 'BN'*(n*n)
        filename = 'hbn_%d%d1.in'%(n,n)

    with open(header_path,mode='r') as f:
        read_data = f.read()

    read_data = read_data.replace('REPLACE_FN',fn)
    read_data = read_data.replace('REPLACE_CELLDM1',str(alat*n/Bohr))
    read_data = read_data.replace('REPLACE_CELLDM3',str(clat/(n*alat))) # celldm(3) = c/a
    read_data = read_data.replace('REPLACE_NAT',str(len(ase_atoms.symbols)))
    read_data = read_data.replace('REPLACE_ECUTWFC',str(ecutwfc))
    read_data = read_data.replace('REPLACE_ECUTRHO',str(ecutwfc*ecutrho))

    read_data += '\nATOMIC_SPECIES\n'
    read_data += 'C ' + str(atom_species[0]) + ' ' + C + '\n'
    read_data += 'B ' + str(atom_species[1]) + ' ' + B + '\n'
    read_data += 'N ' + str(atom_species[2]) + ' ' + N + '\n'

    read_data += 'ATOMIC_POSITIONS { angstrom }\n'
    #np.set_printoptions(suppress=True)
    for at in ase_atoms:
        position = np.array2string(at.position,separator=' ',precision=8,max_line_width=200,suppress_small=True)[1:-1] + ' '
        position = position.replace('. ',' ')
        position = ' '.join(position.split())
        read_data += at.symbol + ' ' + position + '\n'

    read_data += 'K_POINTS (automatic)\n'
    read_data += (str(k) + ' ')*2 + '1 0 0 0'

    out_path = os.path.normpath(folder_path + filename)
    with open(out_path,mode='w') as f:
        f.write(read_data)
