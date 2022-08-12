import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pbcscf
from pyscf.pbc.tools import pyscf_ase

# set up the basic cell
'''
A = (alat/2)*[0 1 1;
        1 0 1;
        1 1 0];
'''
from ase.build import bulk
from ase.build import make_supercell
a1 = bulk('MgO','rocksalt',a=4.2568)
a1 = bulk('C','diamond',a=3.567095)
print(a1.cell)

# build supercell
N = 3
P = [[N,0,0],
     [0,N,0],
     [0,0,N]]
a2 = make_supercell(a1, P, wrap=True, tol=1e-05)
print(a2.cell)
print(numpy.linalg.det(a2.cell))
cell = pbcgto.Cell()
cell.verbose = 0
cell.atom=pyscf_ase.ase_atoms_to_pyscf(a2)
cell.a=a2.cell
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.build()
mf=pbcscf.RHF(cell)
S = mf.get_ovlp()
w,v = numpy.linalg.eigh(S)
print(w.min())

'''
This cell obeys the right hand rule for a coordinate system and is used in 
quantum-espresso to define the fcc primitive cell
A = (alat/2)*[-1 0 1;
        0 1 1;
        -1 1 0];
Permute the supercell rows to match
'''
a2.cell[:,0] *= -1
P = [[0,1,0],
     [1,0,0],
     [0,0,1]]
a2.cell = P @ a2.cell
print(a2.cell)
print(numpy.linalg.det(a2.cell))
cell = pbcgto.Cell()
cell.verbose = 0
cell.atom=pyscf_ase.ase_atoms_to_pyscf(a2)
cell.a=a2.cell
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.build()
mf=pbcscf.RHF(cell)
S = mf.get_ovlp()
w,v = numpy.linalg.eigh(S)
print(w.min())