import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pbcscf
from pyscf.pbc.tools import pyscf_ase

from ase.build import make_supercell, bulk
from ase.lattice.cubic import Diamond

import sys
import argparse
parser = argparse.ArgumentParser(description="Setup a surface calculation")
# electronic structure arguments
parser.add_argument('-w',help="size of supercell",default=1,type=int)
parser.add_argument('-g',help="0 - pristine; 1 - vacancy; 2 - vacancy+ghost",default=0,type=int)
parser.add_argument('-d',help="0 - GDF; 1 - RSDF;",default=0,type=int)
parser.add_argument('-j',help="name of file to save job (ignored)",default=None)
args = parser.parse_args(sys.argv[1:])

a = 3.567095
#ase_atom=Diamond(symbol='C', latticeconstant=a)
ase_atom=bulk('C','diamond',a=a)
if args.w > 1:
    ase_atom = make_supercell(ase_atom,numpy.diag([args.w]*3))

print(ase_atom.get_volume())

cell = pbcgto.Cell()
cell.verbose = 5
atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)

basis = {'default':'gth-cc-pvdz'}
pseudo = {'default':'gth2-hf'}
if args.g == 1:
    atom.pop(0)
elif args.g == 2:
    atom[0][0] = 'ghost-' + atom[0][0]
    basis[atom[0][0]] = basis['default']
    pseudo[atom[0][0]] = pseudo['default']

cell.atom=atom
cell.a=ase_atom.cell
cell.basis = basis
cell.pseudo = pseudo
cell.build()

# SCF calculation might take a while
if args.d == 0:
    mf = pbcscf.RHF(cell).density_fit()
elif args.d == 1:
    mf = pbcscf.RHF(cell).rs_density_fit()
    mf.with_df.use_bvk = False
mf.kernel()