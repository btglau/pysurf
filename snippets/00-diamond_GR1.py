'''
Regional embedding example for the GR1 defect in diamond (C vacancy)
2x2x2 supercell

Demonstrates embedding for ground and excited state CCSD
'''

from pyscf.pbc import gto, scf, df, tools
from pyscf.cc import dfccsd
from pyscf import lib
import emb

# ghost labels the C vacancy
# '!' labels the atoms in the embedding region
cell = gto.Cell()
atom = [
['ghost-C!', (0., 0., 0.)],
['C!', (0.8917, 0.8917, 0.8917)],
['C', (0.    , 1.7834, 1.7834)],
['C!', (0.8917, 2.6751, 2.6751)],
['C', (1.7834, 1.7834, 0.    )],
['C!', (2.6751, 2.6751, 0.8917)],
['C', (1.7834, 3.5668, 1.7834)],
['C', (2.6751, 4.4585, 2.6751)],
['C', (1.7834, 0.    , 1.7834)],
['C!', (2.6751, 0.8917, 2.6751)],
['C', (1.7834, 1.7834, 3.5668)],
['C', (2.6751, 2.6751, 4.4585)],
['C', (3.5668, 1.7834, 1.7834)],
['C', (4.4585, 2.6751, 2.6751)],
['C', (3.5668, 3.5668, 3.5668)],
['C', (4.4585, 4.4585, 4.4585)]
]
a = [[3.5668, 0.    , 3.5668],
     [3.5668, 3.5668, 0.    ],
     [0.    , 3.5668, 3.5668]]
basis = 'gth-dzvp'
pseudo = 'gth-pade'

cell.build(parse_arg = False,      
               a = a,
               atom = atom,
               verbose = 1, # set to 5 to see the make_dmet output
               pseudo = pseudo,
               basis = basis,
               spin = None)

# SCF calculation might take a while
mf = scf.RHF(cell)
mf.with_df = df.GDF(cell)
# mf = scf.RHF(cell).density_fit() is a one line replacement for above
ehf = mf.kernel()

# the GR1 vacancy SCF solution is not stable with exxdiv, but we can also fix that
# with a stability analysis. The alternative solution is initializing RHF without
# exxdiv: mf = scf.RHF(cell,exxdiv=None)
from pyscf.ao2mo.incore import iden_coeffs
mo1 = mf.stability()[0]
for cycles in range(5):
    if iden_coeffs(mo1,mf.mo_coeff):
        break
    else:
        print('Unstable solution found')
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1 = mf.stability()[0]

# create regional embedding object
# defaults: eigenvalue threshold of 0.1 for occupied and virtual
# szv is used for the fragment basis
aolabels = '!'
ro_coeff, frozen, emb_dat = emb.make_rdmet(cell,mf,ft=(0.1,0.1),aolabels=aolabels,minao='minao')

# canonicalize
co_coeff, co_energy = emb.canonicalize(mf,ro_coeff,frozen,fock_ao=True)

# we use DFCCSD here because it has the most memory efficient implementation
# for real integrals at the gamma point
# CCSD calculations are done without the ewald exxdiv correction
class DFCCSD(dfccsd.RCCSD):
    def ao2mo(self, mo_coeff=None):
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = dfccsd._make_df_eris(self, mo_coeff)
        
        madelung = tools.madelung(self._scf.cell, self._scf.kpt)
        eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)
        
        return eris

def _adjust_occ(mo_energy, nocc, shift):
    '''Modify occupied orbital energy'''
    mo_energy = mo_energy.copy()
    mo_energy[:nocc] += shift
    return mo_energy

# use the semi-canonicalized regional embedding orbitals
# the cc module recalculates the Fock matrix / mo_energies from the provided
# mo_coeff's
mycc = DFCCSD(mf,frozen=frozen,mo_coeff=co_coeff)
eris = mycc.ao2mo()
e, t1, t2 = mycc.kernel(eris=eris)

# eom-ee-ccsd calculation
eome, eomv = mycc.eomee_ccsd_singlet(eris=eris)

'''
[blau@rusty1 ~]$ seff 999372
Job ID: 999372
Cluster: slurm
User/Group: blau/blau
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 40
CPU Utilized: 06:26:04
CPU Efficiency: 96.20% of 06:41:20 core-walltime
Job Wall-clock time: 00:10:02
Memory Utilized: 5.63 GB
Memory Efficiency: 0.75% of 750.00 GB

Embed vacancy + nearest neighbours (5 atoms total)
converged SCF energy = -82.574394982492
DFCCSD converged
E(DFCCSD) = -72.78626647116053  E_corr = -0.4151539508425309
EOM-CCSD root 0 E = 0.1096697727486331  qpwt = 0.955422  conv = True

Only embed the vacancy (ghost-C):
DFCCSD converged
E(DFCCSD) = -72.38283328931045  E_corr = -0.01172076916451058
EOM-CCSD root 0 E = 0.9802106688055432  qpwt = 0.997139  conv = True
'''