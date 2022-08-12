from pyscf.pbc import gto, scf, df

# ghost labels the C vacancy
cell = gto.Cell()
atom = [
['C', (0., 0., 0.)], # or ['C', (0., 0., 0.)]
['C', (0.8917, 0.8917, 0.8917)],
['C', (0.    , 1.7834, 1.7834)],
['C', (0.8917, 2.6751, 2.6751)],
['C', (1.7834, 1.7834, 0.    )],
['C', (2.6751, 2.6751, 0.8917)],
['C', (1.7834, 3.5668, 1.7834)],
['C', (2.6751, 4.4585, 2.6751)],
['C', (1.7834, 0.    , 1.7834)],
['C', (2.6751, 0.8917, 2.6751)],
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
basis = 'gth-cc-pvdz'
pseudo = 'gth2-hf'

cell.build(parse_arg = False,
               a = a,
               atom = atom,
               verbose = 5,
               pseudo = pseudo,
               basis = basis,
               spin = None)

# SCF calculation might take a while
mf = scf.RHF(cell).density_fit()
#mf = scf.RHF(cell).rs_density_fit()
mf.kernel()