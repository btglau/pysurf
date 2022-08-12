from pyscf.pbc import gto, scf, df

# ghost labels the C vacancy
cell = gto.Cell()
atom = [
['C', (0., 0., 0.)], # or ['GHOST-C', (0., 0., 0.)]
['C', (0.8917, 0.8917, 0.8917)]
]
a = [[1.7834, 0.    , 1.7834],
    [1.7834, 1.7834, 0.    ],
    [0.    , 1.7834, 1.7834]]
basis = 'gth-cc-pvdz'
pseudo = 'gth2-HF'

cell.build(parse_arg = False,
               a = a,
               atom = atom,
               verbose = 5,
               pseudo = pseudo,
               basis = basis,
               spin = None)

# SCF calculation might take a while
#mf = scf.RHF(cell).density_fit()
mf = scf.RHF(cell).rs_density_fit()
mf.kernel()