'''
maximally localized regional embedding

try to apply regional embedding one atom at a time, as an alternative to 
other local orbital methods
'''
import surf
from regional_embedding import emb
import sys
import os
import numpy
import copy
import cubegen
import math

print(sys.version)

args = surf.getArgs(sys.argv[1:])
args,cell,mf = surf.init_calc(args)
ref_atom = copy.deepcopy(cell.atom)

mlre_label = '@'

if not mf.converged:
    print('SCF did not converge: will not try MLRE')
    raise NotImplementedError

mlfrozen = None
ro_coeff = mf.mo_coeff
mo_occ = mf.mo_occ
nmo = mo_occ.shape[-1]
nmoidx = numpy.arange(nmo)
mlre_labels = ['' for ix in range(nmo)]

for ix in range(len(cell.atom)):
    atom = copy.deepcopy(ref_atom)
    atom[ix][0] += mlre_label
    cell.build(parse_arg = False,atom = atom,)
    atom_label = f'{ix}_{atom[ix][0]}'
    print(atom_label)

    # norb = ao's on atom
    # nocc/nvir = occupied/virtual ao's on the atom
    norb = len(cell.search_ao_label(mlre_label))
    nocc = emb.get_frag_nelec(cell,mlre_label)/2
    if math.floor(nocc) % 2 == 0:
        nocc = math.ceil(nocc)
    else:
        nocc = math.floor(nocc)
    if ix == len(cell.atom) - 1:
        nvir = len(mlfrozen) - nocc
    else:
        nvir = norb - nocc

    ro_coeff, frozen, emb_dat = emb.make_rdmet(cell,ro_coeff,mo_occ,frozen=mlfrozen,ft=(nocc,nvir),aolabels=mlre_label,minao='same')
    if mlfrozen is None:
        mlfrozen = numpy.empty(0,dtype=frozen.dtype)
    # we want to add the active indices of the current atom to mlfrozen, so invert
    # the indices of the returned frozen list
    if frozen is not None:
        active = nmoidx[~numpy.isin(nmoidx,frozen)]
    else:
        active = nmoidx[~numpy.isin(nmoidx,mlfrozen)]
    ix3 = 1
    for ix2 in active:
        mlre_labels[ix2] = atom_label
        if ix3 <= nocc:
            mlre_labels[ix2] += '_occ' 
        else:
            mlre_labels[ix2] += '_vir'
        ix3 += 1
    mlfrozen = numpy.concatenate((mlfrozen,active))


cube_path = os.path.normpath('../output_cube/' + 'mlre' + args.JOBID)
trailing_zeros = int(numpy.log10(ro_coeff.shape[-1])) + 1
for ind in range(ro_coeff.shape[-1]):
    cube_path_mo = cube_path + '_' + mlre_labels[ind].replace(mlre_label,'') + '_mo' + '{0}'.format(ind).zfill(trailing_zeros)
    cube_path_mo += '.cube'
    cubegen.orbital(args,cell,cube_path_mo,ro_coeff[:,ind])
    print(cube_path_mo)