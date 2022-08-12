#C8 i16

from pyscf import gto, dft, scf, lib, fci, mcscf, lo, cc
from pyscf.mcscf import avas
import numpy as np
import scipy

def makeEmbeddedSpace(mf, aolabels, threshold=0.1):
    #properties of the input molecular object with basis Bas1
    if isinstance(mf, scf.uhf.UHF):
        print ("Have not implemented the uhf")
        exit(0)
    else:
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
    nocc = np.count_nonzero(mo_occ != 0)
    nsingle = mf.mol.spin
    ndouble = nocc - nsingle
    norb = mf.mo_coeff.shape[0]


    #make a mol with the minao basis Bas2
    mol = mf.mol
    pmol = mol.copy()
    pmol.atom = mol._atom
    pmol.unit = 'B'
    pmol.symmetry = False
    pmol.basis = 'minao'
    pmol.build(False, False)
    
    
    #find the labeled orbitals in main basis Bas1, and minao Bas2
    baslst1 = mol.search_ao_label(aolabels)
    baslst2 = pmol.search_ao_label(aolabels)
    
    s1 = mol.intor_symmetric('int1e_ovlp')[baslst1][:,baslst1]
    s2 = pmol.intor_symmetric('int1e_ovlp')[baslst2][:,baslst2]
    s21 = gto.intor_cross('int1e_ovlp', pmol, mol)[baslst2]
    s21 = np.dot(s21, mo_coeff)
    
    #find the iao from the occupied orbitals
    if (True):
        sa = s21.T.dot(scipy.linalg.solve(s2, s21, sym_pos=True))
        
        wocc, u = np.linalg.eigh(sa[:ndouble, :ndouble])
        ncas_occ = (wocc > threshold).sum()
        nelecas = (mol.nelectron) - (wocc < threshold).sum() * 2

        mocore = mo_coeff[: ,:ndouble].dot(u[:,wocc<threshold])
        mocas  = np.hstack((mo_coeff[: ,:ndouble].dot(u[:,wocc>threshold]), mo_coeff[:,ndouble:nocc]))

    #find the virtual orbitals
    if (True):

        sa11 = mol.intor_symmetric('int1e_ovlp')[baslst1] 
        sa11 = np.dot(sa11, mo_coeff[:,nocc:])  #overlap of 
        sb  = sa11.T.dot(scipy.linalg.solve(s1, sa11, sym_pos=True))
        wvirt, u = np.linalg.eigh(sb)

        nvirt_occ = (wvirt >threshold).sum()
        movirt = mo_coeff[:, nocc:].dot(u[:, wvirt>threshold])
        movirt2 = mo_coeff[:, nocc:].dot(u[:, wvirt<threshold])

    mo = 1.*mf.mo_coeff
    mo = np.hstack((mocore, mocas, movirt, movirt2))
    #mo = np.hstack((mocore, mocas, mo_coeff[:, nocc:].dot(u)))
    frozen = list(range(nocc - ncas_occ)) + list(range(nocc+nvirt_occ, norb))
    return mo, frozen



mol = gto.Mole()
mol.atom='''
 C                 -3.45546600    0.13923700   -0.96113200
 H                 -3.50419100    0.57569600    0.04005600
 H                 -2.56510800    0.53558500   -1.46710600
 C1                 -3.31698800   -1.39450400   -0.85633400
 H1                 -2.35174400   -1.62971400   -0.37688300
 C                 -4.45545800   -2.01147000   -0.01694900
 H                 -4.34831100   -1.68865000    1.02333500
 H                 -5.40551100   -1.61725700   -0.38885300
 C                 -4.42940400   -3.54189200   -0.12117300
 H                 -5.26162600   -3.96036500    0.45992300
 H                 -3.50553900   -3.92689100    0.32755500
 C2                 -4.51578000   -3.98294300   -1.57307400
 H3                 -3.87823300   -3.05514400   -2.11399600
 H2                 -5.51715300   -3.84570900   -1.99437700
 C                 -3.88498500   -5.30416300   -1.96849000
 H                 -2.84301000   -5.32417000   -1.63012500
 H                 -3.86008700   -5.38137300   -3.06040400
 O                 -4.64012000    0.50670900   -1.64274700
 H                 -4.64425100   -0.02669600   -2.44799500
 O1                 -3.34456500   -1.87485600   -2.17913300
 C                 -4.63283900   -6.52256400   -1.39669700
 H                 -5.67361900   -6.53907400   -1.73315200
 H                 -4.63439100   -6.51414100   -0.30349800
 C                 -3.96780562   -7.83347253   -1.85588044
 H                 -4.51135350   -8.66641453   -1.46133853
 H                 -3.97093699   -7.87960955   -2.92488071
 H                 -2.95905788   -7.86521916   -1.50046498
'''

mol.basis = "cc-pvdz"
mol.max_memory = 100000
mol.verbose = 4
mol.charge = 0 #*****
mol.spin = 1  #*****
mol.build()


#mf = dft.RKS(mol).density_fit()
mf = scf.ROHF(mol)
mf.chkfile = 'ts.chk'
mf.init_guess = 'chk'
mf.kernel()


aolabels =  ['O1', 'H1', 'H2', 'H3', 'C1', 'C2']
mo, frozen = makeEmbeddedSpace(mf, aolabels)
mf.mo_coeff = mo


#now change the mf
mycc = cc.UCCSD(mf).set(frozen=frozen)
mycc.kernel()


