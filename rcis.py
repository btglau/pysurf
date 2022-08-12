# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
# +modifications by Bryan Lau (blau1270@gmail.com)

import time
import numpy
from functools import reduce

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf import __config__
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, _mo_without_core, _mo_energy_without_core
from pyscf.tdscf.rhf import get_nto


def kernel(cis, nroots=1):
    cput0 = (time.process_time(), time.perf_counter())
    log = logger.Logger(cis.stdout, cis.verbose)
    cis.dump_flags()

    matvec, diag = cis.gen_matvec()

    size = cis.vector_size()
    nroots = min(nroots, size)
    user_guess = False
    guess = cis.get_init_guess(nroots, diag)

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    # GHF or customized RHF/UHF may be of complex type
    real_system = (cis._scf.mo_coeff[0].dtype == numpy.double)

    eig = lib.davidson1
    def pickeig(w, v, nroots, envs):
        real_idx = numpy.where(abs(w.imag) < 1e-3)[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
    conv, es, vs = eig(matvec, guess, precond, pick=pickeig,
                       tol=cis.conv_tol, max_cycle=cis.max_cycle,
                       max_space=cis.max_space, nroots=nroots, verbose=log)

    if cis.verbose >= logger.INFO:
        for n, en, vn, convn in zip(range(nroots), es, vs, conv):
            logger.info(cis, 'CIS root %d E = %.16g  conv = %s',
                        n, en, convn)
        log.timer('CIS', *cput0)

    xy = [(xi.reshape(cis.nocc,cis.nmo - cis.nocc)*numpy.sqrt(.5),0) for xi in vs]
    return conv, es.real, xy


def matvec(cis, vector):
    nocc = cis.nocc
    nmo = cis.nmo

    r1 = cis.vector_to_amplitudes(vector, nmo, nocc)
    eris = cis.eris 

    mo_energy = eris.mo_energy
    e_vir = mo_energy[nocc:]
    e_occ = mo_energy[:nocc]
    Hr1  = lib.einsum('a,ia->ia', e_vir, r1)
    Hr1 -= lib.einsum('i,ia->ia', e_occ, r1)

    Hr1 -= lib.einsum('jiab,jb->ia', eris.oovv, r1)
    if cis.singlet is True:
        Hr1 += 2*lib.einsum('iabj,jb->ia', eris.ovvo.conj(), r1)

    vector = cis.amplitudes_to_vector(Hr1)
    return vector


class CIS(lib.StreamObject):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', 50) 
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', 1e-7)

        self.frozen = frozen
        self.singlet = True

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e = None
        self.v = None
        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        if self.singlet:
            logger.info(self, 'singlet')
        else:
            logger.info(self, 'triplet')
        if self.frozen is not None:
            logger.info(self, 'frozen orbitals %s', self.frozen)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def get_init_guess(self, nroots=1, diag=None):
        idx = diag.argsort()
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', numpy.double)
        nroots = min(nroots, size)
        guess = []
        for i in idx[:nroots]:
            g = numpy.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc*nvir

    def get_diag(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        Hr1 = numpy.zeros((nocc,nvir))

        eris = self.eris 

        mo_energy = eris.mo_energy
        Hr1 = mo_energy[None,nocc:] - mo_energy[:nocc,None]

        for i in range(nocc):
            for a in range(nvir):
                Hr1[i,a] -= eris.oovv[i,i,a,a]
                if self.singlet is True:
                    Hr1[i,a] += 2*eris.ovvo[i,a,a,i].conj()

        return self.amplitudes_to_vector(Hr1)

    matvec = matvec

    def gen_matvec(self, diag=None, **kwargs):
        if diag is None: diag = self.get_diag()
        matvec = lambda xs: [self.matvec(x) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        nvir = nmo - nocc
        return vector.reshape((nocc,nvir))

    def amplitudes_to_vector(self, r1):
        return r1.ravel()

    def kernel(self, nroots=1):
        self.eris = self.ao2mo(self.mo_coeff)
        self.converged, self.e, self.xy = kernel(self, nroots)
        return self.e, self.xy

    def ao2mo(self, mo_coeff=None):
        return _make_eris(self,mo_coeff)

def _make_eris(mycc, mo_coeff=None):
    nocc = mycc.nocc
    nmo = mycc.nmo
    eris = _ChemistsERIs()
    eris._common_init_(mycc,mo_coeff)
    mo_coeff = eris.mo_coeff
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    # total storage: ovpq
    if getattr(mycc._scf, 'with_df', None):
        logger.warn(mycc,'DF-HF is found. (ia|jb) is computed based on the DF '
                '3-tensor integrals.\n'
                'native DF for RCIS is not implemented')
        logger.debug(mycc,'transform (ia|jb) with_df')
        #eris.oovv = mycc._scf.with_df.ao2mo([orbo,orbo,orbv,orbv], compact=False).reshape(nocc,nocc,nmo-nocc,nmo-nocc)
        #eris.ovvo = mycc._scf.with_df.ao2mo([orbo,orbv,orbv,orbo], compact=False).reshape(nocc,nmo-nocc,nmo-nocc,nocc)

        # more efficient ao2mo, IO wise?
        opvq = mycc._scf.with_df.ao2mo([orbo,mo_coeff,orbv,mo_coeff], compact=False).reshape(nocc,nmo,nmo-nocc,nmo)
        eris.oovv = opvq[:,:nocc,:,nocc:]
        eris.ovvo = opvq[:,nocc:,:,:nocc]
    else:
        eris.oovv = ao2mo.general(mol, [orbo,orbo,orbv,orbv], compact=False).reshape(nocc,nocc,nmo-nocc,nmo-nocc)
        eris.ovvo = ao2mo.general(mol, [orbo,orbv,orbv,orbo], compact=False).reshape(nocc,nmo-nocc,nmo-nocc,nocc)
    return eris


class _ChemistsERIs:
    '''(pq|rs)'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.e_hf = None

        self.oovv = None
        self.ovvo = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

# Note: Recomputed fock matrix and HF energy since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(numpy.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        self.nocc = mycc.nocc
        self.mol = mycc.mol

        # Note self.mo_energy can be different to fock.diagonal().
        # self.mo_energy is used in the initial guess function (to generate
        # MP2 amplitudes) and CCSD update_amps preconditioner.
        # fock.diagonal() should only be used to compute the expectation value
        # of Slater determinants.
        self.mo_energy = self.fock.diagonal().real
        return self


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import tdscf

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    
    myci = tdscf.TDA(mf)
    myci.nstates = 4
    log = logger.Logger(myci.stdout, 9)
    cput0 = (time.process_time(), time.perf_counter())
    es_ref, v = myci.kernel()
    log.timer(myci.__class__, *cput0)
    myci.singlet = False
    et_ref, v = myci.kernel()

    cput0 = (time.process_time(), time.perf_counter())
    myci = CIS(mf)
    log.timer(myci.__class__, *cput0)
    es, v = myci.kernel(nroots=4)
    myci.singlet = False
    et, v = myci.kernel(nroots=4)

    print(es_ref-es)
    print(et_ref-et)

    print(es*27.21139)
    myci = CIS(mf, frozen=[0,1])
    es_frz, v = myci.kernel(nroots=4)
    print(es_frz*27.21139)
    myci = CIS(mf, frozen=[0,1,2,3])
    es_frz, v = myci.kernel(nroots=4)
    print(es_frz*27.21139)
    myci = CIS(mf, frozen=[0,1,2,3,10,11,12])
    es_frz, v = myci.kernel(nroots=4)
    print(es_frz*27.21139)

    '''
    [blau@rusty2 pysurf]$ p3pyscf rcis.py
    max memory 16000
    threads 1
    [ 1.53214108e-11  1.29325994e-11 -5.40070100e-10  9.76108083e-13]
    [-1.59100511e-12  1.26859634e-12 -1.38980061e-09 -7.17925719e-12]
    [ 9.21539477 10.99035033 11.83381569 13.62302971]
    [ 9.21659597 10.99346539 11.85527359 13.64198922]
    [ 9.2291491  10.99430156 23.0559715  25.05317876]
    [ 9.24046562 11.29744068 23.0577421  25.07735914]
    '''

    '''
    With my modifications to ao2mo to not do a full incore transformation
    [blau@rusty1 pysurf]$ p3pyscf rcis.py
    max memory 16000
    threads 1
    [ 1.53213553e-11  1.29321553e-11 -5.40069101e-10  9.76885239e-13]
    [-1.59117164e-12  1.26865185e-12 -1.38980000e-09 -7.17909066e-12]
    [ 9.21539477 10.99035033 11.83381569 13.62302971]
    [ 9.21659597 10.99346539 11.85527359 13.64198922]
    [ 9.2291491  10.99430156 23.0559715  25.05317876]
    [ 9.24046562 11.29744068 23.0577421  25.07735914]
    '''

    print('\ntest PBC (diamond?) Compare native pyscf response vs. ao2mo, both w/ DF')
    # nelectron = 32
    from pyscf.pbc import gto, scf, tdscf
    import ccg
    import numpy
    import copy

    cell = gto.Cell()
    # .a is a matrix for lattice vectors.
    cell.a = '''
    3.5668  0       0
    0       3.5668  0
    0       0       3.5668'''
    cell.atom = '''C     0.      0.      0.    
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.    
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751'''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.build()

    mf = scf.RHF(cell).density_fit() # GDF
    mf.exxdiv = None
    ehf = mf.kernel()
    print("HF energy (per unit cell) = %.17g" % ehf)
    nroots = 6

    # pyscf, exxdiv=None inherited from scf
    myci = tdscf.TDA(mf)
    print(myci.__class__)
    myci.nstates = nroots
    cput0 = (time.process_time(), time.perf_counter())
    es_ref, v = myci.kernel()
    log.timer(myci.__class__, *cput0)
    myci.singlet = False
    et_ref, v = myci.kernel()

    print(es_ref*27.21139)
    print(et_ref*27.21139)

    # this module, with no changes to exxdiv
    # should always be the same as the mf calculation with whatever exxdiv is chosen
    print('\nCCG eri cis with no modifications')
    myci = CIS(mf)
    print(myci.__class__)
    cput0 = (time.process_time(), time.perf_counter())
    es, v = myci.kernel(nroots=nroots)
    log.timer(myci.__class__, *cput0)
    myci.singlet = False
    et, v = myci.kernel(nroots=nroots)

    print(es_ref-es)
    print(et_ref-et)

    print(es*27.21139)
    myci = CIS(mf, frozen=list(range(4)))
    es_frz, v = myci.kernel(nroots=nroots)
    print(es_frz*27.21139)
    myci = CIS(mf, frozen=list(range(8)))
    es_frz, v = myci.kernel(nroots=nroots)
    print(es_frz*27.21139)
    myci = CIS(mf, frozen=list(range(8)) + list(range(24,32)))
    es_frz, v = myci.kernel(nroots=nroots)
    print(es_frz*27.21139)

    # ccg eri cis without exxdiv
    print('\nCCG eri cis with no exxdiv, regardless of scf')
    myci = ccg.CIS(mf)
    print(myci.__class__)
    cput0 = (time.process_time(), time.perf_counter())
    es, v = myci.kernel(nroots=nroots)
    log.timer(myci.__class__, *cput0)
    myci.singlet = False
    et, v = myci.kernel(nroots=nroots)

    print(es_ref-es)
    print(et_ref-et)

    print(es*27.21139)
    myci = ccg.CIS(mf, frozen=list(range(4)))
    es_frz, v = myci.kernel(nroots=nroots)
    print(es_frz*27.21139)
    myci = ccg.CIS(mf, frozen=list(range(8)))
    es_frz, v = myci.kernel(nroots=nroots)
    print(es_frz*27.21139)
    myci = ccg.CIS(mf, frozen=list(range(8)) + list(range(24,32)))
    es_frz, v = myci.kernel(nroots=nroots)
    print(es_frz*27.21139)

    # ccg response cis without exxdiv
    print('\nCCG response no exxdiv, regardless of scf')
    myci = ccg.TDA(copy.copy(mf))
    print(myci.__class__)
    cput0 = (time.process_time(), time.perf_counter())
    es, v = myci.kernel(nstates=nroots)
    log.timer(myci.__class__, *cput0)
    myci.singlet = False
    et, v = myci.kernel(nstates=nroots)

    print(es_ref-es)
    print(et_ref-et)

    print(es*27.21139)
    myci = ccg.TDA(copy.copy(mf), frozen=list(range(4)))
    es_frz, v = myci.kernel(nstates=nroots)
    print(es_frz*27.21139)
    myci = ccg.TDA(copy.copy(mf), frozen=list(range(8)))
    es_frz, v = myci.kernel(nstates=nroots)
    print(es_frz*27.21139)
    myci = ccg.TDA(copy.copy(mf), frozen=list(range(8)) + list(range(24,32)))
    es_frz, v = myci.kernel(nstates=nroots)
    print(es_frz*27.21139)