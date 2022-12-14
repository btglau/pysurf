#!/usr/bin/env python
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         James D. McClain
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import time
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf import __config__


def kernel(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    matvec, diag = eom.gen_matvec(imds, left, **kwargs)

    size = eom.vector_size()
    nroots = min(nroots, size)
    if guess is not None:
        user_guess = True
        for g in guess:
            assert g.size == size
    else:
        user_guess = False
        guess = eom.get_init_guess(nroots, koopmans, diag)

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    # GHF or customized RHF/UHF may be of complex type
    real_system = (eom._cc._scf.mo_coeff[0].dtype == np.double)

    eig = lib.davidson_nosym1
    if user_guess or koopmans:
        assert len(guess) == nroots
        def eig_close_to_init_guess(w, v, nroots, envs):
            x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            snorm = np.einsum('pi,pi->i', s.conj(), s)
            idx = np.argsort(-snorm)[:nroots]
            return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)
        conv, es, vs = eig(matvec, guess, precond, pick=eig_close_to_init_guess,
                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                           max_space=eom.max_space, nroots=nroots, verbose=log)
    else:
        def pickeig(w, v, nroots, envs):
            real_idx = np.where(abs(w.imag) < 1e-3)[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
        conv, es, vs = eig(matvec, guess, precond, pick=pickeig,
                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                           max_space=eom.max_space, nroots=nroots, verbose=log)

    if eom.verbose >= logger.INFO:
        for n, en, vn, convn in zip(range(nroots), es, vs, conv):
            r1, r2 = eom.vector_to_amplitudes(vn)
            if isinstance(r1, np.ndarray):
                qp_weight = np.linalg.norm(r1)**2
            else: # for EOM-UCCSD
                r1 = np.hstack([x.ravel() for x in r1])
                qp_weight = np.linalg.norm(r1)**2
            logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, qp_weight, convn)
        log.timer('EOM-CCSD', *cput0)
    if nroots == 1:
        return conv[0], es[0].real, vs[0]
    else:
        return conv, es.real, vs


def biorthonormalize(r_ee,l_ee,eee_r,eee_l,nocc,nmo,build_M,norm,norm_l,full = False):
    deg_range = 0
    deg_start = 0
    deg = False
    IPnroots = len(r_ee)

    if full:
        M = build_M(r_ee,l_ee,nocc,nmo)
        lvecs = np.asarray(l_ee)
        rvecs = np.asarray(r_ee)

        from scipy.linalg import lu
        P,L,U = lu(M)
        lvecs = np.matmul(P.transpose(1,0),lvecs)
        inv = np.linalg.inv(L)
        lvecs = np.matmul(inv, lvecs)
        inv = np.linalg.inv(U)
        rvecs = np.matmul(rvecs.transpose(1,0),inv).transpose(1,0)

        for j in range(IPnroots):
            l_ee[j] = lvecs[j,:]
            r_ee[j] = rvecs[j,:]

    else:
        for i in range(IPnroots):
            if (i < IPnroots-1) and (round(eee_r[i],4) == round(eee_r[i+1],4)):
                deg_range += 1
                if not deg:
                    deg = True
                    deg_start = i
            else:
                if deg_range == 0:
                    r_ee[i] = norm(r_ee[i],nocc,nmo)
                    l_ee[i] = norm_l(r_ee[i],l_ee[i],nocc,nmo)
                else:
                    deg_range += 1
                    deg_end = deg_start+deg_range


                    M = build_M(r_ee[deg_start:deg_end],l_ee[deg_start:deg_end],nocc,nmo)
                    lvecs = np.asarray(l_ee[deg_start:deg_end])
                    rvecs = np.asarray(r_ee[deg_start:deg_end])

                    from scipy.linalg import lu
                    P,L,U = lu(M)
                    lvecs = np.matmul(P.transpose(1,0),lvecs)
                    inv = np.linalg.inv(L)
                    lvecs = np.matmul(inv, lvecs)
                    inv = np.linalg.inv(U)
                    rvecs = np.matmul(rvecs.transpose(1,0),inv).transpose(1,0)

                    for j in range(deg_range):
                        k = j + deg_start
                        l_ee[k] = lvecs[j,:]
                        r_ee[k] = rvecs[j,:]
                    deg = False
                    deg_range = 0
                    lvecs = None
                    rvecs = None

    return r_ee,l_ee

class EOM(lib.StreamObject):
    def __init__(self, cc):
        self.mol = cc.mol
        self._cc = cc
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.max_memory = cc.max_memory

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', cc.max_cycle)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', cc.conv_tol)
        self.partition = getattr(__config__, 'eom_rccsd_EOM_partition', None)

##################################################
# don't modify the following attributes, they are not input options
        self.e = None
        self.v = None
        self.nocc = cc.nocc
        self.nmo = cc.nmo
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'partition = %s', self.partition)
        #logger.info(self, 'nocc = %d', self.nocc)
        #logger.info(self, 'nmo = %d', self.nmo)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self


def _sort_left_right_eigensystem(eom, right_converged, right_evals, right_evecs,
                                 left_converged, left_evals, left_evecs, tol=1e-6):
    '''Ensures the left and right eigenvectors correspond to the same eigenvalue.

    Note:
        Useful for perturbative methods that need both eigenstates.  Right now, just
        simply checks for equality between left and right eigenvalues, but can be
        extended to make sure the overlap between states is sufficiently large.

    Kwargs:
        eom : :class:`EOM`
            Class holding EOM results.
        right_converged : array-like of bool
            Whether the right eigenstates converged.
        right_evals : array-like
            Eigenvalues of right eigenstates.
        right_evecs : array-like of ndarray
            Eigenvectors of right eigenstates.
        left_converged : array-like of bool
            Whether the left eigenstates converged.
        left_evals : array-like
            Eigenvalues of left eigenstates.
        left_evecs : array-like of ndarray
            Eigenvectors of left eigenstates.
        tol : float
            Tolerance for determining whether a left and right eigenvalue
            should be considered equal.
    '''
    log = logger.Logger(eom.stdout, eom.verbose)

    right_evecs, left_evecs = [np.atleast_2d(x) for x in [right_evecs, left_evecs]]
    right_evals, left_evals = [np.atleast_1d(x) for x in [right_evals, left_evals]]
    right_converged, left_converged = [np.atleast_1d(x) for x in [right_converged, left_converged]]

    srt_right_idx = []
    srt_left_idx = []
    left_idx = [idx for idx in range(len(left_evals)) if left_converged[idx]]
    right_idx = [idx for idx in range(len(right_evals)) if right_converged[idx]]
    if len(right_idx) != len(left_idx):
        log.warn('Number of converged left and right eigenvalues are not equal.\n'
                 '    No. Left = %3d, No. Right = %3d.' %
                 (len(left_idx), len(right_idx)))

    for ir_idx, ir in enumerate(right_idx):
        found = False
        for il_idx, il in enumerate(left_idx):
            if abs(right_evals[ir] - left_evals[il]) < tol:
                found = True
                srt_right_idx.append(ir)
                srt_left_idx.append(il)
                break
        if found:
            left_idx.pop(il_idx)
        else:
            log.warn('No converged left eigenvalue corresponding to right eigenvalue '
                     '%.6g (right idx=%3d).\nWill not perform perturbation on this state.'
                     % (right_evals[ir], ir))

    log.info('Resulting left/right eigenstates:')
    log.info('Left Eigen (idx) <-> Right Eigen (idx)')
    for il, ir in zip(srt_left_idx, srt_right_idx):
        log.info('%10.6g (%3d)      %10.6g (%3d)',
                 left_evals[il], il, right_evals[ir], ir)
    return (right_evals[srt_right_idx], right_evecs[srt_right_idx], left_evecs[srt_left_idx])


def perturbed_ccsd_kernel(eom, nroots=1, koopmans=False, right_guess=None,
            left_guess=None, eris=None, imds=None):
    '''Wrapper for running perturbative excited-states that require both left
    and right amplitudes.'''
    if imds is None:
        imds = eom.make_imds(eris=eris)

    # Right eigenvectors
    r_converged, r_e, r_v = \
               kernel(eom, nroots, koopmans=koopmans, guess=right_guess, left=False,
                      eris=eris, imds=imds)
    # Left eigenvectors
    l_converged, l_e, l_v = \
               kernel(eom, nroots, koopmans=koopmans, guess=right_guess, left=True,
                      eris=eris, imds=imds)

    e, r_v, l_v = _sort_left_right_eigensystem(eom, r_converged, r_e, r_v, l_converged, l_e, l_v)
    e_star = eom.ccsd_star_contract(e, r_v, l_v, imds=imds)
    return e_star


########################################
# EOM-IP-CCSD
########################################

def ipccsd(eom, nroots=1, left=False, koopmans=False, guess=None,
           partition=None, eris=None, imds=None):
    '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested
        partition : bool or str
            Use a matrix-partitioning for the doubles-doubles block.
            Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
            or 'full' (full diagonal elements).
        koopmans : bool
            Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
    '''
    if partition is not None:
        eom.partition = partition.lower()
        assert eom.partition in ['mp','full']
    eom.converged, eom.e, eom.v \
            = kernel(eom, nroots, koopmans, guess, left, eris=eris, imds=imds)
    return eom.e, eom.v

def ipccsd_star(eom, nroots=1, koopmans=False, right_guess=None,
                left_guess=None, eris=None, imds=None):
    """Calculates CCSD* perturbative correction.

    Simply calls the relevant `kernel()` function and `perturb_star` of the
    `eom` class.

    Returns:
        e_t_a_star (list of float):
            The IP-CCSD* energy.
    """
    return perturbed_ccsd_kernel(eom, nroots=nroots, koopmans=koopmans,
               right_guess=right_guess, left_guess=left_guess, eris=eris,
               imds=imds)

def vector_to_amplitudes_ip(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
    return r1, r2

def amplitudes_to_vector_ip(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    partition = None
    if hasattr(eom, 'partition'): partition = eom.partition
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ip(vector, nmo, nocc)

    # 1h-1h block
    Hr1 = -np.einsum('ki,k->i', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += 2*np.einsum('ld,ild->i', imds.Fov, r2)
    Hr1 +=  -np.einsum('kd,kid->i', imds.Fov, r2)
    Hr1 += -2*np.einsum('klid,kld->i', imds.Wooov, r2)
    Hr1 +=    np.einsum('lkid,kld->i', imds.Wooov, r2)

    # 2h1p-1h block
    Hr2 = -np.einsum('kbij,k->ijb', imds.Wovoo, r1)
    # 2h1p-2h1p block
    if partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += lib.einsum('bd,ijd->ijb', fvv, r2)
        Hr2 += -lib.einsum('ki,kjb->ijb', foo, r2)
        Hr2 += -lib.einsum('lj,ilb->ijb', foo, r2)
    elif partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ip(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += lib.einsum('bd,ijd->ijb', imds.Lvv, r2)
        Hr2 += -lib.einsum('ki,kjb->ijb', imds.Loo, r2)
        Hr2 += -lib.einsum('lj,ilb->ijb', imds.Loo, r2)
        Hr2 +=  lib.einsum('klij,klb->ijb', imds.Woooo, r2)
        Hr2 += 2*lib.einsum('lbdj,ild->ijb', imds.Wovvo, r2)
        Hr2 +=  -lib.einsum('kbdj,kid->ijb', imds.Wovvo, r2)
        Hr2 +=  -lib.einsum('lbjd,ild->ijb', imds.Wovov, r2) #typo in Ref
        Hr2 +=  -lib.einsum('kbid,kjd->ijb', imds.Wovov, r2)
        tmp = 2*np.einsum('lkdc,kld->c', imds.Woovv, r2)
        tmp += -np.einsum('kldc,kld->c', imds.Woovv, r2)
        Hr2 += -np.einsum('c,ijcb->ijb', tmp, imds.t2)

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def lipccsd_matvec(eom, vector, imds=None, diag=None):
    '''For left eigenvector'''
    # Note this is not the same left EA equations used by Nooijen and Bartlett.
    # Small changes were made so that the same type L2 basis was used for both the
    # left EA and left IP equations.  You will note more similarity for these
    # equations to the left IP equations than for the left EA equations by Nooijen.
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ip(vector, nmo, nocc)

    # 1h-1h block
    Hr1 = -np.einsum('ki,i->k', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += -np.einsum('kbij,ijb->k', imds.Wovoo, r2)

    # 2h1p-1h block
    Hr2 = -np.einsum('kd,l->kld', imds.Fov, r1)
    Hr2 += 2.*np.einsum('ld,k->kld', imds.Fov, r1)
    Hr2 += -np.einsum('klid,i->kld', 2.*imds.Wooov-imds.Wooov.transpose(1,0,2,3), r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += lib.einsum('bd,klb->kld', fvv, r2)
        Hr2 += -lib.einsum('ki,ild->kld', foo, r2)
        Hr2 += -lib.einsum('lj,kjd->kld', foo, r2)
    elif eom.partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ip(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += lib.einsum('bd,klb->kld', imds.Lvv, r2)
        Hr2 += -lib.einsum('ki,ild->kld', imds.Loo, r2)
        Hr2 += -lib.einsum('lj,kjd->kld', imds.Loo, r2)
        Hr2 += lib.einsum('lbdj,kjb->kld', 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2), r2)
        Hr2 += -lib.einsum('kbdj,ljb->kld', imds.Wovvo, r2)
        Hr2 += lib.einsum('klij,ijd->kld', imds.Woooo, r2)
        Hr2 += -lib.einsum('kbid,ilb->kld', imds.Wovov, r2)
        tmp = np.einsum('ijcb,ijb->c', imds.t2, r2)
        Hr2 += -np.einsum('lkdc,c->kld', 2.*imds.Woovv-imds.Woovv.transpose(1,0,2,3), tmp)

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    partition = None
    if hasattr(eom, 'partition'): partition = eom.partition
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = -np.diag(imds.Loo)
    Hr2 = np.zeros((nocc,nocc,nvir), dtype)
    for i in range(nocc):
        for j in range(nocc):
            for b in range(nvir):
                if partition == 'mp':
                    Hr2[i,j,b] += fvv[b,b]
                    Hr2[i,j,b] += -foo[i,i]
                    Hr2[i,j,b] += -foo[j,j]
                else:
                    Hr2[i,j,b] += imds.Lvv[b,b]
                    Hr2[i,j,b] += -imds.Loo[i,i]
                    Hr2[i,j,b] += -imds.Loo[j,j]
                    Hr2[i,j,b] +=  imds.Woooo[i,j,i,j]
                    Hr2[i,j,b] +=2*imds.Wovvo[j,b,b,j]
                    Hr2[i,j,b] += -imds.Wovvo[i,b,b,i]*(i==j)
                    Hr2[i,j,b] += -imds.Wovov[j,b,j,b]
                    Hr2[i,j,b] += -imds.Wovov[i,b,i,b]
                    Hr2[i,j,b] += -2*np.dot(imds.Woovv[j,i,b,:], t2[i,j,:,b])
                    Hr2[i,j,b] += np.dot(imds.Woovv[i,j,b,:], t2[i,j,:,b])

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector


def build_M_ip(self,r_ip,l_ip,nocc,nmo):

    return 2*np.inner(np.asarray(l_ip),np.asarray(r_ip))

def normalize_ip(self,r_ip,nocc,nmo):
    n = 2*np.dot(r_ip,r_ip)
    n = np.sqrt(n)
    r_ip /= n
    return r_ip

def normalize_l_ip(self,r_ip,l_ip,nocc,nmo):
    n = 2*np.dot(l_ip,r_ip)
    l_ip /= n
    return l_ip

def ipccsd_star_contract(eom, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, imds=None):
    from pyscf.cc.ccsd_t import _sort_eri, _sort_t2_vooo_
    cpu1 = cpu0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)
    if imds is None:
        imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    eris = imds.eris

    fock = eris.fock
    nocc, nvir = t1.shape
    nmo = nocc + nvir

    dtype = np.result_type(t1, t2, eris.ovoo.dtype)
    if eom._cc.incore_complete:
        ftmp = None
        eris_vvop = np.zeros((nvir,nvir,nocc,nmo), dtype)
    else:
        ftmp = lib.H5TmpFile()
        eris_vvop = ftmp.create_dataset('vvop', (nvir,nvir,nocc,nmo), dtype)

    orbsym = _sort_eri(eom._cc, eris, nocc, nvir, eris_vvop, log)
    mo_energy, t1T, t2T, vooo, fvo, restore_t2_inplace = \
            _sort_t2_vooo_(eom._cc, orbsym, t1, t2, eris)

    cpu1 = log.timer_debug1('CCSD(T) sort_eri', *cpu1)

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now)
    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc**3*6))))

    fock_mo_energy = np.asarray(eris.fock.diagonal())
    mo_e_occ = np.asarray(mo_energy[:nocc])
    mo_e_vir = np.asarray(mo_energy[nocc:])

    def contract_l2p(l1, l2, a0, a1, b0, b1, cache_vvop, out=None):
        '''Create perturbed l2.'''
        if out is None:
            out = np.zeros((nocc,)*3 + (a1-a0,b1-b0), dtype=dtype)
        out += 0.5*np.einsum('abij,k->ijkab', cache_vvop[:,:,:,:nocc].conj(), l1)
        out += lib.einsum('abie,jke->ijkab', cache_vvop[:,:,:,nocc:].conj(), l2)
        out += -lib.einsum('bjkm,ima->ijkab', vooo[b0:b1], l2[:,:,a0:a1])
        out += -lib.einsum('bjim,mka->ijkab', vooo[b0:b1], l2[:,:,a0:a1])
        return out

    def contract_pl2p(l1, l2, a0, a1, b0, b1, cache_vvop_a, cache_vvop_b):
        '''Create P(ia|jb) of perturbed l2.'''
        out = contract_l2p(l1, l2, a0, a1, b0, b1, cache_vvop_a)
        out += contract_l2p(l1, l2, b0, b1, a0, a1, cache_vvop_b).transpose(1,0,2,4,3)  # P(ia|jb)
        return out

    def contract_r2p(r1, r2, a0, a1, b0, b1, cache_vvop, out=None):
        '''Create perturbed r2.'''
        if out is None:
            out = np.zeros((nocc,)*3 + (a1-a0,b1-b0), dtype=dtype)
        tmp = np.einsum('mkbe,m->bke', eris.oovv[:,:,b0:b1,:], r1)
        out += -lib.einsum('bke,aeji->ijkab', tmp, t2T[a0:a1])
        tmp = np.einsum('mebj,m->bej', eris.ovvo[:,:,b0:b1,:], r1)
        out += -lib.einsum('bej,aeki->ijkab', tmp, t2T[a0:a1])
        tmp = np.einsum('mjnk,n->mjk', eris.oooo, r1)
        out += lib.einsum('mjk,abmi->ijkab', tmp, t2T[a0:a1,b0:b1])
        out += lib.einsum('abie,kje->ijkab', cache_vvop[:,:,:,nocc:], r2)
        out += -lib.einsum('bjkm,mia->ijkab', vooo[b0:b1].conj(), r2[:,:,a0:a1])
        out += -lib.einsum('bjim,kma->ijkab', vooo[b0:b1].conj(), r2[:,:,a0:a1])
        return out

    def contract_pr2p(r1, r2, a0, a1, b0, b1, cache_vvop_a, cache_vvop_b):
        '''Create P(ia|jb) of perturbed r2.'''
        out = contract_r2p(r1, r2, a0, a1, b0, b1, cache_vvop_a)
        out += contract_r2p(r1, r2, b0, b1, a0, a1, cache_vvop_b).transpose(1,0,2,4,3)  # P(ia|jb)
        return out

    ipccsd_evecs  = np.array(ipccsd_evecs)
    lipccsd_evecs = np.array(lipccsd_evecs)
    e = []
    ipccsd_evecs, lipccsd_evecs = [np.atleast_2d(x) for x in [ipccsd_evecs, lipccsd_evecs]]
    ipccsd_evals = np.atleast_1d(ipccsd_evals)
    for eval_, evec_, levec_ in zip(ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
        l1, l2 = eom.vector_to_amplitudes(levec_)
        r1, r2 = eom.vector_to_amplitudes(evec_)
        ldotr = np.dot(l1, r1) + np.dot(l2.ravel(), r2.ravel())
        l1 /= ldotr
        l2 /= ldotr
        l2 = 1./3*(l2 + 2.*l2.transpose(1,0,2))

        deltaE = 0.0
        eijk = (mo_e_occ[:,None,None,None,None] +
                mo_e_occ[None,:,None,None,None] +
                mo_e_occ[None,None,:,None,None] + eval_)
        for a0, a1 in lib.prange_tril(0, nvir, blksize):
            b0, b1 = 0, a1
            eijkab = (eijk - mo_e_vir[a0:a1][None,None,None,:,None] -
                      mo_e_vir[b0:b1][None,None,None,None,:])
            eijkab = 1./eijkab
            vvov_a = eris_vvop[a0:a1,b0:b1,:,:]
            vvov_b = eris_vvop[b0:b1,a0:a1,:,:]
            lijkab = contract_pl2p(l1, l2, a0, a1, b0, b1, vvov_a, vvov_b)
            rijkab = contract_pr2p(r1, r2, a0, a1, b0, b1, vvov_a, vvov_b)

            lijkab = 4.*lijkab \
                   - 2.*lijkab.transpose(1,0,2,3,4) \
                   - 2.*lijkab.transpose(2,1,0,3,4) \
                   - 2.*lijkab.transpose(0,2,1,3,4) \
                   + 1.*lijkab.transpose(1,2,0,3,4) \
                   + 1.*lijkab.transpose(2,0,1,3,4)

            # Symmetry factors (1 for a == b, 2 for a < b)
            fac = 2*np.ones_like(rijkab, dtype=int)
            triu_idx = np.triu_indices(a1-a0, a0+1, m=b1-b0)
            fac[:,:,:,triu_idx[0],triu_idx[1]] = 0
            fac[:,:,:,np.arange(a1-a0),np.arange(a0,b1)] = 1
            eijkab *= fac

            deltaE += np.einsum('ijkab,ijkab,ijkab', lijkab, rijkab, eijkab)
        deltaE = 0.5*deltaE.real
        logger.info(eom, "ipccsd energy, star energy, delta energy = %16.12f, %16.12f, %16.12f",
                    eval_, eval_+deltaE, deltaE)
        e.append(eval_+deltaE)
    t2 = restore_t2_inplace(t2T)
    return e

class EOMIP(EOM):
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(int(size), dtype)
                g[self.nocc-n-1] = 1.0
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype)
                g[i] = 1.0
                guess.append(g)
        return guess

    kernel = ipccsd
    ipccsd = ipccsd
    ipccsd_star = ipccsd_star

    matvec = ipccsd_matvec
    l_matvec = lipccsd_matvec
    get_diag = ipccsd_diag
    ccsd_star_contract = ipccsd_star_contract

    build_M = build_M_ip
    norm = normalize_ip
    norm_l = normalize_l_ip

    def ipccsd_star_contract(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, imds=None):
        return self.ccsd_star_contract(ipccsd_evals, ipccsd_evecs, lipccsd_evecs, imds=imds)

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        if left:
            matvec = lambda xs: [self.l_matvec(x, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ip(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*nocc*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ip(self.partition)
        return imds

    def gen_ip_fs(self,r_ip,l_ip):
        tdm_r = self.gen_ip_tdms(r_ip)
        tdm_l = self.gen_ip_tdms(l_ip,left = True)
        f = tdm_r*tdm_l

        return f

    def gen_ip_tdms(self,ip_vec,left=False,prange=None):
# prange and qrange allow the calculation of a subset of the TDM
        if not isinstance(ip_vec,list): ip_vec = [ip_vec]
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo-nocc
        nstates = len(ip_vec)
        if prange is None: prange = range(nmo)
        if isinstance(prange,int): prange = [prange]

        tdm = np.zeros((nstates,len(prange)))
        for p in prange:
            if left:
                e_vector = ip_tdm_l_imd(self,p).astype(ip_vec[0].dtype)
            else:
                e_vector = -ip_tdm_r_imd(self,p).astype(ip_vec[0].dtype)
            tdm[:,p] = np.einsum('i,vi->v', e_vector,ip_vec)

        tdm *= 2
        return tdm

    def biorthonormalize(self,r_ip,l_ip,e_r,e_l,full = False):
        if (len(r_ip) != len(l_ip)):
            print('The number of right and left hand eigenvectors must be the same in order to biorthonormalise them')
            return r_ip, l_ip
        if (np.any(np.round(e_r,4) != np.round(e_l,4))):
            print('The right and left hand eigenvalues do not agree to 4 decimal places. Proceed with care')
        nocc = self.nocc
        nmo = self.nmo

        return biorthonormalize(r_ip,l_ip,e_r,e_l,nocc,nmo,self.build_M,self.norm,self.norm_l,full=full)


    @property
    def eip(self):
        return self.e


class EOMIP_Ta(EOMIP):
    '''Class for EOM IPCCSD(T)*(a) method by Matthews and Stanton.'''
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_t3p2_ip(self._cc, self.partition)
        return imds

########################################
# EOM-EA-CCSD
########################################

def eaccsd(eom, nroots=1, left=False, koopmans=False, guess=None,
           partition=None, eris=None, imds=None):
    '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

    Args:
        See also ipccd()
    '''
    return ipccsd(eom, nroots, left, koopmans, guess, partition, eris, imds)

def eaccsd_star(eom, nroots=1, koopmans=False, right_guess=None,
        left_guess=None, eris=None, imds=None, **kwargs):
    """Calculates CCSD* perturbative correction.

    Args:
        See also ipccd_star()
    """
    return perturbed_ccsd_kernel(eom, nroots=nroots, koopmans=koopmans,
               right_guess=right_guess, left_guess=left_guess, eris=eris,
               imds=imds)

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
    return r1, r2

def amplitudes_to_vector_ea(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1995) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    partition = None
    if hasattr(eom, 'partition'): partition = eom.partition
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    r1, r2 = vector_to_amplitudes_ea(vector, nmo, nocc)

    # Eq. (37)
    # 1p-1p block
    Hr1 =  np.einsum('ac,c->a', imds.Lvv, r1)
    # 1p-2p1h block
    Hr1 += np.einsum('ld,lad->a', 2.*imds.Fov, r2)
    Hr1 += np.einsum('ld,lda->a',   -imds.Fov, r2)
    Hr1 += np.einsum('alcd,lcd->a', 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2), r2)
    # Eq. (38)
    # 2p1h-1p block
    Hr2 = np.einsum('abcj,c->jab', imds.Wvvvo, r1)
    # 2p1h-2p1h block
    if partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 +=  lib.einsum('ac,jcb->jab', fvv, r2)
        Hr2 +=  lib.einsum('bd,jad->jab', fvv, r2)
        Hr2 += -lib.einsum('lj,lab->jab', foo, r2)
    elif partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ea(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 +=  lib.einsum('ac,jcb->jab', imds.Lvv, r2)
        Hr2 +=  lib.einsum('bd,jad->jab', imds.Lvv, r2)
        Hr2 += -lib.einsum('lj,lab->jab', imds.Loo, r2)
        Hr2 += lib.einsum('lbdj,lad->jab', 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2), r2)
        Hr2 += -lib.einsum('lajc,lcb->jab', imds.Wovov, r2)
        Hr2 += -lib.einsum('lbcj,lca->jab', imds.Wovvo, r2)
        for a in range(nvir):
            Hr2[:,a,:] += lib.einsum('bcd,jcd->jb', imds.Wvvvv[a], r2)
        tmp = np.einsum('klcd,lcd->k', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2), r2)
        Hr2 += -np.einsum('k,kjab->jab', tmp, imds.t2)

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector

def leaccsd_matvec(eom, vector, imds=None, diag=None):
    # Note this is not the same left EA equations used by Nooijen and Bartlett.
    # Small changes were made so that the same type L2 basis was used for both the
    # left EA and left IP equations.  You will note more similarity for these
    # equations to the left IP equations than for the left EA equations by Nooijen.
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    r1, r2 = vector_to_amplitudes_ea(vector, nmo, nocc)

    # Eq. (30)
    # 1p-1p block
    Hr1 = np.einsum('ac,a->c', imds.Lvv, r1)
    # 1p-2p1h block
    Hr1 += np.einsum('abcj,jab->c', imds.Wvvvo, r2)
    # Eq. (31)
    # 2p1h-1p block
    Hr2 = 2.*np.einsum('c,ld->lcd', r1, imds.Fov)
    Hr2 +=  -np.einsum('d,lc->lcd', r1, imds.Fov)
    Hr2 += np.einsum('a,alcd->lcd', r1, 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2))
    # 2p1h-2p1h block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += lib.einsum('lad,ac->lcd', r2, fvv)
        Hr2 += lib.einsum('lcb,bd->lcd', r2, fvv)
        Hr2 += -lib.einsum('jcd,lj->lcd', r2, foo)
    elif eom.partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ea(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += lib.einsum('lad,ac->lcd', r2, imds.Lvv)
        Hr2 += lib.einsum('lcb,bd->lcd', r2, imds.Lvv)
        Hr2 += -lib.einsum('jcd,lj->lcd', r2, imds.Loo)
        Hr2 += lib.einsum('jcb,lbdj->lcd', r2, 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2))
        Hr2 += -lib.einsum('lajc,jab->lcb', imds.Wovov, r2)
        Hr2 += -lib.einsum('lbcj,jab->lca', imds.Wovvo, r2)
        for a in range(nvir):
            Hr2 += lib.einsum('lb,bcd->lcd', r2[:,a,:], imds.Wvvvv[a])
        tmp = np.einsum('ijcb,ibc->j', imds.t2, r2)
        Hr2 += -np.einsum('kjfe,j->kef', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2),tmp)

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    partition = None
    if hasattr(eom, 'partition'): partition = eom.partition
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape

    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = np.diag(imds.Lvv)
    Hr2 = np.zeros((nocc,nvir,nvir), dtype)
    for a in range(nvir):
        if eom.partition != 'mp':
            _Wvvvva = np.array(imds.Wvvvv[a])
        for b in range(nvir):
            for j in range(nocc):
                if eom.partition == 'mp':
                    Hr2[j,a,b] += fvv[a,a]
                    Hr2[j,a,b] += fvv[b,b]
                    Hr2[j,a,b] += -foo[j,j]
                else:
                    Hr2[j,a,b] += imds.Lvv[a,a]
                    Hr2[j,a,b] += imds.Lvv[b,b]
                    Hr2[j,a,b] += -imds.Loo[j,j]
                    Hr2[j,a,b] += 2*imds.Wovvo[j,b,b,j]
                    Hr2[j,a,b] += -imds.Wovov[j,b,j,b]
                    Hr2[j,a,b] += -imds.Wovov[j,a,j,a]
                    Hr2[j,a,b] += -imds.Wovvo[j,b,b,j]*(a==b)
                    Hr2[j,a,b] += _Wvvvva[b,a,b]
                    Hr2[j,a,b] += -2*np.dot(imds.Woovv[:,j,a,b], t2[:,j,a,b])
                    Hr2[j,a,b] += np.dot(imds.Woovv[:,j,b,a], t2[:,j,a,b])

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector

def eaccsd_star_contract(eom, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, imds=None):
    cpu1 = cpu0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)
    if imds is None:
        imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    eris = imds.eris

    fock = eris.fock
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    dtype = np.result_type(t1, t2, eris.ovoo.dtype)
    # Notice we do not use `sort_eri` as compared to the eaccsd_star.
    # The sort_eri does not produce eri's that are read-in quickly for the current contraction
    # scheme.  Here, we have that the block loop is over occupied indices whereas in the
    # sort_eri it is done over virtual indices (due to the permutation over occupied indices
    # in ipccsd_star versus virtual indices in eaccsd_star).
    cpu1 = log.timer_debug1('CCSD(T) sort_eri', *cpu1)  # Left if new sort_eri implemented

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*6))))

    mo_energy = np.asarray(eris.mo_energy)
    fock_mo_energy = np.asarray(eris.fock.diagonal())
    mo_e_occ = np.asarray(mo_energy[:nocc])
    mo_e_vir = np.asarray(mo_energy[nocc:])

    def contract_l2p(l1, l2, i0, i1, j0, j1, cache_ovvv_i, cache_ovvv_j, out=None):
        '''Create perturbed l2.'''
        if out is None:
            out = np.zeros((i1-i0,j1-j0) + (nvir,)*3, dtype=dtype)
        out += -0.5*np.einsum('iajb,c->ijabc', eris.ovov[i0:i1,:,j0:j1], l1)
        out += lib.einsum('iajm,mbc->ijabc', eris.ovoo[i0:i1,:,j0:j1], l2)
        out -= lib.einsum('iaeb,jec->ijabc', cache_ovvv_i, l2[j0:j1])
        out -= lib.einsum('jbec,iae->ijabc', cache_ovvv_j, l2[i0:i1])
        return out

    def contract_pl2p(l1, l2, i0, i1, j0, j1, cache_ovvv_i, cache_ovvv_j):
        '''Create P(ia|jb) of perturbed l2.'''
        out = contract_l2p(l1, l2, i0, i1, j0, j1, cache_ovvv_i, cache_ovvv_j)
        tmp = contract_l2p(l1, l2, j0, j1, i0, i1, cache_ovvv_j, cache_ovvv_i)
        tmp = tmp.transpose(1,0,3,2,4)  # P(ia|jb)
        out = out + tmp
        return out

    def _get_vvvv(eris):
        if eris.vvvv is None and getattr(eris, 'vvL', None) is not None:  # DF eris
            vvL = np.asarray(eris.vvL)
            nvir = int(np.sqrt(eris.vvL.shape[0]*2))
            return ao2mo.restore(1, lib.dot(vvL, vvL.T), nvir)
        elif len(eris.vvvv.shape) == 2:  # DO not use .ndim here for h5py library
                                         # backward compatbility
            nvir = int(np.sqrt(eris.vvvv.shape[0]*2))
            return ao2mo.restore(1, np.asarray(eris.vvvv), nvir)
        else:
            return eris.vvvv

    def contract_r2p(r1, r2, i0, i1, j0, j1, cache_ovvv_i, cache_ovvv_j, out=None):
        '''Create perturbed r2.'''
        if out is None:
            out = np.zeros((i1-i0,j1-j0) + (nvir,)*3, dtype=dtype)
        tmp = np.einsum('becf,f->bce', _get_vvvv(eris), r1)
        out += -lib.einsum('bce,ijae->ijabc', tmp, t2[i0:i1,j0:j1])
        tmp = np.einsum('mjce,e->mcj', eris.oovv[:,j0:j1], r1)
        out += lib.einsum('mcj,imab->ijabc', tmp, t2[i0:i1])
        tmp = np.einsum('jbem,e->mbj', eris.ovvo[j0:j1], r1)
        out += lib.einsum('mbj,imac->ijabc', tmp, t2[i0:i1])
        out += lib.einsum('iajm,mbc->ijabc', eris.ovoo[i0:i1,:,j0:j1].conj(), r2)
        out += -lib.einsum('iaeb,jec->ijabc', cache_ovvv_i.conj(), r2[j0:j1])
        out += -lib.einsum('jbec,iae->ijabc', cache_ovvv_j.conj(), r2[i0:i1])
        return out

    def contract_pr2p(r1, r2, i0, i1, j0, j1, cache_ovvv_i, cache_ovvv_j):
        '''Create P(ia|jb) of perturbed r2.'''
        out = contract_r2p(r1, r2, i0, i1, j0, j1, cache_ovvv_i, cache_ovvv_j)
        tmp = contract_r2p(r1, r2, j0, j1, i0, i1, cache_ovvv_j, cache_ovvv_i)
        tmp = tmp.transpose(1,0,3,2,4)  # P(ia|jb)
        out = out + tmp
        return out

    eaccsd_evecs  = np.array(eaccsd_evecs)
    leaccsd_evecs = np.array(leaccsd_evecs)
    e = []
    eaccsd_evecs, leaccsd_evecs = [np.atleast_2d(x) for x in [eaccsd_evecs, leaccsd_evecs]]
    eaccsd_evals = np.atleast_1d(eaccsd_evals)
    for eval_, evec_, levec_ in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
        l1, l2 = eom.vector_to_amplitudes(levec_)
        r1, r2 = eom.vector_to_amplitudes(evec_)
        ldotr = np.dot(l1, r1) + np.dot(l2.ravel(),r2.ravel())
        l1 /= ldotr
        l2 /= ldotr
        l2 = 1./3*(1.*l2 + 2.*l2.transpose(0,2,1))
        r2 = r2.transpose(0,2,1)

        deltaE = 0.0
        eabc = (mo_e_vir[None,None,:,None,None] +
                mo_e_vir[None,None,None,:,None] +
                mo_e_vir[None,None,None,None,:] - eval_)

        for i0, i1 in lib.prange_tril(0, nocc, blksize):
            j0, j1 = 0, i1
            eijabc = (mo_e_occ[i0:i1][:,None,None,None,None] +
                      mo_e_occ[j0:j1][None,:,None,None,None] - eabc)
            eijabc = 1./eijabc
            cache_ovvv_i = eris.get_ovvv(slice(i0,i1))
            cache_ovvv_j = eris.get_ovvv(slice(j0,j1))
            lijabc = contract_pl2p(l1, l2, i0, i1, j0, j1, cache_ovvv_i, cache_ovvv_j)
            rijabc = contract_pr2p(r1, r2, i0, i1, j0, j1, cache_ovvv_i, cache_ovvv_j)

            lijabc =  4.*lijabc \
                    - 2.*lijabc.transpose(0,1,3,2,4) \
                    - 2.*lijabc.transpose(0,1,4,3,2) \
                    - 2.*lijabc.transpose(0,1,2,4,3) \
                    + 1.*lijabc.transpose(0,1,3,4,2) \
                    + 1.*lijabc.transpose(0,1,4,2,3)

            # Symmetry factors (1 for a == b, 2 for a < b)
            fac = 2*np.ones_like(rijabc, dtype=int)
            triu_idx = np.triu_indices(i1-i0,i0+1,m=j1-j0)
            fac[triu_idx[0],triu_idx[1],:,:,:] = 0
            fac[np.arange(i1-i0),np.arange(i0,j1)] = 1
            eijabc *= fac

            deltaE += np.einsum('ijabc,ijabc,ijabc',lijabc,rijabc,eijabc)
        deltaE = 0.5*deltaE.real
        logger.info(eom, "eaccsd energy, star energy, delta energy = %16.12f, %16.12f, %16.12f",
                    eval_, eval_+deltaE, deltaE)
        e.append(eval_+deltaE)
    return e


class EOMEA(EOM):
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(size, dtype)
                g[n] = 1.0
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(size, dtype)
                g[i] = 1.0
                guess.append(g)
        return guess

    kernel = eaccsd
    eaccsd = eaccsd
    eaccsd_star = eaccsd_star

    matvec = eaccsd_matvec
    l_matvec = leaccsd_matvec
    get_diag = eaccsd_diag
    ccsd_star_contract = eaccsd_star_contract

    build_M = build_M_ip
    norm = normalize_ip
    norm_l = normalize_l_ip

    def eaccsd_star_contract(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, imds=None):
        return self.ccsd_star_contract(eaccsd_evals, eaccsd_evecs, leaccsd_evecs, imds=imds)

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        if left:
            matvec = lambda xs: [self.l_matvec(x, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ea(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nvir + nocc*nvir*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ea(self.partition)
        return imds

    def gen_ea_fs(self,r_ea,l_ea):
        tdm_r = self.gen_ea_tdms(r_ea)
        tdm_l = self.gen_ea_tdms(l_ea,left = True)
        f = tdm_r*tdm_l

        return f

    def gen_ea_tdms(self,ea_vec,left=False,prange=None):
# prange and qrange allow the calculation of a subset of the TDM
        if not isinstance(ea_vec,list): ea_vec = [ea_vec]
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo-nocc
        nstates = len(ea_vec)
        if prange is None: prange = range(nmo)
        if isinstance(prange,int): prange = [prange]

        tdm = np.zeros((nstates,len(prange)))
        for p in prange:
            if left:
                e_vector = ea_tdm_l_imd(self,p).astype(ea_vec[0].dtype)
            else:
                e_vector = -ea_tdm_r_imd(self,p).astype(ea_vec[0].dtype)
            tdm[:,p] = np.einsum('i,vi->v', e_vector,ea_vec)

        tdm *= 2
        return tdm

    def biorthonormalize(self,r_ea,l_ea,e_r,e_l,full = False):
        if (len(r_ea) != len(l_ea)):
            print('The number of right and left hand eigenvectors must be the same in order to biorthonormalise them')
            return r_ea, l_ea
        if (np.any(np.round(e_r,4) != np.round(e_l,4))):
            print('The right and left hand eigenvalues do not agree to 4 decimal places. Proceed with care')
        nocc = self.nocc
        nmo = self.nmo

        return biorthonormalize(r_ea,l_ea,e_r,e_l,nocc,nmo,self.build_M,self.norm,self.norm_l,full=full)

    @property
    def eea(self):
        return self.e


class EOMEA_Ta(EOMEA):
    '''Class for EOM EACCSD(T)*(a) method by Matthews and Stanton.'''
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_t3p2_ea(self._cc, self.partition)
        return imds

########################################
# EOM-EE-CCSD
########################################

#TODO: double spin-flip EOM-EE

def eeccsd(eom, nroots=1, koopmans=False, guess=None, eris=None, imds=None):
    '''Calculate N-electron neutral excitations via EOM-EE-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested
        koopmans : bool
            Calculate Koopmans'-like (1p1h) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
    '''
    if eris is None: eris = eom._cc.ao2mo()
    if imds is None: imds = eom.make_imds(eris)

    spinvec_size = eom.vector_size()
    nroots = min(nroots, spinvec_size)

    diag_eeS, diag_eeT, diag_sf = eom.get_diag(imds)
    guess_eeS = []
    guess_eeT = []
    guess_sf = []
    if guess:
        for g in guess:
            if g is None: # beta->alpha spin-flip excitation
                pass
            elif g.size == diag_eeS.size:
                guess_eeS.append(g)
            elif g.size == diag_eeT.size:
                guess_eeT.append(g)
            else:
                guess_sf.append(g)
        nroots_eeS = len(guess_eeS)
        nroots_eeT = len(guess_eeT)
        nroots_sf = len(guess_sf)
        if len(guess) != nroots:
            logger.warn(eom, 'Number of states in initial guess %d does not '
                        'equal to nroots %d.', len(guess), nroots)
    else:
        deeS = np.sort(diag_eeS)[:nroots]
        deeT = np.sort(diag_eeT)[:nroots]
        dsf = np.sort(diag_sf)[:nroots]
        dmax = np.sort(np.hstack([deeS,deeT,dsf,dsf]))[nroots-1]
        nroots_eeS = np.count_nonzero(deeS <= dmax)
        nroots_eeT = np.count_nonzero(deeT <= dmax)
        nroots_sf = np.count_nonzero(dsf <= dmax)
        guess_eeS = guess_eeT = guess_sf = None

    def eomee_sub(cls, nroots, guess, diag):
        ee_sub = cls(eom._cc)
        ee_sub.__dict__.update(eom.__dict__)
        e, v = ee_sub.kernel(nroots, koopmans, guess, eris, imds, diag=diag)
        if nroots == 1:
            e, v = [e], [v]
            ee_sub.converged = [ee_sub.converged]
        return list(ee_sub.converged), list(e), list(v)

    e0 = e1 = e2 = []
    v0 = v1 = v2 = []
    conv0 = conv1 = conv2 = []
    if nroots_eeS > 0:
        conv0, e0, v0 = eomee_sub(EOMEESinglet, nroots_eeS, guess_eeS, diag_eeS)
    if nroots_eeT > 0:
        conv2, e2, v2 = eomee_sub(EOMEETriplet, nroots_eeT, guess_eeT, diag_eeT)
    if nroots_sf > 0:
        conv1, e1, v1 = eomee_sub(EOMEESpinFlip, nroots_sf, guess_sf, diag_sf)
        # The associated solutions of beta->alpha excitations
        e1 = e1 + e1
        conv1 = conv1 + conv1
        v1 = v1 + [None] * len(v1)
# beta->alpha spin-flip excitations, the coefficients are (-r1, (-r2[0], r2[1]))
# as below.  The EOMEESpinFlip class only handles alpha->beta excitations.
# Setting beta->alpha to None to bypass the vectors in initial guess
        #for i in range(nroots_sf):
        #    r1, r2 = vector_to_amplitudes_eomsf(v1[i], eom.nmo, eom.nocc)
        #    v1.append(amplitudes_to_vector_eomsf(-r1, (-r2[0], r2[1])))

    e = np.hstack([e0,e2,e1])
    idx = e.argsort()
    e = e[idx]
    conv = conv0 + conv2 + conv1
    conv = [conv[x] for x in idx]
    v = v0 + v2 + v1
    v = [v[x] for x in idx]

    if nroots == 1:
        conv = conv[0]
        e = e[0]
        v = v[0]
    eom.converged = conv
    eom.e = e
    eom.v = v
    return eom.e, eom.v


def eomee_ccsd_singlet(eom, nroots=1, left=False, koopmans=False, guess=None,
                       eris=None, imds=None, diag=None):
    '''EOM-EE-CCSD singlet
    '''
    eom.converged, eom.e, eom.v \
            = kernel(eom, nroots, koopmans, guess, left, eris=eris, imds=imds, diag=diag)
    return eom.e, eom.v

def eomee_ccsd_triplet(eom, nroots=1, left=False, koopmans=False, guess=None,
                       eris=None, imds=None, diag=None):
    '''EOM-EE-CCSD triplet
    '''
    return eomee_ccsd_singlet(eom, nroots, left, koopmans, guess, eris, imds, diag)

def eomsf_ccsd(eom, nroots=1, koopmans=False, guess=None,
               eris=None, imds=None, diag=None):
    '''Spin flip EOM-EE-CCSD
    '''
    return eomee_ccsd_singlet(eom, nroots, koopmans, guess, eris, imds, diag)

vector_to_amplitudes_ee = vector_to_amplitudes_singlet = ccsd.vector_to_amplitudes
amplitudes_to_vector_ee = amplitudes_to_vector_singlet = ccsd.amplitudes_to_vector

def amplitudes_to_vector_eomsf(t1, t2, out=None):
    nocc, nvir = t1.shape
    t2baaa, t2aaba = t2
    otril = np.tril_indices(nocc, k=-1)
    vtril = np.tril_indices(nvir, k=-1)
    baaa = np.take(t2baaa.reshape(nocc*nocc,nvir*nvir),
                   vtril[0]*nvir+vtril[1], axis=1)
    vector = np.hstack((t1.ravel(), baaa.ravel(), t2aaba[otril].ravel()))
    return vector

def vector_to_amplitudes_eomsf(vector, nmo, nocc):
    nvir = nmo - nocc
    t1 = vector[:nocc*nvir].reshape(nocc,nvir).copy()
    pvec = vector[t1.size:]

    nbaaa = nocc*nocc*nvir*(nvir-1)//2
    naaba = nocc*(nocc-1)//2*nvir*nvir
    t2baaa = np.zeros((nocc*nocc,nvir*nvir), dtype=vector.dtype)
    t2aaba = np.zeros((nocc*nocc,nvir*nvir), dtype=vector.dtype)
    otril = np.tril_indices(nocc, k=-1)
    vtril = np.tril_indices(nvir, k=-1)

    v = pvec[:nbaaa].reshape(nocc*nocc,nvir*(nvir-1)//2)
    t2baaa[:,vtril[0]*nvir+vtril[1]] = v
    t2baaa[:,vtril[1]*nvir+vtril[0]] = -v

    v = pvec[nbaaa:nbaaa+naaba].reshape(-1,nvir*nvir)
    t2aaba[otril[0]*nocc+otril[1]] = v
    t2aaba[otril[1]*nocc+otril[0]] = -v

    t2baaa = t2baaa.reshape(nocc,nocc,nvir,nvir)
    t2aaba = t2aaba.reshape(nocc,nocc,nvir,nvir)
    return t1, (t2baaa, t2aaba)

def amplitudes_to_vector_triplet(t1, t2, out=None):
    t2aa, t2ab = t2
    dtype = np.result_type(t1, t2aa, t2ab)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size1 = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    size = size1 + nov*(nov-1)//2
    vector = np.ndarray(size, dtype, buffer=out)
    ccsd.amplitudes_to_vector_s4(t1, t2[0], out=vector)
    t2ab = t2[1].transpose(0,2,1,3).reshape(nov,nov)
    #lib.pack_tril(t2ab, out=vector[size1:])
    vector[size1:] = t2ab[np.tril_indices(nov, k = -1)]
    return vector

def vector_to_amplitudes_triplet(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    size1 = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    size = size1 + nov*(nov+1)//2
    t1, t2aa = ccsd.vector_to_amplitudes_s4(vector[:size1], nmo, nocc)
    #t2ab = lib.unpack_tril(vector[size1:size], filltriu=2)
    t2ab = np.zeros((nov,nov))
    idx = np.tril_indices(nov,k = -1)
    for ij,(i,j) in enumerate(zip(*idx)):
        t2ab[i,j] = vector[size1+ij]
        t2ab[j,i] =-vector[size1+ij].conj()

    t2ab = t2ab.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3).copy()
    return t1, (t2aa, t2ab)

def eeccsd_matvec_singlet(eom, vector, imds=None):
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc

    r1, r2 = vector_to_amplitudes_singlet(vector, nmo, nocc)
    t1, t2, eris = imds.t1, imds.t2, imds.eris
    nocc, nvir = t1.shape

    Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += np.einsum('me,imae->ia',imds.Fov, r2) * 2
    Hr1 -= np.einsum('me,imea->ia',imds.Fov, r2)

    #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    #:Hr2 += lib.einsum('ijef,aebf->ijab', tau2, eris_vvvv) * .5
    tau2 = _make_tau(r2, r1, t1, fac=2)
    #Hr2 = eom._cc._add_vvvv(None, tau2, eris, with_ovvv=False, t2sym='jiba')
    eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    Hr2 = lib.einsum('ijef,aebf->ijab', tau2, eris_vvvv)# * .5

    woOoO = np.asarray(imds.woOoO)
    Hr2 += lib.einsum('mnij,mnab->ijab', woOoO, r2)
    Hr2 *= .5
    woOoO = None

    Hr2 += lib.einsum('be,ijae->ijab', imds.Fvv   , r2)
    Hr2 -= lib.einsum('mj,imab->ijab', imds.Foo   , r2)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now - Hr2.size*8e-6)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        theta = r2[p0:p1] * 2 - r2[p0:p1].transpose(0,1,3,2)
        Hr1 += lib.einsum('mfae,mife->ia', ovvv, theta)
        theta = None
        tmp = lib.einsum('meaf,ijef->maij', ovvv, tau2)
        Hr2 -= lib.einsum('ma,mbij->ijab', t1[p0:p1], tmp)
        tmp  = lib.einsum('meaf,me->af', ovvv, r1[p0:p1]) * 2
        tmp -= lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
        Hr2 += lib.einsum('af,ijfb->ijab', tmp, t2)
        ovvv = tmp = None
    tau2 = None
    Hr2 -= lib.einsum('mbij,ma->ijab', imds.woVoO, r1)

    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc*nvir**2*2))))
    for p0, p1 in lib.prange(0, nvir, nocc):
        Hr2 += lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])

    woVVo = np.asarray(imds.woVVo)
    tmp = lib.einsum('mbej,imea->jiab', woVVo, r2)
    Hr2 += tmp
    tmp *= .5
    Hr2 += tmp.transpose(0,1,3,2)
    tmp = None

    woVvO = woVVo * .5
    woVVo = None
    woVvO += np.asarray(imds.woVvO)
    theta = r2*2 - r2.transpose(0,1,3,2)
    Hr1 += np.einsum('maei,me->ia', woVvO, r1) * 2
    Hr2 += lib.einsum('mbej,imae->ijab', woVvO, theta)
    woVvO = None

    woOoV = np.asarray(imds.woOoV)
    Hr1-= lib.einsum('mnie,mnae->ia', woOoV, theta)
    tmp = lib.einsum('nmie,me->ni', woOoV, r1) * 2
    tmp-= lib.einsum('mnie,me->ni', woOoV, r1)
    Hr2 -= lib.einsum('ni,njab->ijab', tmp, t2)
    tmp = woOoV = None

    eris_ovov = np.asarray(eris.ovov)
    tmp  = np.einsum('mfne,mf->en', eris_ovov, r1) * 2
    tmp -= np.einsum('menf,mf->en', eris_ovov, r1)
    tmp  = np.einsum('en,nb->eb', tmp, t1)
    tmp += lib.einsum('menf,mnbf->eb', eris_ovov, theta)
    Hr2 -= lib.einsum('eb,ijea->jiab', tmp, t2)
    tmp = None

    tmp = lib.einsum('nemf,imef->ni', eris_ovov, theta)
    Hr1 -= lib.einsum('na,ni->ia', t1, tmp)
    Hr2 -= lib.einsum('mj,miab->ijba', tmp, t2)
    tmp = theta = None

    tau2 = _make_tau(r2, r1, t1, fac=2)
    tmp = lib.einsum('menf,ijef->mnij', eris_ovov, tau2)
    tau2 = None

    tau = _make_tau(t2, t1, t1)
    tau *= .5
    Hr2 += lib.einsum('mnij,mnab->ijab', tmp, tau)
    tau = tmp = eris_ovov = None

    Hr2 = Hr2 + Hr2.transpose(1,0,3,2)
    vector = amplitudes_to_vector_ee(Hr1, Hr2)
    return vector


def leeccsd_matvec_singlet(self, vector, imds = None):
    if imds is None: imds = self.make_imds()

    if not imds.made_lee_imds:
        imds.make_lee()

    l1, l2 = self.vector_to_amplitudes(vector)
    t1, t2, eris = imds.t1, imds.t2, imds.eris
    nocc, nvir = t1.shape

    for i in range(nocc):
        for j in range(nvir):
            l2[i,i,j,j] *= 2.0 # Unclear why this term is needed

    Hl1  = lib.einsum('ae,ia->ie', imds.Fvv, l1)
    Hl1 -= lib.einsum('mi,ia->ma', imds.Foo, l1)

    Hl2 = 2*lib.einsum('me,ia->imae',imds.Fov, l1)
    Hl2 -= lib.einsum('me,ia->miae',imds.Fov, l1)

    Hl2+= lib.einsum('mnij,ijab->mnab', imds.woOoO, l2) * .5
    Hl2+= lib.einsum('be,ijab->ijae', imds.Fvv   , l2)
    Hl2-= lib.einsum('mj,ijab->imab', imds.Foo   , l2)

    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
    #:Hl1 += lib.einsum('mfae,imef->ia', eris_ovvv, rho)
    #:tmp = lib.einsum('meaf,ijef->maij', eris_ovvv, tau2)
    #:Hl2 -= lib.einsum('ma,mbij->ijab', t1, tmp)
    #:tmp  = lib.einsum('meaf,me->af', eris_ovvv, l1) * 2
    #:tmp -= lib.einsum('mfae,me->af', eris_ovvv, l1)
    mem_now = lib.current_memory()[0]
    max_memory = lib.param.MAX_MEMORY - mem_now
    blksize = max(int(max_memory*1e6/8/(nvir**3*3)), 2)
    tau2 = _make_tau(l2, l1*0.0, t1, fac=2)

    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1)
        ovvv = lib.unpack_tril(ovvv).reshape(-1,nvir,nvir,nvir)
        Hl2 += 2*lib.einsum('mfae,ia->imef', ovvv, l1[p0:p1])
        Hl2 -= lib.einsum('mfae,ia->imfe', ovvv, l1[p0:p1])

        tmp = lib.einsum('ma,ijab->mbij', t1[p0:p1], tau2)
        Hl2 -= lib.einsum('meaf,maij->ijef', ovvv, tmp)

        tmp = lib.einsum('ijab,ijfb->af', l2[:,p0:p1], t2)
        Hl1 += lib.einsum('meaf,af->me', ovvv, tmp) * 2
        Hl1 -= lib.einsum('mfae,af->me', ovvv, tmp)

        ovvv = tmp = None

    Hl1 -= lib.einsum('mbij,ijab->ma', imds.woVoO, l2)

    Hl2-= 2*lib.einsum('mnie,ia->mnae', imds.woOoV, l1)
    Hl2+= lib.einsum('mnie,ia->mnea', imds.woOoV, l1)

    tmp = lib.einsum('ijab,njab->ni', l2, t2)
    Hl1-= lib.einsum('nmie,ni->me', imds.woOoV, tmp) * 2
    Hl1+= lib.einsum('mnie,ni->me', imds.woOoV, tmp)
    tmp = None

    for p0, p1 in lib.prange(0, nvir, nocc):
        Hl1 += lib.einsum('cdak,ikcd->ia', np.asarray(imds.wvVvO[p0:p1,:,:,:]), l2[:,:,p0:p1,:]) # wvVvO needs creating specially; done by make_lee(), which is not efficient

    oVVo = np.asarray(imds.woVVo)
    Hl2 += lib.einsum('mbej,jiab->imea', oVVo, l2)
    Hl2 += lib.einsum('mbej,jiba->imea', oVVo, l2) * .5
    oVvO = np.asarray(imds.woVvO) + oVVo * .5
    oVVo = tmp = None
    Hl1 += lib.einsum('maei,ia->me', oVvO, l1) * 2
    Hl2 += 2*lib.einsum('mbej,ijab->imae', oVvO, l2)
    Hl2 -= lib.einsum('mbej,ijab->imea', oVvO, l2)
    oVvO = None

    eris_ovov = np.asarray(eris.ovov)
    tau2 = _make_tau(l2, l1*0.0, t1, fac=2)
    tau = _make_tau(t2, t1, t1)
    tmp = lib.einsum('ijab,mnab->mnij', tau2, tau) * .5
    tau2 = None
    Hl2 += lib.einsum('menf,mnij->ijef', eris_ovov, tmp)
    tau = tmp = None

    tmp = lib.einsum('na,ia->ni', t1, l1)
    Hl2 -= 2*lib.einsum('nemf,ni->imef', eris_ovov, tmp)
    Hl2 += lib.einsum('nemf,ni->imfe', eris_ovov, tmp)

    tmp = lib.einsum('ijba,miab->mj', l2, t2)
    Hl2 -= 2*lib.einsum('nemf,ni->imef', eris_ovov, tmp)
    Hl2 += lib.einsum('nemf,ni->imfe', eris_ovov, tmp)
    tmp = None

    tmp  = lib.einsum('jiab,ijea->eb', l2, t2)
    tmp  = lib.einsum('eb,nb->en', tmp, t1)
    Hl1 -= lib.einsum('mfne,en->mf', eris_ovov, tmp) * 2
    Hl1 += lib.einsum('menf,en->mf', eris_ovov, tmp)

    tmp = lib.einsum('jiab,ijea->eb', l2, t2)
    Hl2 -= 2*lib.einsum('menf,eb->mnbf', eris_ovov, tmp)
    Hl2 += lib.einsum('menf,eb->mnfb', eris_ovov, tmp)
    tmp = eris_ovov = rho = None

    tau2 = _make_tau(l2, l1*0.0, t1, fac=2)
    eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv),t1.shape[1])
    Hl2 += lib.einsum('ijab,aebf->ijef', tau2, eris_vvvv) * .5
    #l_add_vvvv_(self, tau2, eris, Hl2)
    tau2 = None

    Hl2 = Hl2 + Hl2.transpose(1,0,3,2)

    for i in range(nocc):
        for j in range(nvir):
            Hl2[i,i,j,j] *= 0.5 # Unclear why this term is needed

    vector = self.amplitudes_to_vector(Hl1, Hl2)
    return vector

def build_M_ee(self,r_ee,l_ee,nocc,nmo):

    deg_range = len(r_ee)
    r1 = []
    r2 = []
    l1 = []
    l2 = []
    for j in range(deg_range):
        r1j,r2j = vector_to_amplitudes_ee(r_ee[j],nmo,nocc)
        r1.append(r1j)
        r2.append(r2j)
        l1j,l2j = vector_to_amplitudes_ee(l_ee[j],nmo,nocc)
        l1.append(l1j)
        l2.append(l2j)

    l1j = l2j = r1j = r2j = None
    l1 = np.asarray(l1)
    l2 = np.asarray(l2)
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)

    M = 2*np.einsum('xia,yia->xy',l1,r1)
    M += 2*np.einsum('xijab,yijab->xy',l2,r2)
    M -= np.einsum('xijab,yijba->xy',l2,r2)
    return M

def normalize_ee(self,r_ee,nocc,nmo):
    r1,r2 = vector_to_amplitudes_ee(r_ee,nmo,nocc)
    n = 2*np.sum(np.square(r1)) + 2*np.sum(np.square(r2)) - np.einsum('ijab,ijba',r2,r2)
    n = np.sqrt(n)
    r_ee /= n
    return r_ee

def normalize_l_ee(self,r_ee,l_ee,nocc,nmo):
    r1,r2 = vector_to_amplitudes_ee(r_ee,nmo,nocc)
    l1,l2 = vector_to_amplitudes_ee(l_ee,nmo,nocc)
    n = 2*np.einsum('ia,ia',l1,r1)
    n += 2*np.einsum('ijab,ijab',l2,r2)
    n -= np.einsum('ijab,ijba',l2,r2)
    l_ee /= n
    return l_ee


def tdm_r_imd(eom,q,s):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc,nvir),dtype=cc.t1.dtype)
    vector2 = np.zeros((nocc,nocc,nvir,nvir),dtype=cc.t1.dtype)
    if q < nocc:
        if s < nocc:
            vector1[q,:] -= cc.l1[s,:]
            vector1 += np.einsum('iab,b->ia', cc.l2[s,:,:,:], cc.t1[q,:])
            vector1 -= 2*np.einsum('iab,b->ia', cc.l2[:,s,:,:], cc.t1[q,:])

            vector2[:,q,:,:] += cc.l2[s,:,:,:]
            vector2[q,:,:,:] += cc.l2[s,:,:,:].transpose(0,2,1)
            vector2[:,q,:,:] -= 2*cc.l2[s,:,:,:].transpose(0,2,1)
            vector2[q,:,:,:] -= 2*cc.l2[s,:,:,:]

            for j in range(nvir):
                vector2[q,q,j,j] *= 0.5

        else:
            sp = s-nocc

            vector1[q][sp] = 1.0 #1

            vector1[q,:] -= np.einsum('ia,i->a', cc.l1, cc.t1[:,sp])
            vector1[:,sp] -= np.einsum('ia,a->i', cc.l1, cc.t1[q,:])

            vector1 += 2*np.einsum('ijab,jb->ia', cc.l2, cc.t2[q,:,sp,:])
            vector1 -= np.einsum('ijab,jb->ia', cc.l2, cc.t2[q,:,:,sp])

            vector1[q,:] -= 2*np.einsum('ijab,ijb->a', cc.l2, cc.t2[:,:,sp,:])
            vector1[q,:] += np.einsum('ijab,ijb->a', cc.l2, cc.t2[:,:,:,sp])

            vector1[:,sp] += np.einsum('ijab,jab->i', cc.l2, cc.t2[:,q,:,:])
            vector1[:,sp] -= 2*np.einsum('ijab,jab->i', cc.l2, cc.t2[q,:,:,:])

            vector1 += np.einsum('ijab,a,j->ib', cc.l2, cc.t1[q,:],cc.t1[:,sp])
            vector1 -= 2*np.einsum('ijab,b,j->ia', cc.l2, cc.t1[q,:],cc.t1[:,sp])

            vector2[q,:,sp,:] += 2*cc.l1
            vector2[:,q,sp,:] -= cc.l1
            vector2[:,q,:,sp] += 2*cc.l1
            vector2[q,:,:,sp] -= cc.l1

            vector2[:,q,:,:] -= 2*np.einsum('ijab,j->iab', cc.l2, cc.t1[:,sp])
            vector2[q,:,:,:] += np.einsum('ijab,j->iab', cc.l2, cc.t1[:,sp])
            vector2[q,:,:,:] -= 2*np.einsum('ijab,j->iba', cc.l2, cc.t1[:,sp])
            vector2[:,q,:,:] += np.einsum('ijab,j->iba', cc.l2, cc.t1[:,sp])

            vector2[:,:,sp,:] += 2*np.einsum('ijab,b->jia', cc.l2, cc.t1[q,:])
            vector2[:,:,:,sp] -= np.einsum('ijab,b->jia', cc.l2, cc.t1[q,:])
            vector2[:,:,:,sp] += 2*np.einsum('ijab,b->ija', cc.l2, cc.t1[q,:])
            vector2[:,:,sp,:] -= np.einsum('ijab,b->ija', cc.l2, cc.t1[q,:])

            for i in range(nocc):
                for j in range(nvir):
                    vector2[i,i,j,j] *= 0.5

    else:
        qp = q-nocc
        if s < nocc:
            vector1 += 2*cc.l2[s,:,qp,:]
            vector1 -= cc.l2[s,:,:,qp]

        else:
            sp = s-nocc
            vector1[:,sp] += cc.l1[:,qp]
            vector1 += 2*np.einsum('ija,j->ia', cc.l2[:,:,:,qp], cc.t1[:,sp])
            vector1 -= np.einsum('ija,j->ia', cc.l2[:,:,qp,:], cc.t1[:,sp])

            vector2[:,:,sp,:] -= cc.l2[:,:,qp,:].transpose(1,0,2)
            vector2[:,:,:,sp] -= cc.l2[:,:,qp,:]
            vector2[:,:,sp,:] += 2*cc.l2[:,:,qp,:]
            vector2[:,:,:,sp] += 2*cc.l2[:,:,qp,:].transpose(1,0,2)

            for i in range(nocc):
                vector2[i,i,sp,sp] *= 0.5

    return amplitudes_to_vector_ee(vector1,vector2)


def tdm_r0_imd(eom,q,s):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    if q < nocc:
        if s < nocc:
            e0 = -np.dot(cc.t1[q,:],cc.l1[s,:])
            e0 -= 2*np.dot(cc.t2[q,:,:,:].reshape(-1),cc.l2[s,:,:,:].reshape(-1))
            e0 += np.dot(cc.t2[q,:,:,:].reshape(-1),cc.l2[:,s,:,:].reshape(-1))

        else:
            sp = s - nocc
            e0 = 2*np.dot(cc.l1.reshape(-1),cc.t2[q,:,sp,:].reshape(-1))
            e0 -= np.dot(cc.l1.reshape(-1),cc.t2[:,q,sp,:].reshape(-1))
            e0 -= np.dot(cc.l1.reshape(-1),np.einsum('i,a->ia',cc.t1[:,sp],cc.t1[q,:]).reshape(-1))

            tmp = np.einsum('ija,b->ijab',cc.t2[:,:,sp,:],cc.t1[q,:])
            tmp -= 2*np.einsum('ijb,a->ijab',cc.t2[:,:,sp,:],cc.t1[q,:])
            tmp -= np.einsum('iab,j->ijab',cc.t2[q,:,:,:],cc.t1[:,sp])
            tmp += 2*np.einsum('jab,i->ijab',cc.t2[q,:,:,:],cc.t1[:,sp])
            e0 += np.dot(cc.l2.reshape(-1),tmp.reshape(-1))

    else:
        qp = q - nocc
        if s < nocc:
            e0 = cc.l1[s,qp]
        else:
            sp = s - nocc
            e0 = np.dot(cc.t1[:,sp],cc.l1[:,qp])
            e0 += 2*np.dot(cc.t2[:,:,sp,:].reshape(-1),cc.l2[:,:,qp,:].reshape(-1))
            e0 -= np.dot(cc.t2[:,:,sp,:].reshape(-1),cc.l2[:,:,:,qp].reshape(-1))

    return e0


def get_Hrow(eom,eris=None):
    imds = _IMDS(eom._cc,eris=eris)
    imds._make_shared_1e()
    Hr1 = imds.Fov
    Hr2 = np.copy(imds.eris.ovov)
    eom.Hrow = eom.amplitudes_to_vector(Hr1,Hr2)

    return eom.amplitudes_to_vector(Hr1,Hr2)

def get_r0(eom,r_ee,eris=None):
    if not isinstance(r_ee,list): r_ee = [r_ee]
    nroots = len(r_ee)
    r0 = np.zeros(nroots, dtype=r_ee[0].dtype)
    Hrow = get_Hrow(eom,eris)
    Hr1, Hr2 = eom.vector_to_amplitudes(Hrow)
    Hr1 = Hr1.reshape(-1)
    Hr2 = Hr2.reshape(-1)
    for i in range(nroots):
        r1, r2 = eom.vector_to_amplitudes(r_ee[i])
        r1 = r1.reshape(-1)
        r2 = r2.reshape(-1)
        r0[i] = (np.dot(Hr1,r1) + np.dot(Hr2,r2))/eom.eee[i]

    return r0


def tdm_l_imd(eom,p,r):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc,nvir),dtype=cc.t1.dtype)
    vector2 = np.zeros((nocc,nocc,nvir,nvir),dtype=cc.t1.dtype)
    if p < nocc:
        if r < nocc:
            vector1[r,:] -= cc.t1[p,:]

            vector2[:,r,:,:] += cc.t2[p,:,:,:]
            vector2[r,:,:,:] += cc.t2[p,:,:,:].transpose(0,2,1)
            vector2[:,r,:,:] -= 2*cc.t2[p,:,:,:].transpose(0,2,1)
            vector2[r,:,:,:] -= 2*cc.t2[p,:,:,:]

            for j in range(nvir):
                vector2[p,p,j,j] *= 0.5

        else:
            rp = r-nocc

            vector1 += 2*cc.t2[p,:,rp,:]
            vector1 -= cc.t2[:,p,rp,:]
            vector1 -= np.einsum('i,a->ia', cc.t1[:,rp], cc.t1[p,:])

            vector2 -= 2*np.einsum('ijb,a->ijab', cc.t2[:,:,rp,:], cc.t1[p,:])
            vector2 -= np.einsum('iab,j->ijab', cc.t2[p,:,:,:], cc.t1[:,rp])
            vector2 -= 2*np.einsum('jib,a->ijba', cc.t2[:,:,rp,:], cc.t1[p,:])
            vector2 -= np.einsum('iba,j->jiab', cc.t2[p,:,:,:], cc.t1[:,rp])

            vector2 += np.einsum('ija,b->ijab', cc.t2[:,:,rp,:], cc.t1[p,:])
            vector2 += 2*np.einsum('jab,i->ijab', cc.t2[p,:,:,:], cc.t1[:,rp])
            vector2 += np.einsum('jia,b->ijba', cc.t2[:,:,rp,:], cc.t1[p,:])
            vector2 += 2*np.einsum('jba,i->jiab', cc.t2[p,:,:,:], cc.t1[:,rp])

            for i in range(nocc):
                for j in range(nvir):
                    vector2[i,i,j,j] *= 0.5

    else:
        pp = p-nocc
        if r < nocc:
            vector1[r,pp] += 1.0

        else:
            rp = r-nocc
            vector1[:,pp] += cc.t1[:,rp]

            vector2[:,:,pp,:] -= cc.t2[:,:,rp,:].transpose(1,0,2)
            vector2[:,:,:,pp] -= cc.t2[:,:,rp,:]
            vector2[:,:,pp,:] += 2*cc.t2[:,:,rp,:]
            vector2[:,:,:,pp] += 2*cc.t2[:,:,rp,:].transpose(1,0,2)

            for i in range(nocc):
                vector2[i,i,pp,pp] *= 0.5

    return amplitudes_to_vector_ee(vector1,vector2)


def tdm_l_imd_bvec(eom,p,r):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc,nvir),dtype=cc.t1.dtype)
    vector2 = np.zeros((nocc,nocc,nvir,nvir),dtype=cc.t1.dtype)
    if p < nocc:
        if r < nocc:
            vector1[r,:] -= cc.t1[p,:]

            vector2[:,r,:,:] += cc.t2[p,:,:,:]
            vector2[r,:,:,:] += cc.t2[p,:,:,:].transpose(0,2,1)
            vector2[:,r,:,:] -= 2*cc.t2[p,:,:,:].transpose(0,2,1)
            vector2[r,:,:,:] -= 2*cc.t2[p,:,:,:]

            # for j in range(nvir): # These terms are excluded as when the linear equations are solved
                # vector2[p,p,j,j] *= 0.5   #  double-counting is accounted for in the e-vector

        else:
            rp = r-nocc

            vector1 += 2*cc.t2[p,:,rp,:]
            vector1 -= cc.t2[:,p,rp,:]
            vector1 -= np.einsum('i,a->ia', cc.t1[:,rp], cc.t1[p,:])

            vector2 -= 2*np.einsum('ijb,a->ijab', cc.t2[:,:,rp,:], cc.t1[p,:])
            vector2 -= np.einsum('iab,j->ijab', cc.t2[p,:,:,:], cc.t1[:,rp])
            vector2 -= 2*np.einsum('jib,a->ijba', cc.t2[:,:,rp,:], cc.t1[p,:])
            vector2 -= np.einsum('iba,j->jiab', cc.t2[p,:,:,:], cc.t1[:,rp])

            vector2 += np.einsum('ija,b->ijab', cc.t2[:,:,rp,:], cc.t1[p,:])
            vector2 += 2*np.einsum('jab,i->ijab', cc.t2[p,:,:,:], cc.t1[:,rp])
            vector2 += np.einsum('jia,b->ijba', cc.t2[:,:,rp,:], cc.t1[p,:])
            vector2 += 2*np.einsum('jba,i->jiab', cc.t2[p,:,:,:], cc.t1[:,rp])

            # for i in range(nocc):   # These terms are excluded as when the linear equations are solved
                # for j in range(nvir):   #  double-counting is accounted for in the e-vector
                    # vector2[i,i,j,j] *= 0.5

    else:
        pp = p-nocc
        if r < nocc:
            vector1[r,pp] += 1.0

        else:
            rp = r-nocc
            vector1[:,pp] += cc.t1[:,rp]

            vector2[:,:,pp,:] -= cc.t2[:,:,rp,:].transpose(1,0,2)
            vector2[:,:,:,pp] -= cc.t2[:,:,rp,:]
            vector2[:,:,pp,:] += 2*cc.t2[:,:,rp,:]
            vector2[:,:,:,pp] += 2*cc.t2[:,:,rp,:].transpose(1,0,2)

            # for i in range(nocc):   # These terms are excluded as when the linear equations are solved 
                # vector2[i,i,pp,pp] *= 0.5    #  double-counting is accounted for in the e-vector

    return amplitudes_to_vector_ee(vector1,vector2)


def e_rdm_l_imd(eom,p,q,l1_ee,l2_ee):
    cc = eom._cc
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nocc,nvir),dtype=l1_ee.dtype)
    vector2 = np.zeros((nocc,nocc,nvir,nvir),dtype=l2_ee.dtype)
    if p < nocc:
        if q < nocc:
            vector1[p,:] -= l1_ee[q,:]

            vector1 += np.einsum('iab,a->ib', l2_ee[:,q,:,:], cc.t1[p,:])
            vector1 -= 2*np.einsum('iab,a->ib', l2_ee[q,:,:,:], cc.t1[p,:])

            vector2[:,p,:,:] += l2_ee[q,:,:,:]
            vector2[p,:,:,:] += l2_ee[q,:,:,:].transpose(0,2,1)
            vector2[:,p,:,:] -= 2*l2_ee[q,:,:,:].transpose(0,2,1)
            vector2[p,:,:,:] -= 2*l2_ee[q,:,:,:]

            for j in range(nvir):
                vector2[q,q,j,j] *= 0.5

        else:
            qp = q - nocc

            vector1[p,:] -= np.einsum('ia,i->a', l1_ee, cc.t1[:,qp])
            vector1[:,qp] -= np.einsum('ia,a->i', l1_ee, cc.t1[p,:])

            vector1 += 2*np.einsum('ijab,jb->ia', l2_ee, cc.t2[p,:,qp,:])
            vector1 -= np.einsum('ijab,jb->ia', l2_ee, cc.t2[p,:,:,qp])

            vector1[p,:] -= 2*np.einsum('ijab,ijb->a', l2_ee, cc.t2[:,:,qp,:])
            vector1[p,:] += np.einsum('ijab,ijb->a', l2_ee, cc.t2[:,:,:,qp])

            vector1[:,qp] += np.einsum('ijab,jab->i', l2_ee, cc.t2[:,p,:,:])
            vector1[:,qp] -= 2*np.einsum('ijab,jab->i', l2_ee, cc.t2[p,:,:,:])

            vector1 -= np.einsum('ijab,a,j->ib', l2_ee, cc.t1[p,:],cc.t1[:,qp])
            vector1 += 2*np.einsum('ijab,b,j->ia', l2_ee, cc.t1[p,:],cc.t1[:,qp])

            vector2[p,:,qp,:] += 2*l1_ee
            vector2[:,p,qp,:] -= l1_ee
            vector2[:,p,:,qp] += 2*l1_ee
            vector2[p,:,:,qp] -= l1_ee

            vector2[:,p,:,:] -= 2*np.einsum('ijab,j->iab', l2_ee, cc.t1[:,qp])
            vector2[p,:,:,:] += np.einsum('ijab,j->iab', l2_ee, cc.t1[:,qp])
            vector2[p,:,:,:] -= 2*np.einsum('ijab,j->iba', l2_ee, cc.t1[:,qp])
            vector2[:,p,:,:] += np.einsum('ijab,j->iba', l2_ee, cc.t1[:,qp])

            vector2[:,:,qp,:] += 2*np.einsum('ijab,b->jia', l2_ee, cc.t1[p,:])
            vector2[:,:,:,qp] -= np.einsum('ijab,b->jia', l2_ee, cc.t1[p,:])
            vector2[:,:,:,qp] += 2*np.einsum('ijab,b->ija', l2_ee, cc.t1[p,:])
            vector2[:,:,qp,:] -= np.einsum('ijab,b->ija', l2_ee, cc.t1[p,:])

            for i in range(nocc):
                for j in range(nvir):
                    vector2[i,i,j,j] *= 0.5

    else:
        pp = p - nocc
        if q < nocc:
            vector1 += 2*l2_ee[q,:,pp,:]
            vector1 -= l2_ee[q,:,:,pp]

        else:
            qp = q - nocc
            vector1[:,qp] += l1_ee[:,pp]
            vector1 += 2*np.einsum('ija,j->ia', l2_ee[:,:,:,pp], cc.t1[:,qp])
            vector1 -= np.einsum('ija,j->ia', l2_ee[:,:,pp,:], cc.t1[:,qp])

            vector2[:,:,qp,:] -= l2_ee[:,:,pp,:].transpose(1,0,2)
            vector2[:,:,:,qp] -= l2_ee[:,:,pp,:]
            vector2[:,:,qp,:] += 2*l2_ee[:,:,pp,:]
            vector2[:,:,:,qp] += 2*l2_ee[:,:,pp,:].transpose(1,0,2)

            for i in range(nocc):
                vector2[i,i,qp,qp] *= 0.5

    return cc.amplitudes_to_vector(vector1,vector2)

def e_rdm_l0_imd(eom,p,q,l1_ee,l2_ee):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    if p < nocc:
        if q < nocc:
            e0 = np.dot(cc.t1[p,:],l1_ee[q,:])

            e0 -= 2*np.dot(cc.t2[p,:,:,:].reshape(-1),l2_ee[q,:,:,:].reshape(-1))
            e0 += np.dot(cc.t2[p,:,:,:].reshape(-1),cc.l2[:,q,:,:].reshape(-1))

        else:
            qp = q - nocc

            e0 = 2*np.dot(l1_ee.reshape(-1),cc.t2[p,:,qp,:].reshape(-1))
            e0 -= np.dot(l1_ee.reshape(-1),cc.t2[:,p,qp,:].reshape(-1))

            e0 -= np.dot(l1_ee.reshape(-1),np.einsum('i,a->ia',cc.t1[:,qp],cc.t1[p,:]).reshape(-1))

            tmp = np.einsum('ija,b->ijab',cc.t2[:,:,qp,:],cc.t1[p,:])
            tmp -= 2*np.einsum('ijb,a->ijab',cc.t2[:,:,qp,:],cc.t1[p,:])
            tmp -= np.einsum('iab,j->ijab',cc.t2[p,:,:,:],cc.t1[:,qp])
            tmp += 2*np.einsum('jab,i->ijab',cc.t2[p,:,:,:],cc.t1[:,qp])
            e0 -= np.dot(l2_ee.reshape(-1),tmp.reshape(-1))
    else:
        pp = p - nocc
        if q < nocc:
            e0 = l1_ee[q,pp]
        else:
            qp = q - nocc

            e0 = np.dot(cc.t1[:,qp],l1_ee[:,pp])

            e0 += 2*np.dot(cc.t2[:,:,qp,:].reshape(-1),l2_ee[:,:,pp,:].reshape(-1))
            e0 -= np.dot(cc.t2[:,:,qp,:].reshape(-1),l2_ee[:,:,:,pp].reshape(-1))

    return e0

def ip_tdm_l_imd(eom,p):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=cc.t1.dtype)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=cc.t1.dtype)
    if p < nocc:
        vector1[ p ] = 1.0
    else:
        vector1 += cc.t1[:,p-nocc]
        vector2 += cc.t2[:,:,p-nocc,:]
    return amplitudes_to_vector_ip(vector1,vector2)


def ip_tdm_r_imd(eom,p):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=cc.t1.dtype)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=cc.t1.dtype)
    if p < nocc:
        vector1[ p ] = -1.0
        vector1 += np.einsum('ia,a->i', cc.l1, cc.t1[p,:])
        vector1 += 2*np.einsum('ilcd,lcd->i', cc.l2, cc.t2[p,:,:,:])
        vector1 -=   np.einsum('ilcd,ldc->i', cc.l2, cc.t2[p,:,:,:])

        vector2[p,:,:] += -2.*cc.l1
        vector2[:,p,:] += cc.l1
        vector2 += 2*np.einsum('c,ijcb->ijb', cc.t1[p,:], cc.l2)
        vector2 -=   np.einsum('c,jicb->ijb', cc.t1[p,:], cc.l2)
    else:
        vector1 += -cc.l1[:,p-nocc]
        vector2 += -2*cc.l2[:,:,p-nocc,:] + cc.l2[:,:,:,p-nocc]
    return amplitudes_to_vector_ip(vector1,vector2)


def ea_tdm_l_imd(eom,p):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        # Changed both to minus
        vector1 += -cc.t1[p,:]
        vector2 += -cc.t2[p,:,:,:]
    else:
        vector1[ p-nocc ] = 1.0
    return amplitudes_to_vector_ea(vector1,vector2)

def ea_tdm_r_imd(eom,p):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        # Changed both to plus
        vector1 += cc.l1[p,:]
        vector2 += (2*cc.l2[p,:,:,:] - cc.l2[:,p,:,:])
        pass
    else:
        vector1[ p-nocc ] = -1.0
        vector1 += np.einsum('ia,i->a', cc.l1, cc.t1[:,p-nocc])
        vector1 += 2*np.einsum('klca,klc->a', cc.l2, cc.t2[:,:,:,p-nocc])
        vector1 -=   np.einsum('klca,lkc->a', cc.l2, cc.t2[:,:,:,p-nocc])

        vector2[:,p-nocc,:] += -2.*cc.l1
        vector2[:,:,p-nocc] += cc.l1
        vector2 += 2*np.einsum('k,jkba->jab', cc.t1[:,p-nocc], cc.l2)
        vector2 -=   np.einsum('k,jkab->jab', cc.t1[:,p-nocc], cc.l2)
    return amplitudes_to_vector_ea(vector1,vector2)

def tppe_tdm_l_imd(eom,p,r1_ee,r2_ee):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    r1_ee = r1_ee.reshape((nocc, nvir))
    r2_ee = r2_ee.reshape((nocc,nocc, nvir, nvir))
    vector1 = np.zeros((nocc),dtype=r1_ee.dtype)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=r1_ee.dtype)
    if p < nocc:
        vector2[:,p,:] -= r1_ee[:,:]
        vector2[p,:,:] += 2*r1_ee[:,:]
    else:
        vector1 += r1_ee[:,p-nocc]

        vector2 += 2*r2_ee[:,:,p-nocc,:]
        vector2 -= r2_ee[:,:,p-nocc,:].transpose(1,0,2)

        vector2 += 2*np.einsum('i,ja->ija', cc.t1[:,p-nocc],r1_ee)
        vector2 -= np.einsum('j,ia->ija', cc.t1[:,p-nocc],r1_ee)

    return amplitudes_to_vector_ip(vector1,vector2)

def tppe_tdm_r_imd(eom,q,l1_ee,l2_ee):
    cc = eom._cc
    nocc, nvir = cc.t1.shape
    l1_ee = l1_ee.reshape((nocc, nvir))
    l2_ee = l2_ee.reshape((nocc,nocc, nvir, nvir))
    vector1 = np.zeros((nocc),dtype=l1_ee.dtype)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=l1_ee.dtype)
    if q < nocc:
        vector1 -= np.einsum('ia,a->i', l1_ee, cc.t1[q,:])
        vector1 -= 2*np.einsum('ijab,jab->i', l2_ee, cc.t2[q,:,:,:])
        vector1 +=   np.einsum('ijab,jba->i', l2_ee, cc.t2[q,:,:,:])

        vector2[q,:,:] += 2.*l1_ee
        vector2[:,q,:] -= l1_ee
        vector2 -= 2*np.einsum('ijab,a->ijb', l2_ee, cc.t1[q,:])
        vector2 +=   np.einsum('jiab,a->ijb', l2_ee, cc.t1[q,:])
    else:
        vector1 += l1_ee[:,q-nocc]
        vector2 += 2*l2_ee[:,:,q-nocc,:] - l2_ee[:,:,:,q-nocc]
    return amplitudes_to_vector_ip(vector1,vector2)

def eeccsd_matvec_triplet(eom, vector, imds=None):
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc

    r1, r2 = vector_to_amplitudes_triplet(vector, nmo, nocc)
    r2aa, r2ab = r2
    t1, t2, eris = imds.t1, imds.t2, imds.eris
    nocc, nvir = t1.shape

    Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += np.einsum('me,imae->ia',imds.Fov, r2aa)
    Hr1 += np.einsum('ME,iMaE->ia',imds.Fov, r2ab)

    tau2ab = np.einsum('ia,jb->ijab', r1, t1)
    tau2ab-= np.einsum('ia,jb->ijab', t1, r1)
    tau2ab+= r2ab
    tau2aa = np.einsum('ia,jb->ijab', r1, t1)
    tau2aa-= np.einsum('ia,jb->jiab', r1, t1)
    tau2aa = tau2aa - tau2aa.transpose(0,1,3,2)
    tau2aa+= r2aa

    #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    #:Hr2aa += lib.einsum('ijef,aebf->ijab', tau2aa, eris_vvvv) * .25
    #:Hr2ab += lib.einsum('ijef,aebf->ijab', tau2ab, eris_vvvv) * .5
    Hr2aa = eom._cc._add_vvvv(None, tau2aa, eris, with_ovvv=False, t2sym='jiba')
    Hr2ab = eom._cc._add_vvvv(None, tau2ab, eris, with_ovvv=False, t2sym='-jiba')

    woOoO = np.asarray(imds.woOoO)
    Hr2aa += lib.einsum('mnij,mnab->ijab', woOoO, r2aa)
    Hr2ab += lib.einsum('mNiJ,mNaB->iJaB', woOoO, r2ab)
    Hr2aa *= .25
    Hr2ab *= .5
    woOoO = None

    Hr2aa += lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2aa)
    Hr2aa -= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2aa)
    Hr2ab += lib.einsum('BE,iJaE->iJaB', imds.Fvv, r2ab)
    Hr2ab -= lib.einsum('MJ,iMaB->iJaB', imds.Foo, r2ab)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now - Hr2aa.size*8e-6)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    tmp1 = np.zeros((nvir,nvir), dtype=r1.dtype)
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        theta = r2aa[:,p0:p1] + r2ab[:,p0:p1]
        Hr1 += lib.einsum('mfae,imef->ia', ovvv, theta)
        theta = None
        tmpaa = lib.einsum('meaf,ijef->maij', ovvv, tau2aa)
        tmpab = lib.einsum('meAF,iJeF->mAiJ', ovvv, tau2ab)
        tmp1 += lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
        Hr2aa+= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmpaa)
        Hr2ab-= lib.einsum('mb,mAiJ->iJbA', t1[p0:p1], tmpab)
        ovvv = tmpaa = tmpab = None
    tau2aa = tau2ab = None

    woVVo = np.asarray(imds.woVVo)
    Hr1 += np.einsum('maei,me->ia', woVVo, r1)
    Hr2aa += lib.einsum('mbej,imae->ijba', woVVo, r2ab)
    Hr2ab += lib.einsum('MBEJ,iMEa->iJaB', woVVo, r2aa)
    Hr2ab += lib.einsum('MbeJ,iMeA->iJbA', woVVo, r2ab)

    woVVo = woVVo + np.asarray(imds.woVvO)
    theta = r2aa + r2ab
    tmp = lib.einsum('mbej,imae->ijab', woVVo, theta)
    woVVo = None

    woOoV = np.asarray(imds.woOoV)
    Hr1 -= lib.einsum('mnie,mnae->ia', woOoV, theta)
    tmpa = lib.einsum('mnie,me->ni', woOoV, r1)
    tmp += lib.einsum('ni,njab->ijab', tmpa, t2)
    tmp -= lib.einsum('af,ijfb->ijab', tmp1, t2)
    tmp -= lib.einsum('mbij,ma->ijab', imds.woVoO, r1)

    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc*nvir**2*2))))
    for p0,p1 in lib.prange(0, nvir, blksize):
        tmp += lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])

    Hr2aa += tmp
    Hr2ab += tmp
    tmp = woOoV = None

    eris_ovov = np.asarray(eris.ovov)
    tmpa = -lib.einsum('menf,imfe->ni', eris_ovov, theta)
    Hr1 += lib.einsum('na,ni->ia', t1, tmpa)
    tmp  = lib.einsum('mj,imab->ijab', tmpa, t2)
    tmp1 = np.einsum('menf,mf->en', eris_ovov, r1)
    tmpa = lib.einsum('en,nb->eb', tmp1, t1)
    tmpa-= lib.einsum('menf,mnbf->eb', eris_ovov, theta)
    tmp += lib.einsum('eb,ijae->ijab', tmpa, t2)
    Hr2aa += tmp
    Hr2ab -= tmp
    tmp = theta = tmp1 = tmpa = None

    tau2aa = np.einsum('ia,jb->ijab', r1, t1)
    tau2aa-= np.einsum('ia,jb->jiab', r1, t1)
    tau2aa = tau2aa - tau2aa.transpose(0,1,3,2)
    tau2aa+= r2aa
    tmpaa = lib.einsum('menf,ijef->mnij', eris_ovov, tau2aa)
    tau2aa = None
    tmpaa *= .25
    tau = _make_tau(t2, t1, t1)
    Hr2aa += lib.einsum('mnij,mnab->ijab', tmpaa, tau)
    tmpaa = tau = None

    tau2ab = np.einsum('ia,jb->ijab', r1, t1)
    tau2ab-= np.einsum('ia,jb->ijab', t1, r1)
    tau2ab+= r2ab
    tmpab = lib.einsum('meNF,iJeF->mNiJ', eris_ovov, tau2ab)
    tau2ab = None
    tmpab *= .5
    tau = _make_tau(t2, t1, t1)
    Hr2ab += lib.einsum('mNiJ,mNaB->iJaB', tmpab, tau)
    tmpab = tau = None
    eris_ovov = None

    Hr2aa = Hr2aa - Hr2aa.transpose(0,1,3,2)
    Hr2aa = Hr2aa - Hr2aa.transpose(1,0,2,3)
    Hr2ab = Hr2ab - Hr2ab.transpose(1,0,3,2)
    vector = amplitudes_to_vector_triplet(Hr1, (Hr2aa,Hr2ab))
    return vector


def leeccsd_matvec_triplet(self, vector, imds=None, diag=None):
    
#   NOTE: This is the same as in rccsd_slow, and not efficient

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(self.stdout, self.verbose)

    if not hasattr(self,'imds'):
        from pyscf.cc import rccsd_slow
        self.imds = rccsd_slow._IMDS(self._cc)
    if not self.imds.made_ee_imds:
        self.imds.make_ee()
    imds = self.imds

    nocc = self.nocc
    nmo = self.nmo
    nvir = nmo - nocc
    l1, l2 = vector_to_amplitudes_triplet(vector,nmo,nocc)
    l2aa, l2ab = l2[:]

    # t2bar_ijab = r_ijab - r_ijba
    t2_bar = imds.t2 - imds.t2.transpose(0,1,3,2)

    # Wbar_cjab = W_cjab - W_cjba
    wvovv_bar = imds.wvOvV - imds.wvOvV.transpose(0,1,3,2)
    # Wbar_ijkb = W_ijkb - W_ijbk
    wooov_bar = imds.woOoV - imds.woOoV.transpose(1,0,2,3)
    # Wbar_iabj = W_iabj - W_iajb
    wovvo_bar = imds.woVvO - imds.woVoV.transpose(0,1,3,2)

    # Wbar_iajb = W_iajb - W_iabj
    wovov_bar = imds.woVoV - imds.woVvO.transpose(0,1,3,2)
    # Wbar_ijab = W_ijab - W_ijba
    woovv_bar = imds.woOvV - imds.woOvV.transpose(1,0,2,3)
    # Wbar_abci = W_abci - W_baci
    wvvvo_bar = imds.wvVvO - imds.wvVvO.transpose(1,0,2,3)
    # Wbar_iajk = W_iajk - W_iakj
    wovoo_bar = imds.woVoO - imds.woVoO.transpose(0,1,3,2)
    # Wbar_abcd = W_abcd - W_abdc
    wvvvv_bar = imds.wvVvV - np.transpose(imds.wvVvV,(0,1,3,2))
    # Wbar_ijkl = W_ijkl - W_ijlk
    woooo_bar = imds.woOoO - imds.woOoO.transpose(0,1,3,2)

    # l_ia <- - F_ca l_ic
    Hl1 = np.einsum('ca,ic->ia', imds.Fvv, l1)
    # l_ia <- + F_ik l_ka
    Hl1 -= np.einsum('ik,ka->ia', imds.Foo, l1)

    # l_ia <-  W_ci[ka] l_kc
    Hl1 += np.einsum('icak,kc->ia', wovvo_bar, l1)
    # l_ia <-  W_CiKa l_KC
    Hl1 -= np.einsum('icak,kc->ia', imds.woVvO, l1)

    # l_ia <- 0.5 * W_ic[kl] l_klac
    Hl1 -= 0.5 * np.einsum('ickl,klac->ia', wovoo_bar, l2aa)
    # l_ia <- W_iCkL l_kLaC
    Hl1 -= np.einsum('ickl,klac->ia', imds.woVoO, l2ab)

    # l_ia <- 0.5 * W_cdak l_ikcd
    Hl1 += 0.5 * np.einsum('cdak,ikcd->ia', wvvvo_bar, l2aa)
    # l_ia <- W_cDaK l_iKcD
    Hl1 += np.einsum('cdak,ikcd->ia', imds.wvVvO, l2ab)

    # l_ia <- - 0.5 * W_id[ac] T_kl[ce] l_klde
    Hl1 += 0.5 * np.einsum('dica,klce,klde->ia', wvovv_bar, t2_bar, l2aa)
    # l_ia <- - W_id[ac] T_kLcE l_kLdE
    Hl1 += np.einsum('dica,klce,klde->ia', wvovv_bar, imds.t2, l2ab)
    # l_ia <- - 0.5 * W_iDaC T_KL[CE] l_KLDE
    Hl1 -= 0.5 * np.einsum('dica,klce,klde->ia', imds.wvOvV, t2_bar, l2aa)
    # # l_ia <- - W_iDaC T_KlCe l_KlDe
    Hl1 -= np.einsum('dica,klce,klde->ia', imds.wvOvV, imds.t2, l2ab)

    # l_ia <- 0.5 * W_il[ak] T_lm[cd] l_kmcd
    Hl1 -= 0.5 * np.einsum('lika,lmcd,kmcd->ia', wooov_bar, t2_bar, l2aa)
    # l_ia <- W_il[ak] T_lMcD l_kMcD
    Hl1 -= np.einsum('lika,lmcd,kmcd->ia', wooov_bar, imds.t2, l2ab)
    # l_ia <- 0.5 * W_iLaK T_LM[CD] l_KMCD
    Hl1 += 0.5 * np.einsum('lika,lmcd,kmcd->ia', imds.woOoV, t2_bar, l2aa)
    # l_ia <- W_iLaK T_LmCd l_KmCd
    Hl1 += np.einsum('lika,lmcd,kmcd->ia', imds.woOoV, imds.t2, l2ab)

#       ------------------------------------------

    Hl2aa = np.zeros_like(l2ab)

    # l_ijab <= F_ia l_jb
    Hl2aa += np.einsum('ia,jb->ijab', imds.Fov, l1)

    # l_ijab <= 0.5 * W_cj[ab] l_ic
    Hl2aa += 0.5 * np.einsum('cjab,ic->ijab', wvovv_bar, l1)
    # l_ijab <= - 0.5 * W_ij[kb] l_ka
    Hl2aa -= 0.5 * np.einsum('ijkb,ka->ijab', wooov_bar, l1)

    # l_ijab <= 0.5 * F_cb l_ijac
    Hl2aa += 0.5 * np.einsum('cb,ijac->ijab', imds.Fvv, l2aa)
    # l_ijab <= - 0.5 * F_jk l_ikab
    Hl2aa -= 0.5 * np.einsum('jk,ikab->ijab', imds.Foo, l2aa)

    # l_ijab <=  W_cj[kb] l_ikac
    Hl2aa += np.einsum('jcbk,ikac->ijab', wovvo_bar, l2aa)
    # l_ijab <=  W_CjKb l_iKaC
    Hl2aa += np.einsum('jcbk,ikac->ijab', imds.woVvO, l2ab)

    # l_ijab <= 0.125 * W_cd[ab] l_ijcd
    Hl2aa += 0.125 * np.einsum('cdab,ijcd->ijab', wvvvv_bar, l2aa)
    # l_ijab <= 0.125 * W_ij[kl] l_klab
    Hl2aa += 0.125 * np.einsum('ijkl,klab->ijab', woooo_bar, l2aa)

    # l_ijab <= - 0.25 * W_ik[ab] T_kl[cd] l_jlcd
    tmp = 0.25 * np.einsum('klcd,jlcd->kj', t2_bar, l2aa)
    Hl2aa -= np.einsum('ikab,kj->ijab', woovv_bar, tmp)
    # l_ijab <= - 0.5 * W_ik[ab] T_kLcD l_jLcD
    tmp = 0.5 * np.einsum('klcd,jlcd->kj', imds.t2, l2ab)
    Hl2aa -= np.einsum('ikab,kj->ijab', woovv_bar, tmp)
    # l_ijab <= - 0.25 * W_ij[ac] T_kl[cd] l_klbd
    tmp = 0.25 * np.einsum('klcd,klbd->cb', t2_bar, l2aa)
    Hl2aa -= np.einsum('ijac,cb->ijab', woovv_bar, tmp)
    # l_ijab <= - 0.5 * W_ij[ac] T_kLcD l_kLbD
    tmp = 0.5 * np.einsum('klcd,klbd->cb', imds.t2, l2ab)
    Hl2aa -= np.einsum('ijac,cb->ijab', woovv_bar, tmp)

#      ------------------------------------------

    Hl2ab = np.zeros_like(l2ab)

    # l_ijab <= F_ia l_JB
    Hl2ab -= np.einsum('ia,jb->ijab', imds.Fov, l1)

    # l_ijab <= W_cJaB l_ic
    Hl2ab += np.einsum('cjab,ic->ijab', imds.wvOvV, l1)
    # l_ijab <= - W_iJkB l_ka
    Hl2ab -= np.einsum('ijkb,ka->ijab', imds.woOoV, l1)

    # l_ijab <= F_CB l_iJaC
    Hl2ab += np.einsum('cb,ijac->ijab', imds.Fvv, l2ab)
    # l_ijab <= - F_JK l_iKaB
    Hl2ab -= np.einsum('jk,ikab->ijab', imds.Foo, l2ab)

    # l_ijab <=  W_CJ[KB] l_iKaC
    Hl2ab += np.einsum('jcbk,ikac->ijab', wovvo_bar, l2ab)
    Hl2ab -= np.einsum('jcbk,ikac->jiab', imds.woVoV.transpose(0,1,3,2), l2ab.transpose(1,0,2,3))  # This term is inferred from the right hand multiply
    # l_ijab <=  W_cJkB l_ikac
    Hl2ab += np.einsum('jcbk,ikac->ijab', imds.woVvO, l2aa)

    # l_ijab <= 0.5 * W_cDaB l_iJcD
    Hl2ab += 0.5 * np.einsum('cdab,ijcd->ijab', imds.wvVvV, l2ab)
    # l_ijab <= 0.125 * W_iJkL l_kLaB
    Hl2ab += 0.5 * np.einsum('ijkl,klab->ijab', imds.woOoO, l2ab)

    # l_ijab <= - W_iKaB T_klCd l_JlCd
    tmp = np.einsum('klcd,jlcd->kj', imds.t2, l2ab)
    Hl2ab += np.einsum('ikab,kj->ijab', imds.woOvV, tmp)
    # l_ijab <= - 0.5 * W_iKaB T_KL[CD] l_JLCD
    tmp = 0.5 * np.einsum('klcd,jlcd->kj', t2_bar, l2aa)
    Hl2ab += np.einsum('ikab,kj->ijab', imds.woOvV, tmp)
    # l_ijab <= - W_iJaC T_KlCd l_KlBd
    tmp = np.einsum('klcd,klbd->cb', imds.t2, l2ab)
    Hl2ab += np.einsum('ijac,cb->ijab', imds.woOvV, tmp)
    # l_ijab <= - 0.5 * W_ij[ac] T_kLcD l_kLbD
    tmp = 0.5 * np.einsum('klcd,klbd->cb', t2_bar, l2aa)
    Hl2ab += np.einsum('ijac,cb->ijab', imds.woOvV, tmp)

    Hl2aa = Hl2aa - Hl2aa.transpose(0,1,3,2)
    Hl2aa = Hl2aa - Hl2aa.transpose(1,0,2,3)
    Hl2ab = Hl2ab - Hl2ab.transpose(1,0,3,2)

    vector = amplitudes_to_vector_triplet(Hl1, (Hl2aa, Hl2ab))
    log.timer("matvec EOMEE Triplet", *cput0)
    return vector

def build_M_triplet(self,r_ee,l_ee,nocc,nmo):

    deg_range = len(r_ee)
    r1 = []
    r2 = []
    l1 = []
    l2 = []
    for j in range(deg_range):
        r1j,r2j = vector_to_amplitudes_triplet(r_ee[j],nmo,nocc)
        r1.append(r1j)
        r2.append(r2j)
        l1j,l2j = vector_to_amplitudes_triplet(l_ee[j],nmo,nocc)
        l1.append(l1j)
        l2.append(l2j)

    l1j = l2j = r1j = r2j = None
    l1 = np.asarray(l1)
    l2 = np.asarray(l2)
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)

    M = 2*np.einsum('xia,yia->xy',l1,r1)
    M += np.einsum('xijab,yijab->xy',l2[0],r2[0])
    M += np.einsum('xijab,yijab->xy',l2[1],r2[1])
    return M

def normalize_triplet(self,r_ee,nocc,nmo):
    r1,r2 = vector_to_amplitudes_triplet(r_ee,nmo,nocc)
    n = 2*np.sum(np.square(r1)) + np.sum(np.square(r2[0])) + np.sum(np.square(r2[1]))
    n = np.sqrt(n)
    r_ee /= n
    return r_ee

def normalize_l_triplet(self,r_ee,l_ee,nocc,nmo):
    r1,r2 = vector_to_amplitudes_triplet(r_ee,nmo,nocc)
    l1,l2 = vector_to_amplitudes_triplet(l_ee,nmo,nocc)
    n = 2*np.einsum('ia,ia',l1,r1)
    n += np.einsum('ijab,ijab',l2[0],r2[0])
    n += np.einsum('ijab,ijab',l2[1],r2[1])
    l_ee /= n
    return l_ee

def eeccsd_matvec_sf(eom, vector, imds=None):
    '''Spin flip EOM-CCSD'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc

    t1, t2, eris = imds.t1, imds.t2, imds.eris
    r1, r2 = vector_to_amplitudes_eomsf(vector, nmo, nocc)
    r2baaa, r2aaba = r2
    nocc, nvir = t1.shape

    Hr1  = np.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= np.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += np.einsum('me,imae->ia', imds.Fov, r2baaa)
    Hr1 += np.einsum('me,imae->ia', imds.Fov, r2aaba)

    tau2baaa = np.einsum('ia,jb->ijab', r1, t1)
    tau2baaa += r2baaa * .5
    tau2baaa = tau2baaa - tau2baaa.transpose(0,1,3,2)
    tau2aaba = np.einsum('ia,jb->ijab', r1, t1)
    tau2aaba += r2aaba * .5
    tau2aaba = tau2aaba - tau2aaba.transpose(1,0,2,3)

    #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    #:Hr2baaa += .5*lib.einsum('ijef,aebf->ijab', tau2baaa, eris_vvvv)
    #:Hr2aaba += .5*lib.einsum('ijef,aebf->ijab', tau2aaba, eris_vvvv)
    Hr2aaba = eom._cc._add_vvvv(None, tau2aaba, eris, with_ovvv=False, t2sym='-jiab')
    Hr2baaa = eom._cc._add_vvvv(None, tau2baaa, eris, with_ovvv=False, t2sym=False)

    woOoO = np.asarray(imds.woOoO)
    Hr2baaa += lib.einsum('mnij,mnab->ijab', woOoO, r2baaa)
    Hr2aaba += lib.einsum('mnij,mnab->ijab', woOoO, r2aaba)
    Hr2aaba *= .5
    Hr2baaa *= .5
    woOoO = None

    Hr2baaa -= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2baaa)
    Hr2aaba -= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2aaba)
    Hr2baaa -= lib.einsum('mj,miab->jiab', imds.Foo*.5, r2baaa)
    Hr2aaba -= lib.einsum('mj,miab->jiab', imds.Foo*.5, r2aaba)
    Hr2baaa += lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2baaa)
    Hr2aaba += lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2aaba)
    Hr2baaa += lib.einsum('be,ijea->ijba', imds.Fvv*.5, r2baaa)
    Hr2aaba += lib.einsum('be,ijea->ijba', imds.Fvv*.5, r2aaba)

    tmp1 = np.zeros((nvir,nvir), dtype=r1.dtype)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now - Hr2aaba.size*8e-6)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        theta = r2baaa[:,p0:p1] + r2aaba[:,p0:p1]
        Hr1 += lib.einsum('mfae,imef->ia', ovvv, theta)
        theta = None

        tmp1aaba = lib.einsum('meaf,ijef->maij', ovvv, tau2baaa)
        Hr2baaa -= lib.einsum('mb,maij->ijba', t1[p0:p1]*.5, tmp1aaba)
        tmp1aaba = tmp1baaa = tmp1abaa = tmp2aaba = None

        tmp2aaba = lib.einsum('meaf,ijfe->maij', ovvv, tau2baaa)
        Hr2baaa -= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmp2aaba)
        tmp1aaba = tmp1baaa = tmp1abaa = tmp2aaba = None

        tmp1baaa = lib.einsum('meaf,ijef->maij', ovvv, tau2aaba)
        Hr2aaba -= lib.einsum('mb,maij->ijba', t1[p0:p1]*.5, tmp1baaa)
        tmp1aaba = tmp1baaa = tmp1abaa = tmp2aaba = None

        tmp1abaa = lib.einsum('meaf,ijfe->maij', ovvv, tau2aaba)
        Hr2aaba -= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmp1abaa)
        tmp1aaba = tmp1baaa = tmp1abaa = tmp2aaba = None

        tmp = lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
        tmp = lib.einsum('af,jibf->ijab', tmp, t2)
        Hr2baaa -= tmp
        Hr2aaba -= tmp
        tpm = ovvv = None
    tau2aaba = tau2baaa = None

    tmp = lib.einsum('mbij,ma->ijab', imds.woVoO, r1)
    Hr2baaa -= tmp
    Hr2aaba -= tmp
    tmp = None

    woOoV = np.asarray(imds.woOoV)
    Hr1 -= lib.einsum('mnie,mnae->ia', woOoV, r2aaba)
    Hr1 -= lib.einsum('mnie,mnae->ia', woOoV, r2baaa)
    tmp = lib.einsum('mnie,me->ni', woOoV, r1)
    tmp = lib.einsum('ni,njab->ijab', tmp, t2)
    Hr2baaa += tmp
    Hr2aaba += tmp
    tmp = woOoV = None

    for p0,p1 in lib.prange(0, nvir, nocc):
        tmp = lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])
        Hr2baaa += tmp
        Hr2aaba += tmp
        tmp = None

    woVVo = np.asarray(imds.woVVo)
    Hr1 += np.einsum('maei,me->ia', woVVo, r1)
    Hr2baaa += lib.einsum('mbej,miea->jiba', woVVo, r2baaa)
    Hr2aaba += lib.einsum('mbej,miea->jiba', woVVo, r2aaba)
    woVVo = None
    woVvO = np.asarray(imds.woVvO)
    Hr2baaa += lib.einsum('mbej,imae->ijab', woVvO, r2aaba)
    Hr2aaba += lib.einsum('mbej,imae->ijab', woVvO, r2baaa)
    woVvO = woVvO + np.asarray(imds.woVVo)
    Hr2baaa += lib.einsum('mbej,imae->ijab', woVvO, r2baaa)
    Hr2aaba += lib.einsum('mbej,imae->ijab', woVvO, r2aaba)
    woVvO = None

    eris_ovov = np.asarray(eris.ovov)
    theta = r2aaba + r2baaa
    tmp = lib.einsum('nfme,imfe->ni', eris_ovov, theta)
    Hr1 -= np.einsum('na,ni->ia', t1, tmp)
    Hr2baaa -= lib.einsum('mj,imba->jiab', tmp, t2)
    Hr2aaba -= lib.einsum('mj,imba->jiab', tmp, t2)

    tmp = np.einsum('menf,mf->en', eris_ovov, r1)
    tmp = np.einsum('en,nb->eb', tmp, t1)
    tmp-= lib.einsum('menf,mnbf->eb', eris_ovov, theta)
    Hr2baaa += lib.einsum('ea,ijbe->jiab', tmp, t2)
    Hr2aaba += lib.einsum('ea,ijbe->jiab', tmp, t2)
    theta = tmp = None

    tau2baaa = np.einsum('ia,jb->ijab', r1, t1)
    tau2baaa += r2baaa * .5
    tau2baaa = tau2baaa - tau2baaa.transpose(0,1,3,2)
    tau = _make_tau(t2, t1, t1)
    tmp1aaba = lib.einsum('menf,ijef->mnij', eris_ovov, tau2baaa)
    tau2baaa = None
    Hr2baaa += .5*lib.einsum('mnij,mnab->ijab', tmp1aaba, tau)
    tau = tmp1aaba = None

    tau2aaba = np.einsum('ia,jb->ijab', r1, t1)
    tau2aaba += r2aaba * .5
    tau2aaba = tau2aaba - tau2aaba.transpose(1,0,2,3)
    tau = _make_tau(t2, t1, t1)
    tmp1baaa = lib.einsum('menf,ijef->mnij', eris_ovov, tau2aaba)
    tau2aaba = None
    Hr2aaba += .5*lib.einsum('mnij,mnab->ijab', tmp1baaa, tau)
    tau = tmp1baaa = None
    eris_ovov = None

    Hr2baaa = Hr2baaa - Hr2baaa.transpose(0,1,3,2)
    Hr2aaba = Hr2aaba - Hr2aaba.transpose(1,0,2,3)
    vector = amplitudes_to_vector_eomsf(Hr1, (Hr2baaa,Hr2aaba))
    return vector

def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    eris = imds.eris
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    tau = _make_tau(t2, t1, t1)
    nocc, nvir = t1.shape

    Fo = imds.Foo.diagonal()
    Fv = imds.Fvv.diagonal()
    Wovab = np.einsum('iaai->ia', imds.woVVo)
    Wovaa = Wovab + np.einsum('iaai->ia', imds.woVvO)

    eia = lib.direct_sum('-i+a->ia', Fo, Fv)
    Hr1aa = eia + Wovaa
    Hr1ab = eia + Wovab

    eris_ovov = np.asarray(eris.ovov)
    Wvvab = np.einsum('mnab,manb->ab', tau, eris_ovov)
    Wvvaa = .5*Wvvab - .5*np.einsum('mnba,manb->ab', tau, eris_ovov)
    ijb = np.einsum('iejb,ijeb->ijb', eris_ovov, t2)
    Hr2ab = lib.direct_sum('iJB+a->iJaB',-ijb, Fv)
    jab = np.einsum('kajb,kjab->jab', eris_ovov, t2)
    Hr2ab+= lib.direct_sum('-i-jab->ijab', Fo, jab)

    jib = np.einsum('iejb,ijbe->jib', eris_ovov, t2)
    jib = jib + jib.transpose(1,0,2)
    jib-= ijb + ijb.transpose(1,0,2)
    jba = np.einsum('kajb,jkab->jba', eris_ovov, t2)
    jba = jba + jba.transpose(0,2,1)
    jba-= jab + jab.transpose(0,2,1)
    Hr2aa = lib.direct_sum('jib+a->jiba', jib, Fv)
    Hr2aa+= lib.direct_sum('-i+jba->ijba', Fo, jba)
    eris_ovov = None

    Hr2baaa = lib.direct_sum('ijb+a->ijba',-ijb, Fv)
    Hr2baaa += Wovaa.reshape(1,nocc,1,nvir)
    Hr2baaa += Wovab.reshape(nocc,1,1,nvir)
    Hr2baaa = Hr2baaa + Hr2baaa.transpose(0,1,3,2)
    Hr2baaa+= lib.direct_sum('-i+jab->ijab', Fo, jba)
    Hr2baaa-= Fo.reshape(1,-1,1,1)
    Hr2aaba = lib.direct_sum('-i-jab->ijab', Fo, jab)
    Hr2aaba += Wovaa.reshape(1,nocc,1,nvir)
    Hr2aaba += Wovab.reshape(1,nocc,nvir,1)
    Hr2aaba = Hr2aaba + Hr2aaba.transpose(1,0,2,3)
    Hr2aaba+= lib.direct_sum('ijb+a->ijab', jib, Fv)
    Hr2aaba+= Fv.reshape(1,1,1,-1)
    Hr2ab += Wovaa.reshape(1,nocc,1,nvir)
    Hr2ab += Wovab.reshape(nocc,1,1,nvir)
    Hr2ab = Hr2ab + Hr2ab.transpose(1,0,3,2)
    Hr2aa += Wovaa.reshape(1,nocc,1,nvir) * 2
    Hr2aa = Hr2aa + Hr2aa.transpose(0,1,3,2)
    Hr2aa = Hr2aa + Hr2aa.transpose(1,0,2,3)
    Hr2aa *= .5

    Wooab = np.einsum('ijij->ij', imds.woOoO)
    Wooaa = Wooab - np.einsum('ijji->ij', imds.woOoO)
    Hr2aa += Wooaa.reshape(nocc,nocc,1,1)
    Hr2ab += Wooab.reshape(nocc,nocc,1,1)
    Hr2baaa += Wooab.reshape(nocc,nocc,1,1)
    Hr2aaba += Wooaa.reshape(nocc,nocc,1,1)

    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
    #:tmp = np.einsum('mb,mbaa->ab', t1, eris_ovvv)
    #:Wvvaa += np.einsum('mb,maab->ab', t1, eris_ovvv)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    tmp = np.zeros((nvir,nvir), dtype=dtype)
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        tmp += np.einsum('mb,mbaa->ab', t1[p0:p1], ovvv)
        Wvvaa += np.einsum('mb,maab->ab', t1[p0:p1], ovvv)
        ovvv = None
    Wvvaa -= tmp
    Wvvab -= tmp
    Wvvab -= tmp.T
    Wvvaa = Wvvaa + Wvvaa.T
    if eris.vvvv is None: # AO-direct CCSD, vvvv is not generated.
        pass
    elif len(eris.vvvv.shape) == 4:  # DO NOT use .ndim here for h5py library
                                     # backward compatbility
        eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
        tmp = np.einsum('aabb->ab', eris_vvvv)
        Wvvaa += tmp
        Wvvaa -= np.einsum('abba->ab', eris_vvvv)
        Wvvab += tmp
    else:
        for i in range(nvir):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.vvvv[i0:i0+i+1]))
            tmp = np.einsum('bb->b', vvv[i])
            Wvvaa[i] += tmp
            Wvvab[i] += tmp
            tmp = np.einsum('bb->b', vvv[:,:i+1,i])
            Wvvaa[i,:i+1] -= tmp
            Wvvaa[:i  ,i] -= tmp[:i]
            vvv = None

    Hr2aa += Wvvaa.reshape(1,1,nvir,nvir)
    Hr2ab += Wvvab.reshape(1,1,nvir,nvir)
    Hr2baaa += Wvvaa.reshape(1,1,nvir,nvir)
    Hr2aaba += Wvvab.reshape(1,1,nvir,nvir)

    vec_eeS = amplitudes_to_vector_singlet(Hr1aa, Hr2ab)
    vec_eeT = amplitudes_to_vector_triplet(Hr1aa, (Hr2aa,Hr2ab))
    vec_sf = amplitudes_to_vector_eomsf(Hr1ab, (Hr2baaa,Hr2aaba))
    return vec_eeS, vec_eeT, vec_sf


class EOMEE(EOM):
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        if koopmans:
            nocc = self.nocc
            nvir = self.nmo - nocc
            idx = diag[:nocc*nvir].argsort()
        else:
            idx = diag.argsort()

        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess

    kernel = eeccsd
    eeccsd = eeccsd
    get_diag = eeccsd_diag

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc*nvir + nocc*nocc*nvir*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ee()
        return imds

    @property
    def eee(self):
        return self.e


class EOMEESinglet(EOMEE):
    kernel = eomee_ccsd_singlet
    eomee_ccsd_singlet = eomee_ccsd_singlet
    matvec = eeccsd_matvec_singlet
    l_matvec = leeccsd_matvec_singlet

    build_M = build_M_ee
    norm = normalize_ee
    norm_l = normalize_l_ee

    def biorthonormalize(self,r_ee,l_ee,e_r,e_l,full = False):
        if (len(r_ee) != len(l_ee)):
            print('The number of right and left hand eigenvectors must be the same in order to biorthonormalise them')
            return r_ee, l_ee
        if (np.any(np.round(e_r,4) != np.round(e_l,4))):
            print('The right and left hand eigenvalues do not agree to 4 decimal places. Proceed with care')
        nocc = self.nocc
        nmo = self.nmo

        return biorthonormalize(r_ee,l_ee,e_r,e_l,nocc,nmo,self.build_M,self.norm,self.norm_l,full=full)

    def gen_matvec(self, imds=None, left=False, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[0]
        if left:
            matvec = lambda xs: [self.l_matvec(x, imds) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag


    def gen_e_rdms(self,r_ee,l_ee,prange=None,qrange=None):
        if not isinstance(r_ee,list): r_ee = [r_ee]
        if not isinstance(l_ee,list): l_ee = [l_ee]
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo-nocc
        nstates = len(r_ee)
        if prange is None: prange = range(nmo)
        if qrange is None: qrange = range(nmo)
        if isinstance(prange,int): prange = [prange]
        if isinstance(prange,int): qrange = [qrange]

        r0 = get_r0(self,r_ee)
        e_rdm = np.zeros((nstates,len(prange),len(qrange)))
        for n in range(nstates):
            l1_ee , l2_ee = self.vector_to_amplitudes(l_ee[n])
            for p in prange:
                for q in qrange:
                    e_vector = e_rdm_l_imd(self,p,q,l1_ee,l2_ee)
                    e0 = e_rdm_l0_imd(self,p,q,l1_ee,l2_ee)
                    if (p < nocc): e_rdm[n,p,q] = (p == q)
                    if (p < nocc) and (q >= nocc): e_rdm[n,p,q] = self._cc.t1[p,q-nocc]
                    e_rdm[n,p,q] += np.dot(e_vector,r_ee[n])
                    e_rdm[n,p,q] += e0*r0[n]

        return 2*e_rdm

    def gen_fs(self,r_ee,l_ee):
        tdm_r = self.gen_tdms(r_ee)
        tdm_l = self.gen_tdms(l_ee,left = True)
        f = tdm_r*tdm_l

        return f

    def gen_tdms(self,ee_vec,left=False,prange=None,qrange=None,eris=None):
# prange and qrange allow the calculation of a subset of the TDM
        if not isinstance(ee_vec,list): ee_vec = [ee_vec]
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo-nocc
        nstates = len(ee_vec)
        if prange is None: prange = range(nmo)
        if qrange is None: qrange = range(nmo)
        if isinstance(prange,int): prange = [prange]
        if isinstance(prange,int): qrange = [qrange]

        r0 = get_r0(self,ee_vec,eris)
        tdm = np.zeros((nstates,len(prange),len(qrange)))
        # bug: originally prange, but should be enumerate, code assumes
        # prange = range(nmo), but what if ia block is requested?
        for ip,p in enumerate(prange):
            for iq,q in enumerate(qrange):
                cput0 = (logger.process_clock(), logger.perf_counter())
                e0 = tdm_r0_imd(self,p,q)
                if left:
                    e_vector = tdm_l_imd(self,p,q).astype(ee_vec[0].dtype)
                else:
                    e_vector = tdm_r_imd(self,p,q).astype(ee_vec[0].dtype)
                tdm[:,ip,iq] = np.einsum('i,vi->v', e_vector,ee_vec)
                if not left: tdm[:,ip,iq] += e0*r0
                logger.timer_debug1(self, f'gen tdm {p} {q}', *cput0)
        tdm *= 2
        return tdm

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_singlet(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_singlet(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nov = nocc * nvir
        return nov + nov*(nov+1)//2


class EOMEETriplet(EOMEE):
    kernel = eomee_ccsd_triplet
    eomee_ccsd_triplet = eomee_ccsd_triplet
    matvec = eeccsd_matvec_triplet
    l_matvec = leeccsd_matvec_triplet


    build_M = build_M_triplet
    norm = normalize_ee
    norm_l = normalize_l_ee

    def biorthonormalize(self,r_ee,l_ee,e_r,e_l,full = False):
        if (len(r_ee) != len(l_ee)):
            print('The number of right and left hand eigenvectors must be the same in order to biorthonormalise them')
            return r_ee, l_ee
        if (np.any(np.round(e_r,4) != np.round(e_l,4))):
            print('The right and left hand eigenvalues do not agree to 4 decimal places. Proceed with care')
        nocc = self.nocc
        nmo = self.nmo

        return biorthonormalize(r_ee,l_ee,e_r,e_l,nocc,nmo,self.build_M,self.norm,self.norm_l,full=full)

    def gen_matvec(self, imds=None, left=False, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[1]
        if left:
            matvec = lambda xs: [self.l_matvec(x, imds) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_triplet(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_triplet(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nov = nocc * nvir
        return nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2 + nov*(nov-1)//2


class EOMEESpinFlip(EOMEE):
    kernel = eomsf_ccsd
    eomsf_ccsd = eomsf_ccsd
    matvec = eeccsd_matvec_sf

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[2]
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_eomsf(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_eomsf(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nbaaa = nocc*nocc*nvir*(nvir-1)//2
        naaba = nocc*(nocc-1)//2*nvir*nvir
        return nocc*nvir + nbaaa + naaba

#TODO: Check whether EOM methods works with rccsd.RCCSD when orbitals are complex
ccsd.CCSD.EOMIP         = lib.class_as_method(EOMIP)
ccsd.CCSD.EOMIP_Ta      = lib.class_as_method(EOMIP_Ta)
ccsd.CCSD.EOMEA         = lib.class_as_method(EOMEA)
ccsd.CCSD.EOMEA_Ta      = lib.class_as_method(EOMEA_Ta)
ccsd.CCSD.EOMEE         = lib.class_as_method(EOMEE)
ccsd.CCSD.EOMEESinglet  = lib.class_as_method(EOMEESinglet)
ccsd.CCSD.EOMEETriplet  = lib.class_as_method(EOMEETriplet)
ccsd.CCSD.EOMEESpinFlip = lib.class_as_method(EOMEESpinFlip)


class _IMDS:
    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.max_memory = cc.max_memory
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared_2e = False
        self.made_lee_imds = False

    def _make_shared_1e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1, t2, eris)
        self.Lvv = imd.Lvv(t1, t2, eris)
        self.Fov = imd.cc_Fov(t1, t2, eris)

        logger.timer_debug1(self, 'EOM-CCSD shared one-electron '
                            'intermediates', *cput0)
        return self

    def _make_shared_2e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1, t2, eris)
        self.Wovvo = imd.Wovvo(t1, t2, eris)
        self.Woovv = np.asarray(eris.ovov).transpose(0,2,1,3)

        self._made_shared_2e = True
        log.timer_debug1('EOM-CCSD shared two-electron intermediates', *cput0)
        return self

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ip_partition != 'mp':
            self._make_shared_2e()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1, t2, eris)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)
        log.timer_debug1('EOM-CCSD IP intermediates', *cput0)
        return self

    def make_t3p2_ip(self, cc, ip_partition=None):
        assert(ip_partition is None)
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ip()  # Make after t1/t2 updated
        self.Wovoo = self.Wovoo + Wovoo

        logger.timer_debug1(self, 'EOM-CCSD(T)a IP intermediates', *cput0)
        return self


    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ea_partition != 'mp':
            self._make_shared_2e()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1, t2, eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1, t2, eris)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, self.Wvvvv)
        log.timer_debug1('EOM-CCSD EA intermediates', *cput0)
        return self

    def make_t3p2_ea(self, cc, ea_partition=None):
        assert(ea_partition is None)
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ea()  # Make after t1/t2 updated
        self.Wvvvo = self.Wvvvo + Wvvvo

        logger.timer_debug1(self, 'EOM-CCSD(T)a EA intermediates', *cput0)
        return self

    def make_lee(self):
        # This needs updating to create wvVvO efficiently, as below
        #from pyscf.cc import rintermediates_slow as imd
        t1,t2,eris = self.t1,self.t2,self.eris
        self.wvVvO = imd.Wvvvo(t1, t2, eris)
        self.made_lee_imds = True

    def make_ee(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        dtype = np.result_type(t1, t2)
        if np.iscomplexobj(t2):
            raise NotImplementedError('Complex integrals are not supported in EOM-EE-CCSD')

        nocc, nvir = t1.shape
        nvir_pair = nvir*(nvir+1)//2

        fswap = lib.H5TmpFile()
        self.saved = lib.H5TmpFile()
        self.wvOvV = self.saved.create_dataset('wvOvV', (nvir,nocc,nvir,nvir), dtype.char)
        self.woVvO = self.saved.create_dataset('woVvO', (nocc,nvir,nvir,nocc), dtype.char)
        self.woVVo = self.saved.create_dataset('woVVo', (nocc,nvir,nvir,nocc), dtype.char)
        self.woOoV = self.saved.create_dataset('woOoV', (nocc,nocc,nocc,nvir), dtype.char)

        foo = eris.fock[:nocc,:nocc]
        fov = eris.fock[:nocc,nocc:]
        fvv = eris.fock[nocc:,nocc:]

        self.Fov = np.zeros((nocc,nvir), dtype=dtype)
        self.Foo = np.zeros((nocc,nocc), dtype=dtype)
        self.Fvv = np.zeros((nvir,nvir), dtype=dtype)

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:self.Fvv  = np.einsum('mf,mfae->ae', t1, eris_ovvv) * 2
        #:self.Fvv -= np.einsum('mf,meaf->ae', t1, eris_ovvv)
        #:self.woVvO = lib.einsum('jf,mebf->mbej', t1, eris_ovvv)
        #:self.woVVo = lib.einsum('jf,mfbe->mbej',-t1, eris_ovvv)
        #:tau = _make_tau(t2, t1, t1)
        #:self.woVoO  = 0.5 * lib.einsum('mebf,ijef->mbij', eris_ovvv, tau)
        #:self.woVoO += 0.5 * lib.einsum('mfbe,ijfe->mbij', eris_ovvv, tau)
        eris_ovoo = np.asarray(eris.ovoo)
        woVoO = np.empty((nocc,nvir,nocc,nocc), dtype=dtype)
        tau = _make_tau(t2, t1, t1)
        theta = t2*2 - t2.transpose(0,1,3,2)

        mem_now = lib.current_memory()[0]
        max_memory = max(0, self.max_memory - mem_now)
        blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
        for seg, (p0,p1) in enumerate(lib.prange(0, nocc, blksize)):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            # transform integrals (ia|bc) -> (ac|ib)
            fswap['ebmf/%d'%seg] = np.einsum('mebf->ebmf', ovvv)

            self.Fvv += np.einsum('mf,mfae->ae', t1[p0:p1], ovvv) * 2
            self.Fvv -= np.einsum('mf,meaf->ae', t1[p0:p1], ovvv)
            woVoO[p0:p1] = lib.einsum('mebf,ijef->mbij', ovvv, tau)
            woVvO = lib.einsum('jf,mebf->mbej', t1, ovvv)
            woVVo = lib.einsum('jf,mfbe->mbej',-t1, ovvv)
            ovvv = None

            eris_ovov = np.asarray(eris.ovov[p0:p1])
            woOoV = lib.einsum('if,mfne->mnie', t1, eris_ovov)
            woOoV+= eris_ovoo[:,:,p0:p1].transpose(2,0,3,1)
            self.woOoV[p0:p1] = woOoV
            woOoV = None

            tmp = lib.einsum('njbf,mfne->mbej', t2, eris_ovov)
            woVvO -= tmp * .5
            woVVo += tmp

            ovoo = lib.einsum('menf,jf->menj', eris_ovov, t1)
            woVvO -= lib.einsum('nb,menj->mbej', t1, ovoo)
            ovoo = lib.einsum('mfne,jf->menj', eris_ovov, t1)
            woVVo += lib.einsum('nb,menj->mbej', t1, ovoo)
            ovoo = None

            ovov = eris_ovov * 2 - eris_ovov.transpose(0,3,2,1)
            woVvO += lib.einsum('njfb,menf->mbej', theta, ovov) * .5

            self.Fov[p0:p1] = np.einsum('nf,menf->me', t1, ovov)
            tilab = np.einsum('ia,jb->ijab', t1[p0:p1], t1) * .5
            tilab += t2[p0:p1]
            self.Foo += lib.einsum('mief,menf->ni', tilab, ovov)
            self.Fvv -= lib.einsum('mnaf,menf->ae', tilab, ovov)
            eris_ovov = ovov = tilab = None

            woVvO -= lib.einsum('nb,menj->mbej', t1, eris_ovoo[p0:p1,:,:])
            woVVo += lib.einsum('nb,nemj->mbej', t1, eris_ovoo[:,:,p0:p1])

            woVvO += np.asarray(eris.ovvo[p0:p1]).transpose(0,2,1,3)
            woVVo -= np.asarray(eris.oovv[p0:p1]).transpose(0,2,3,1)

            self.woVvO[p0:p1] = woVvO
            self.woVVo[p0:p1] = woVVo

        self.Foo += foo + 0.5*np.einsum('me,ie->mi', self.Fov+fov, t1)
        self.Fvv += fvv - 0.5*np.einsum('me,ma->ae', self.Fov+fov, t1)

        # 0 or 1 virtuals
        woOoO = lib.einsum('je,nemi->mnij', t1, eris_ovoo)
        woOoO = woOoO + woOoO.transpose(1,0,3,2)
        woOoO += np.asarray(eris.oooo).transpose(0,2,1,3)

        tmp = lib.einsum('meni,jneb->mbji', eris_ovoo, t2)
        woVoO -= tmp.transpose(0,1,3,2) * .5
        woVoO -= tmp
        tmp = None
        ovoo = eris_ovoo*2 - eris_ovoo.transpose(2,1,0,3)
        woVoO += lib.einsum('nemi,njeb->mbij', ovoo, theta) * .5
        self.Foo += np.einsum('ne,nemi->mi', t1, ovoo)
        ovoo = None

        eris_ovov = np.asarray(eris.ovov)
        woOoO += lib.einsum('ijef,menf->mnij', tau, eris_ovov)
        self.woOoO = self.saved['woOoO'] = woOoO
        woVoO -= lib.einsum('nb,mnij->mbij', t1, woOoO)
        woOoO = None

        tmpoovv = lib.einsum('njbf,nemf->ejmb', t2, eris_ovov)
        ovov = eris_ovov*2 - eris_ovov.transpose(0,3,2,1)
        eris_ovov = None

        tmpovvo = lib.einsum('nifb,menf->eimb', theta, ovov)
        ovov = None

        tmpovvo *= -.5
        tmpovvo += tmpoovv * .5
        woVoO -= lib.einsum('ie,ejmb->mbij', t1, tmpovvo)
        woVoO -= lib.einsum('ie,ejmb->mbji', t1, tmpoovv)
        woVoO += eris_ovoo.transpose(3,1,2,0)

        # 3 or 4 virtuals
        eris_ovvo = np.asarray(eris.ovvo)
        tmpovvo -= eris_ovvo.transpose(1,3,0,2)
        fswap['ovvo'] = tmpovvo
        tmpovvo = None

        eris_oovv = np.asarray(eris.oovv)
        tmpoovv -= eris_oovv.transpose(3,1,0,2)
        fswap['oovv'] = tmpoovv
        tmpoovv = None

        woVoO += lib.einsum('mebj,ie->mbij', eris_ovvo, t1)
        woVoO += lib.einsum('mjbe,ie->mbji', eris_oovv, t1)
        woVoO += lib.einsum('me,ijeb->mbij', self.Fov, t2)
        self.woVoO = self.saved['woVoO'] = woVoO
        woVoO = eris_ovvo = eris_oovv = None

        #:theta = t2*2 - t2.transpose(0,1,3,2)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:ovvv = eris_ovvv*2 - eris_ovvv.transpose(0,3,2,1)
        #:tmpab = lib.einsum('mebf,miaf->eiab', eris_ovvv, t2)
        #:tmpab = tmpab + tmpab.transpose(0,1,3,2) * .5
        #:tmpab-= lib.einsum('mfbe,mifa->eiba', ovvv, theta) * .5
        #:self.wvOvV += eris_ovvv.transpose(2,0,3,1).conj()
        #:self.wvOvV -= tmpab
        nsegs = len(fswap['ebmf'])
        def load_ebmf(slice):
            dat = [fswap['ebmf/%d'%i][slice] for i in range(nsegs)]
            return np.concatenate(dat, axis=2)

        mem_now = lib.current_memory()[0]
        max_memory = max(0, self.max_memory - mem_now)
        blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc*nvir**2*4))))
        for p0, p1 in lib.prange(0, nvir, blksize):
            #:wvOvV  = lib.einsum('mebf,miaf->eiab', ovvv, t2)
            #:wvOvV += lib.einsum('mfbe,miaf->eiba', ovvv, t2)
            #:wvOvV -= lib.einsum('mfbe,mifa->eiba', ovvv, t2)*2
            #:wvOvV += lib.einsum('mebf,mifa->eiba', ovvv, t2)

            ebmf = load_ebmf(slice(p0, p1))
            wvOvV = lib.einsum('ebmf,miaf->eiab', ebmf, t2)
            wvOvV = -.5 * wvOvV.transpose(0,1,3,2) - wvOvV

            # Using the permutation symmetry (em|fb) = (em|bf)
            efmb = load_ebmf((slice(None), slice(p0, p1)))
            wvOvV += np.einsum('ebmf->bmfe', efmb.conj())

            # tmp = (mf|be) - (me|bf)*.5
            tmp = -.5 * ebmf
            tmp += efmb.transpose(1,0,2,3)
            emfb = ebmf = None
            wvOvV += lib.einsum('efmb,mifa->eiba', tmp, theta)
            tmp = None

            wvOvV += lib.einsum('meni,mnab->eiab', eris_ovoo[:,p0:p1], tau)
            wvOvV -= lib.einsum('me,miab->eiab', self.Fov[:,p0:p1], t2)
            wvOvV += lib.einsum('ma,eimb->eiab', t1, fswap['ovvo'][p0:p1])
            wvOvV += lib.einsum('ma,eimb->eiba', t1, fswap['oovv'][p0:p1])

            self.wvOvV[p0:p1] = wvOvV

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)
        return self

def _make_tau(t2, t1, r1, fac=1, out=None):
    tau = np.einsum('ia,jb->ijab', t1, r1)
    tau = tau + tau.transpose(1,0,3,2)
    tau *= fac * .5
    tau += t2
    return tau

def _cp(a):
    return np.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf.cc import rccsd

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

    mycc = rccsd.RCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    myeom = EOMIP(mycc)
    print("IP energies... (right eigenvector)")
    e,v = ipccsd(myeom, nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    print("IP energies... (left eigenvector)")
    le,lv = ipccsd(myeom, nroots=3,left=True)
    print(le[0] - 0.43356040428879794)
    print(le[1] - 0.51876597800180335)
    print(le[2] - 0.67828755013874864)

    e = myeom.ipccsd_star_contract(e, v, lv)
    print(e[0] - 0.43793202073189047)
    print(e[1] - 0.52287073446559729)
    print(e[2] - 0.67994597948852287)

    myeom = EOMEA(mycc)
    print("EA energies... (right eigenvector)")
    e,v = eaccsd(myeom, nroots=3)
    print(e[0] - 0.16737886282063008)
    print(e[1] - 0.24027622989542635)
    print(e[2] - 0.51006796667905585)

    print("EA energies... (left eigenvector)")
    le,lv = eaccsd(myeom, nroots=3, left=True)
    print(le[0] - 0.16737896537079733)
    print(le[1] - 0.24027634198123343)
    print(le[2] - 0.51006809015066612)

    e = myeom.eaccsd_star_contract(e,v,lv)
    print(e[0] - 0.16656250953550664)
    print(e[1] - 0.23944144521387614)
    print(e[2] - 0.41399436888830721)

    myeom = EOMEESpinFlip(mycc)
    np.random.seed(1)
    v = np.random.random(myeom.vector_size())
    r1, r2 = vector_to_amplitudes_eomsf(v, myeom.nmo, myeom.nocc)
    print(lib.finger(r1)    - 0.017703197938757409)
    print(lib.finger(r2[0]) --21.605764517401415)
    print(lib.finger(r2[1]) - 6.5857056438834842)
    print(abs(amplitudes_to_vector_eomsf(r1, r2) - v).max())

    myeom = EOMEE(mycc)
    e,v = myeom.eeccsd(nroots=1)
    print(e - 0.2757159395886167)

    e,v = myeom.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = myeom.eeccsd(nroots=4, koopmans=True)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = myeom.eeccsd(nroots=4, guess=v[:4])
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)


    mycc = ccsd.CCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    myeom = EOMIP(mycc)
    print("IP energies... (right eigenvector)")
    e,v = ipccsd(myeom, nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    print("IP energies... (left eigenvector)")
    le,lv = ipccsd(myeom, nroots=3,left=True)
    print(le[0] - 0.43356040428879794)
    print(le[1] - 0.51876597800180335)
    print(le[2] - 0.67828755013874864)

    e = myeom.ipccsd_star_contract(e, v, lv)
    print(e[0] - 0.43793202073189047)
    print(e[1] - 0.52287073446559729)
    print(e[2] - 0.67994597948852287)

    myeom = EOMEA(mycc)
    print("EA energies... (right eigenvector)")
    e,v = eaccsd(myeom, nroots=3)
    print(e[0] - 0.16737886282063008)
    print(e[1] - 0.24027622989542635)
    print(e[2] - 0.51006796667905585)

    print("EA energies... (left eigenvector)")
    le,lv = eaccsd(myeom, nroots=3, left=True)
    print(le[0] - 0.16737896537079733)
    print(le[1] - 0.24027634198123343)
    print(le[2] - 0.51006809015066612)

    e = myeom.eaccsd_star_contract(e,v,lv)
    print(e[0] - 0.16656250953550664)
    print(e[1] - 0.23944144521387614)
    print(e[2] - 0.41399436888830721)

    myeom = EOMEE(mycc)
    e,v = myeom.eeccsd(nroots=1)
    print(e - 0.2757159395886167)

    e,v = myeom.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = myeom.eeccsd(nroots=4, koopmans=True)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = myeom.eeccsd(nroots=4, guess=v[:4])
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
