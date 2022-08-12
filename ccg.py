"""
Created on Fri Mar 20 07:43:29 2020

post hartree fock gamma point means real integrals, which means I can use the parts 
of molecular post hartee fock pyscf that are the most efficient with memory usage

# modified version of gamma-point ccsd, pbc/cc/ccsd.py, to be more memory efficient
# original ao2mo in cc/rccsd gets cc integrals from slicing fully expanded mo eris
# uses compact mo eris, then techniques from cc/ccsd

# ccg - Coupled Cluster Gamma point (real integrals)

@author: blau
"""

import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import dfccsd
from pyscf.pbc import tools
from pyscf.tdscf import rhf
import rcis


def CCG(mf, frozen=None, mo_coeff=None, mo_occ=None):
    '''
    pick incore ccsd or outcore dfccsd depending on current memory
    '''
    nmo = mf.nmo
    nao = mf.mo_coeff.shape[0]
    nmo_pair = nmo * (nmo+1) // 2
    nao_pair = nao * (nao+1) // 2
    mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
    mem_now = lib.current_memory()[0]
    
    if mem_incore+mem_now < mf.max_memory:
        return CCSD(mf,frozen=frozen,mo_coeff=mo_coeff,mo_occ=mo_occ)
    else:
        return DFCCSD(mf,frozen=frozen,mo_coeff=mo_coeff,mo_occ=mo_occ)

class CCSD(ccsd.CCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.keep_exxdiv = False

    def ao2mo(self, mo_coeff=None):
        '''
        copied code from https://github.com/pyscf/pyscf/blob/master/pyscf/cc/ccsd.py#L1143
        modify to work with the madelung correction (level shift for accurate energies)
        '''
        ao2mofn = _gen_ao2mofn(self._scf)
        exxdiv = self._scf.exxdiv if self.keep_exxdiv else None

        logger.info(self,f'Memory max: {self.max_memory}, pre ao2mo {int(lib.current_memory()[0])}')
        with lib.temporary_env(self._scf, exxdiv=exxdiv):
            logger.info(self, 'ccsd df incore')
            eris = _make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)
        logger.info(self,f'Memory max: {self.max_memory}, post ao2mo + gen ccsd integrals {int(lib.current_memory()[0])}')

        if not self.keep_exxdiv:
            madelung = tools.madelung(self._scf.cell, self._scf.kpt)
            eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)

        return eris

class DFCCSD(dfccsd.RCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        dfccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.keep_exxdiv = False

    def ao2mo(self, mo_coeff=None):
        exxdiv = self._scf.exxdiv if self.keep_exxdiv else None

        logger.info(self,f'Memory max: {self.max_memory}, pre ao2mo {int(lib.current_memory()[0])}')
        with lib.temporary_env(self._scf, exxdiv=exxdiv):
            logger.info(self, 'ccsd df outcore')
            eris = dfccsd._make_df_eris(self, mo_coeff)
        logger.info(self,f'Memory max: {self.max_memory}, post ao2mo + gen ccsd integrals {int(lib.current_memory()[0])}')

        if not self.keep_exxdiv:
            madelung = tools.madelung(self._scf.cell, self._scf.kpt)
            eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)

        return eris

# ERI CIS
class CIS(rcis.CIS):
    def ao2mo(self, mo_coeff=None):
        logger.info(self,f'Memory max: {self.max_memory}, pre ao2mo {int(lib.current_memory()[0])}')
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = rcis._make_eris(self, mo_coeff)
        logger.info(self,f'Memory max: {self.max_memory}, post ao2mo + gen CIS integrals {int(lib.current_memory()[0])}')

        # do not adjust the mo_energy!
        #madelung = tools.madelung(self._scf.cell, self._scf.kpt)
        #eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)

        return eris

# response formulation of CIS
class TDA(rhf.TDA):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        # warning: make sure you copy mf, as _scf is modified
        rhf.TDA.__init__(self, mf)
        # create a fake cc with exxdiv None, which properly initializes frozen
        # mo_coeff, energy, etc
        with lib.temporary_env(mf, exxdiv=None):
            dummycc = CCSD(mf, frozen, mo_coeff, mo_occ)
            eris = ccsd._ChemistsERIs()
            eris._common_init_(dummycc)
        self._scf.mo_occ = self._scf.mo_occ[dummycc.get_frozen_mask()]
        self._scf.mo_coeff = eris.mo_coeff
        self._scf.mo_energy = eris.mo_energy

    def gen_vind(self, mf):
        vind, hdiag = rhf.TDA.gen_vind(self, mf)
        def vindp(x):
            with lib.temporary_env(mf, exxdiv=None):
                return vind(x)
        return vindp, hdiag

    def nuc_grad_method(self):
        raise NotImplementedError

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    '''
    function from ccsd module modified to accept ao2mofn
    '''
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    
    logger.info(mycc,f'Memory max: {mycc.max_memory}, pre ao2mo {int(lib.current_memory()[0])}')
    eri1 = ao2mofn(eris.mo_coeff)
    logger.info(mycc,f'Memory max: {mycc.max_memory}, post ao2mo {int(lib.current_memory()[0])}')

    nvir_pair = nvir * (nvir+1) // 2
    eris.oooo = numpy.empty((nocc,nocc,nocc,nocc))
    eris.ovoo = numpy.empty((nocc,nvir,nocc,nocc))
    eris.ovvo = numpy.empty((nocc,nvir,nvir,nocc))
    eris.ovov = numpy.empty((nocc,nvir,nocc,nvir))
    eris.ovvv = numpy.empty((nocc,nvir,nvir_pair))
    eris.vvvv = numpy.empty((nvir_pair,nvir_pair))

    ij = 0
    outbuf = numpy.empty((nmo,nmo,nmo))
    oovv = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        for j in range(i+1):
            eris.oooo[i,j] = eris.oooo[j,i] = buf[j,:nocc,:nocc]
            oovv[i,j] = oovv[j,i] = buf[j,nocc:,nocc:]
        ij += i + 1
    eris.oovv = oovv
    oovv = None

    ij1 = 0
    for i in range(nocc,nmo):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        eris.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
        eris.ovvo[:,i-nocc] = buf[:nocc,nocc:,:nocc]
        eris.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
        eris.ovvv[:,i-nocc] = lib.pack_tril(buf[:nocc,nocc:,nocc:])
        dij = i - nocc + 1
        lib.pack_tril(buf[nocc:i+1,nocc:,nocc:],
                      out=eris.vvvv[ij1:ij1+dij])
        ij += i + 1
        ij1 += dij
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    eri1 = None
    logger.info(mycc,f'Memory max: {mycc.max_memory}, ccsd integrals {int(lib.current_memory()[0])}')
    return eris
    
def _gen_ao2mofn(mf,compact=True):
    # Compact = True modification
    with_df = mf.with_df
    kpt = mf.kpt
    def ao2mofn(mo_coeff):
        return with_df.ao2mo(mo_coeff, kpt, compact=compact) 
    return ao2mofn

def _adjust_occ(mo_energy, nocc, shift):
    '''Modify occupied orbital energy'''
    mo_energy = mo_energy.copy()
    mo_energy[:nocc] += shift
    return mo_energy

if __name__ == '__main__':
    import surf, sys
    print(sys.version)
    
    # parse the input
    args = surf.getArgs(sys.argv[1:])
    
    # setup calculation
    cell,args = surf.init_cell(args)
    mf,args = surf.init_scf(cell,args)
    
    mf.kernel()

    log = logger.Logger(mf.stdout, mf.verbose)
    
    from pyscf.pbc import cc
    mycc = cc.RCCSD(mf)
    e_corr,t1,t2 = mycc.kernel()

    # ccsd implementation that assumes real integrals
    mycc = CCSD(mf)
    e_corr_s,t1_s,t2_s = mycc.kernel()
    
    # uses compact = True transformation to save 4x space for mo integrals
    mycc = DFCCSD(mf)
    e_corr_t,t1_t,t2_t = mycc.kernel()
    
    print('CCSD ecorr check: {0} {1} {2}'.format(e_corr,e_corr_s,e_corr-e_corr_s))
    print('t1 max/min diff: {0} {1}'.format((t1-t1_s).max(),(t1-t1_s).min()))
    print('t2 max/min diff: {0} {1}'.format((t2-t2_s).max(),(t2-t2_s).min()))
    
    print('DFCCSD ecorr check: {0} {1} {2}'.format(e_corr,e_corr_t,e_corr-e_corr_t))
    print('t1 max/min diff: {0} {1}'.format((t1-t1_t).max(),(t1-t1_t).min()))
    print('t2 max/min diff: {0} {1}'.format((t2-t2_t).max(),(t2-t2_t).min()))
            