"""
Created on Thu Feb  6 11:55:48 2020

An attempt at self-consistent regional embedding

DOI: 10.1021/acs.jctc.9b00933
J. Chem. Theory Comput. 2020, 16, 119−129

DOI: 10.1021/acs.jctc.6b00316
J. Chem. Theory Comput. 2016, 12, 2706−2719

@author: blau
"""

import numpy as np
import scipy.optimize
import scipy.linalg
import types
import copy
import time
from multiprocessing import Pool
from functools import partial

# FIX ME (import from regional_embedding)
from emb import make_rdmet
from emb import canonicalize

try:
    from pyscf import gto, lo, scf
    from pyscf.lib import logger, unpack_tril, pack_tril, num_threads
    from pyscf import mp as mmp # molecular mp
    # more memory efficient ccsd molecular solver for Gamma-point df object
    import ccg
    from pyscf.cc import ccsd_rdm
except ImportError:
    print('Testing on a machine without pyscf')

class SCRE():
    '''
    Self-consistent regional embedding

    This class will continuously modify the mean-field object, because the
    mo_coeff and other attributes will continuously change during the self-consistent
    cycle, so my philosophy is to update the mean-field object rather than
    internal attributes of the SCRE class, to avoid any chance of confusion

    Attributes:
        conv_tol : float
            Difference between successive iterations of the correlation potential

        max_cycle : int
            Number of cycles

        corr : str
            Tuple of strings, first is solver for frozen part, second is for active region

        ft : (float,float)
            Tuple of eigenvalue thresholds
        
        minao: str
            The atomic basis that defines that region for embedding. For gth/ccecp,
            the corresponding szv basis is used.

    Saved results:

    '''

    def __init__(self,mf,aolabels='!',ft=(0.1,0.1),corr=('HF','MP2'),minao='minao'):
        logger.info(mf,'''WARNING: THIS CLASS WILL MODIFY THE get_fock() METHOD
        OF THE PASSED mean-field OBJECT. THE get_fock METHOD WILL 
        ADD mf.U. PLEASE KEEP THIS CHANGE IN MIND IF YOU INTEND 
        TO USE THE INTERNAL mf OBJECT IN SOME LATER CALCULATION. 
        AS LONG AS mf.U IS LEFT AT 0 THEN THE CHANGE SHOULD BE TRANSPARENT.''')  
        self.mf = copy.copy(mf)
        self.mol = self.mf.mol
        # modify get_fock() method of mf instance to include corr potential U
        self.mf.get_fock = types.MethodType(get_fock, self.mf)
        self.mf.U = 0
        
        self.ft = ft # tuple of two numbers
        self.corr = corr
        self.aolabels = aolabels
        self.minao = minao
        self.conv_tol = 1e-4 # 0.0016 Ha for 43 meV for 1 kcal/mol
        self.norm_tol = 1e-4
        self.max_cycle = 10 # set to 0 for single shot (correlated calc on frag and/or bath)

        ######
        # internal attributes
        self.e_corr = None
        self.conv = False
        
        self.rdm_corr = None
        self.rdm_mf = None # 'mf' means mf + U
        self.U = None
        self.Fo = None

        self.ro_coeff = None # regional orbitals
        self.cro_coeff = None # canonicalized regional orbitals
        self.cro_energy = None
        self.frozen = None
        self.emb_dat = None
        
        self.pop_cutoff = 1e-3

    def make_rdmet(self):
        mol = self.mol
        mf = self.mf
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        ft = self.ft
        aolabels = self.aolabels
        minao = self.minao

        self.ro_coeff, self.frozen, self.emb_dat = make_rdmet(mol,mf,ft,aolabels,minao)
            
        return self.ro_coeff, self.frozen, self.emb_dat
        
    def canonicalize(self,fock_ao=True):
        mf = self.mf
        if self.ro_coeff is None:
            self.make_rdmet()
        ro_coeff = self.ro_coeff
        # if the whole system is embedded, frozen is None, but go through the
        # canonicalization procedure anyways
        frozen = self.frozen

        self.cro_coeff, self.cro_energy = canonicalize(mf,ro_coeff,frozen,fock_ao=fock_ao)
            
        return self.cro_coeff, self.cro_energy

    def kernel(self):
        '''
        Self-consistent regional embedding

        Returns
        -------
        conv

        '''
        self.dump_flags()
        
        corr  = self.corr
        mf    = self.mf
        nocc  = self.nocc
        nao   = self.nao
        nmo   = self.nmo
        npair = nao*(nao+1)//2

        # lowdin symmetric orthogonalization Us^(-1/2)U^+
        S  = mf.get_ovlp()
        X, Xi = lowdin(S)

        # s = non-orthogonal AO basis, o = orthogonal AO basis
        U_old = np.zeros((nao,nao))
        U_new = np.zeros((nao,nao))
        e_old = 0.
        e_new = 0.

        # transform F from s -> o and associated quantities
        # includes U (mf.U -> get_fock())
        Fs = mf.get_fock()
        Fo = X @ Fs @ X
        # mf 1RDM oAO basis
        D0 = Xi @ mf.make_rdm1() @ Xi
        
        if 'HF' in corr:
            with_frozen = True
        else:
            with_frozen = False

        conv = False
        for istep in range(self.max_cycle):
            # fragment and bath solvers carried out in non-orthogonal AO basis
            self.make_rdmet()
            self.canonicalize()
            ro_coeff = self.ro_coeff
            frozen   = self.frozen

            frag_frozen = frozen
            frag_solver = make_solver(corr[1],mf,frag_frozen,ro_coeff)
            frag_solver.kernel()
            e_corr_frag = frag_solver.e_corr
            frag_rdm1   = make_rdm1(frag_solver,with_frozen)
            
            bath_frozen = frag_solver.get_frozen_mask()
            frag_solver = None
            
            bath_solver = make_solver(corr[0],mf,bath_frozen,ro_coeff)
            bath_solver.kernel()
            e_corr_bath = bath_solver.e_corr
            if not with_frozen: 
                bath_rdm1   = make_rdm1(bath_solver,with_frozen)
            bath_solver = None
            
            e_old, e_new = e_new, e_corr_bath+e_corr_frag

            # correlated 1RDM: MO -> sAO -> oAO
            if with_frozen:
                Dc = ro_coeff @ frag_rdm1 @ ro_coeff.conj().T
            else:
                Dc = ro_coeff[:,frag_frozen] @ bath_rdm1 @ ro_coeff[:,frag_frozen].conj().T
                Dc += ro_coeff[:,bath_frozen] @ frag_rdm1 @ ro_coeff[:,bath_frozen].conj().T
            Dc  = Xi @ Dc @ Xi
            
            # choose orbitals to add correlation potential
            dD = (Dc - D0).diagonal()
            diag_mask = abs(dD) > self.pop_cutoff
            mask = np.zeros(Fo.shape,dtype=bool)
            mask[np.ix_(diag_mask,diag_mask)] = 1
            mask = pack_tril(mask)
            u = np.random.random_sample(mask.sum())
            u = u/Fo.max()/100
            logger.info(mf,'Number of correlation potentials (pp, pq): %g, %g',diag_mask.sum(),mask.sum()-diag_mask.sum())
                
            # population trace should work as in MO basis
            logger.info(mf,'Population check: Dc: %.9g, bath+frag MO rdm: %.9g, nelec: %g',
                        np.trace(Dc),
                        np.trace(bath_rdm1+frag_rdm1),
                        self.mol.nelectron)
            
            logger.info(mf,'   Before minimization, [Dc-D0]^2 = %.9g',np.linalg.norm((Dc - D0))**2)

            # start from the u of the previous iteration
            t0, w0 = logger.process_clock(), logger.perf_counter()
            res = scipy.optimize.minimize(cost_function,
                                        u,
                                        args=(mf,Fo,Dc,mask),
                                        method='BFGS',
                                        jac=True,
                                        options={'disp':True})
            logger.timer(mf,'Fitting correlation potential', t0, w0)            
            u = res.x
            logger.info(mf,'   Minimize: {} with status {} and msg "{}"'.format(res.success,res.status,res.message))

            U_old, U_new = U_new, u2U(u,mask)
            dU = np.linalg.norm(U_new-U_old)

            # update Fock matrix and other quantitites in orthogonal AO basis
            Fo += U_new
            mo_energy, mo_coeff = mf.eig(Fo,None)
            D0_old, D0 = D0, mf.make_rdm1(mo_coeff=mo_coeff)
            dD0 = np.linalg.norm(D0-D0_old)
            
            # update mf object with new values - for next regional embedding
            mf.U += Xi @ U_new @ Xi
            mf.mo_energy, mf.mo_coeff = mf.eig(Fs + mf.U,S)

            logger.info(mf,'cycle = %d, dE = %.9g, dU (2-norm) = %.9g, dD0 = %.9g',istep+1,e_old-e_new,dU,dD0)
            logger.info(mf,'    E_frag = %.9g',e_corr_frag)
            logger.info(mf,'    E_bath = %.9g',e_corr_bath)
            if abs(e_old-e_new) < self.conv_tol and dU < self.norm_tol:
                logger.info(mf,'CONVERGED self-consistent regional embedding')
                conv = True
                break
            
        if not conv:
            logger.info(mf,'SCRE did not converge - see output')

        self.e_corr = e_new
        self.conv = conv
        self.rdm_corr = Dc
        self.rdm_mf = X @ D0 @ X
        self.U = U_new
        self.Fo = Fo

        # CCSD(T) has no 1RDM: do it at the very end if (T) is requested

        return conv
    
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self.mf, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('fragment threshold = {}'.format(self.ft))
        log.info('conv_tol = %g', self.conv_tol)
        log.info('norm_tol = %g', self.norm_tol)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('bath/frag solver = {}'.format(self.corr))
        log.info('ao label = {}'.format(self.aolabels))
        log.info('minao basis = %s',self.minao)
        log.info('pop cutoff = %g', self.pop_cutoff)        
        log.info('')
        return self
    
def adjust_phase(c):
    idx = np.argmax(abs(c.real), axis=0)
    c[:,c[idx,np.arange(c.shape[1])].real<0] *= -1
    return c

def lowdin(s):
    '''
    Given an overlap matrix, returns:

        X   : transformation to diagonalize the AO basis
        X^-1: transformation to return to the original nonorthogonal AO basis

    '''

    e, v = scipy.linalg.eigh(s)
    idx = e > 1e-15
    s = np.sqrt(e[idx])

    X = np.dot(v[:,idx]/s, v[:,idx].conj().T)
    Xi = np.dot(v[:,idx]*s, v[:,idx].conj().T)

    # adjust phase (largest coeff always positive)
    # use code from scf/hf.py:eig
    # one example 'ft_0.1_0.1/output_frag_gw_w4_augtzv2p/surf_w4_TCCSD__bwgth_aug_tzv2p.mat' 
    # for symmetric Lowdin X had the largest value off the diagonal
    X = adjust_phase(X)
    Xi = adjust_phase(Xi)

    return X, Xi

def u2U(u,mask=False):
    '''
    U = u[np.tril_indices(U.shape)]
    
    You can replace this function if you want to lessen dependence on pyscf
    
    I am not taking the real of the diagonal elements, assuming all u are real
    
    mask = linearized lower triangular array of 0's and 1's that describes where
    elements of u go
    '''
    
    if mask is False:
        return unpack_tril(u)
    else:
        u_mask = np.zeros(mask.size)
        u_mask[mask] = u
        return unpack_tril(u_mask)
        
def cost_function(u,mf,Fo,Dc,mask=False,jac=True):
    '''
    cost function: \sum (D^corr - D^HF)^2
    
    gradient is evaluated with RSPT at D^HF(U)
    '''
    U = u2U(u,mask)
    mo_energy, mo_coeff = mf.eig(Fo + U,None)
    D = mf.make_rdm1(mo_coeff=mo_coeff)
    
    cost = np.linalg.norm(Dc - D)**2
    
    if not jac:
        return cost
    
    nocc = np.count_nonzero(mf.mo_occ)
    occ_coeff = mo_coeff[:,:nocc]
    vir_coeff = mo_coeff[:,nocc:].conj().T # save time in evaluation of X

    nao = occ_coeff.shape[0]
    U[np.diag_indices(nao)] *= 0.5 # prevent double counting of diagonal elements
    u_grad = pack_tril(U)
    ur, uc = np.tril_indices(nao)
    #umunu = [(u,mu,nu) for u,mu,nu in zip(u,ur,uc)]
    
    if mask is not False:
        u_grad = u_grad[mask]
        ur = ur[mask]
        uc = uc[mask]
    
    Dgrad = -2*(Dc-D)
    Y = 1/(mo_energy[nocc:,None] - mo_energy[None,:nocc])
    
    make_grad = partial(make_rdm1_grad,occ_coeff=occ_coeff,vir_coeff=vir_coeff,Y=Y,Dgrad=Dgrad)
    with Pool(processes=num_threads()-1) as pool:
        grad = np.asarray(pool.starmap(make_grad, zip(u_grad,ur,uc)))
    # dD/dL -> dD/dU
    grad = grad/u
    
    return cost, grad

def make_rdm1_grad(u,mu,nu,occ_coeff,vir_coeff,Y,Dgrad):
    '''
    evaluate gradient for one position of the correlation potential
    '''
    #u, mu, nu = umunu
    
    X = u*vir_coeff[:,mu,None]*occ_coeff[None,nu,:]
    X += u.conj()*vir_coeff[:,nu,None]*occ_coeff[None,mu,:]
    Z = vir_coeff.conj().T @ (-X*Y) @ occ_coeff.conj().T
    dD = Z + Z.conj().T
    
    return (Dgrad*dD).sum()

def make_rdm1(mycorr,with_frozen):
    '''
    Return 1RDM from a correlated solver, without the frozen contribution
    Returns the 1RDM in the active MO basis (no frozen MOs)

    Input:
        mycorr : pyscf post-HF object (MP2 or CCSD)
        eris : eris for MP2
    '''

    if isinstance(mycorr,mmp.dfmp2.DFMP2): # MP2
        t2 = mycorr.t2
        doo, dvv = mmp.mp2._gamma1_intermediates(mycorr)
        nocc = doo.shape[0]
        nvir = dvv.shape[0]
        dov = np.zeros((nocc,nvir), dtype=doo.dtype)
        dvo = dov.T
        d1 = (doo, dov, dvo, dvv)
    elif isinstance(mycorr,ccg.DFCCSD):
        t1 = mycorr.t1
        t2 = mycorr.t2
        l1 = mycorr.l1
        l2 = mycorr.l2
        if l1 is None: l1, l2 = mycorr.solve_lambda(t1, t2)
        d1 = ccsd_rdm._gamma1_intermediates(mycorr, t1, t2, l1, l2)
    else:
        return 0
        
    return ccsd_rdm._make_rdm1(mycorr, d1, with_frozen=with_frozen, ao_repr=False)

def make_solver(solver,mf,frozen,mo_coeff):
    '''
    Return properly set up solvers for frozen and active part
    Returns eris as well

    Gamma point calculation
    '''

    #TODO: write for k-point sampling?

    if solver == 'MP2' or solver == 'MP':
        mycorr = mmp.dfmp2.DFMP2(mf,frozen=frozen,mo_coeff=mo_coeff)
    elif solver == 'CCSD':
        mycorr = ccg.DFCCSD(mf,frozen=frozen,mo_coeff=mo_coeff)
    elif solver == 'HF':
        mycorr = empty_solver(mf,frozen=frozen,mo_coeff=mo_coeff)
    else:
        logger.info(mf,'A valid solver was not selected!')
        raise NotImplementedError

    return mycorr

class empty_solver():
    '''
    empty solver that does not do a correlated calculation for a frag/bath space
    '''
    def __init__(self,mf,frozen,mo_coeff):
        self.e_corr = 0
        return
    
    def kernel(self):
        return self.e_corr

def get_fock(mf,*args,**kwargs):
    '''
    New method that adds the correlation potential to the Fock matrix
    Still needs to be patched into a given mf object
    '''
    if isinstance(mf, scf.hf.RHF):
        logger.info(mf,'** RHF F+U')
        return scf.hf.get_fock(mf,*args,**kwargs) + mf.U
    elif isinstance(mf, scf.uhf.UHF):
        logger.info(mf,'** UHF F+U')
        return scf.uhf.get_fock(mf,*args,**kwargs) + mf.U

if __name__ == '__main__':
    '''
    requires 'surf' module from same folder (package)
    
    example input to test:
    '''
    import surf, sys
    from pyscf.pbc.scf import addons
    print(sys.version)
    
    args = surf.getArgs(sys.argv[1:])
    
    mf,cell,args = surf.init_calc(args)

    if not mf.converged:
        mf.kernel()
    else:
        print('\n===== Using converged SCF result =====')
        mf._finalize()
        
    S = mf.get_ovlp()
    print('\n\ncond(S): ',np.linalg.cond(S))
    fock1 = S @ mf.mo_coeff @ np.diag(mf.mo_energy) @ mf.mo_coeff.conj().T @ S
    fock2 = mf.get_fock()
    print('Are these restricted focks the same?',np.allclose(fock1,fock2))
    print('Are these restricted focks the same? abs diff',np.abs(fock1-fock2).max())
    print('Are these restricted focks the same? rel diff',np.abs((fock1-fock2)/fock2).max())
    e,c = mf.eig(fock1,S)
    e2,c2 = mf.eig(fock2,S)
    print('eigenvalues close?',np.allclose(e,e2))
    print('eigenvalues abs diff?',np.abs(e-e2).max())
    #np.set_printoptions(threshold=e.size)
    #print(e-e2)
    print('eigenvalues rel diff?',np.abs((e-e2)/e2).max())
    print('eigenvectors close?',np.allclose(c,c2))
    
    myrre = SCRE(mf)
    ro_coeff, frozen, emb_dat = myrre.make_rdmet()
    
    umf = addons.convert_to_uhf(copy.copy(mf))
    myure = SCRE(umf)
    uro_coeff, ufrozen, uemb_dat = myure.make_rdmet()    
    
    print('RRE same as URE?')
    print('ro_coeff v uro_coeff[0] ',np.abs(ro_coeff-uro_coeff[0]).max())
    print('ro_coeff v uro_coeff[0] ',np.allclose(ro_coeff,uro_coeff[0]))
    print('ro_coeff v uro_coeff[1] ',np.abs(ro_coeff-uro_coeff[1]).max())
    print('ro_coeff v uro_coeff[1] ',np.allclose(ro_coeff,uro_coeff[1]))
    
    print('RHF Canonicalize slow')
    cro_coeff, cro_energy = myrre.canonicalize()
    #cro_coeff = adjust_phase(cro_coeff)
    print('RHF Canonicalize fast')
    myrre2 = SCRE(copy.copy(mf))
    ro_coeff2, frozen2, emb_dat2 = myrre2.make_rdmet()
    print('Are the two RRE starting off the same?',np.allclose(ro_coeff,ro_coeff2))
    cro_coeff2, cro_energy2 = myrre2.canonicalize(fock_ao=False)
    #cro_coeff2 = adjust_phase(cro_coeff2)
    print('RHF Checking if canonicalization through AO->FO is the same as through MO->AO->FO')
    print('coeff ',np.allclose(cro_coeff,cro_coeff2))
    print('energy ',np.allclose(cro_energy,cro_energy2))
    print('coeff diff ',np.abs(cro_coeff-cro_coeff2).max())
    print('energy diff ',np.abs(cro_energy-cro_energy2).max())
    
    ucro_coeff, ucro_energy = myure.canonicalize()
    #ucro_coeff[0] = adjust_phase(ucro_coeff[0])
    #ucro_coeff[1] = adjust_phase(ucro_coeff[1])
    print('UHF canonicalization the same as RHF canonicalization? (coeffs and energy)')
    print('coeff [0] diff ',np.abs(cro_coeff-ucro_coeff[0]).max())
    print('coeff [0] compare',np.allclose(cro_coeff,ucro_coeff[0]))
    print('coeff [1] diff ',np.abs(cro_coeff-ucro_coeff[1]).max())
    print('coeff [1] compare ',np.allclose(cro_coeff,ucro_coeff[1]))
    print('energy [0] compare ',np.allclose(cro_energy,ucro_energy[0]))
    print('energy [1] compare ',np.allclose(cro_energy,ucro_energy[1]))   
    print('energy [0] diff ',np.abs(cro_energy-ucro_energy[0]).max())
    print('energy [1] diff ',np.abs(cro_energy-ucro_energy[1]).max())