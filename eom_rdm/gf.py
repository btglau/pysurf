import collections
import numpy as np
import scipy
from pyscf import lib
from pyscf.cc import eom_rccsd

def greens_func_multiply(ham,cc,imds,vector,linear_part,args=None):
    return np.array(ham(cc,vector,imds=imds) + (linear_part)*vector)

###################
# Greens Drivers  #
###################

class SpectralFunctions:
    def __init__(self,cc):
        self.gmres_tol = 1e-5
        self.dpq_tol = 1e-5
        self._cc = cc
        self.nocc = cc.nocc
        self.nmo = cc.nmo
        self.max_memory = cc.max_memory
        self.verbose = cc.verbose

    def solve_2pgf(self,ps,qs,rs,ss,omega_list,broadening):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        if not isinstance(rs, collections.Iterable): rs = [rs]
        if not isinstance(ss, collections.Iterable): ss = [ss]
        print "Calculating Two-Particle Greens Function"

        nocc, nvir = self._cc.t1.shape
        cc_dtype = self._cc.t1.dtype
        x0 = eom_rccsd.amplitudes_to_vector_ee(np.zeros((nocc,nvir),dtype=cc_dtype),np.zeros((nocc,nocc,nvir,nvir),dtype=cc_dtype))
        p0 = 0.0*x0 + 1.0
        imds = eom_rccsd._IMDS(self._cc)
        imds.make_ee()

        p0 = 0.0*x0 + 1.0
        n = len(x0)
        e_vectors = []
        e0s = np.zeros((len(qs),len(ss)))

        for nq,q in enumerate(qs):
            e_vectors.append([])
            for ns,s in enumerate(ss):
                e_vectors[nq].append([])
                e_vector = eom_rccsd.tdm_r_imd(self,q,s)
                idx = np.flatnonzero(e_vector)
                val = e_vector[idx]
                e_vectors[nq][ns] = [idx,val]
                e0s[nq,ns] = eom_rccsd.tdm_r0_imd(self,q,s)

        gfvals = np.zeros((len(ps),len(qs),len(rs),len(ss),len(omega_list)),dtype=np.complex)
        for ip,p in enumerate(ps):
            for nr,r in enumerate(rs):

                if self.verbose > 3: print "Solve Linear System for p =",p,"r =",r
                b_vector = eom_rccsd.tdm_l_imd_bvec(self,p,r)
                for iomega in range(len(omega_list)):
                    curr_omega = omega_list[iomega]
                    def matr_multiply(vector,args=None):
                        return greens_func_multiply(eom_rccsd.eeccsd_matvec_singlet, self, imds, vector, -curr_omega-1j*broadening)

                    counter = gmres_counter()
                    H = scipy.sparse.linalg.LinearOperator((n,n), matvec = matr_multiply)
                    
                    # Preconditioner
                    P_diag = np.ones(n, dtype = np.complex)*-curr_omega-1j*broadening
                    P_diag += eom_rccsd.eeccsd_diag(self,imds=imds)[0]
                    P = scipy.sparse.diags(P_diag,format='csc')
                    M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
                    M = scipy.sparse.linalg.LinearOperator((n, n), M_x)

                    sol, info = scipy.sparse.linalg.gcrotmk(H,-b_vector, x0 = x0, tol = self.gmres_tol, M = M, callback = counter)
                    if self.verbose > 4 and info == 0: print 'Frequency', np.round(curr_omega,3), 'converged in', counter.niter, 'iterations'
                    if self.verbose > 4 and info != 0: print 'Frequency', np.round(curr_omega,3), 'not converged'

                    s0 = -np.dot(sol,eom_rccsd.amplitudes_to_vector_ee(self._cc.l1,self._cc.l2))

                    x0 = sol

                    for nq,q in enumerate(qs):
                       for ns,s in enumerate(ss):
                           idx,val = e_vectors[nq][ns]
                           total = np.dot(val,sol[idx])
                           total += e0s[nq,ns]*s0
                           gfvals[ip,nq,nr,ns,iomega] = total

        if len(ps) == 1 and len(qs) == 1 and len(rs) == 1 and len(ss) == 1:
            return gfvals[0,0,0,0,:]
        else:
            return gfvals

    def get_abs_spec(self,omega_list,broadening,dpq = None):
        if self.dpq_tol is None: self.dpq_tol = 1e-5
        print "Calculating Absorption Spectrum"

        nocc, nvir = self._cc.t1.shape
        ntot = nocc+nvir
        cc_dtype = self._cc.t1.dtype
        x0 = eom_rccsd.amplitudes_to_vector_ee(np.zeros((nocc,nvir),dtype=cc_dtype),np.zeros((nocc,nocc,nvir,nvir),dtype=cc_dtype))
        p0 = 0.0*x0 + 1.0
        imds = eom_rccsd._IMDS(self._cc)
        imds.make_ee()

        n = len(x0)
        x0 = np.zeros((3,n),dtype=np.complex)
        e_vector = np.zeros((3,n),dtype=np.complex)
        e0s = np.zeros(3,dtype=np.complex)
        
        if dpq is None:
            #----------------Calculate dpq--------------------------
            self._cc.mol.set_common_orig((0.0,0.0,0.0))
            ao_dipole = self._cc.mol.intor_symmetric('int1e_r', comp=3)
            occidx = np.where(self._cc._scf.mo_occ==2)[0]
            viridx = np.where(self._cc._scf.mo_occ==0)[0]
            mo_coeff = self._cc._scf.mo_coeff
            dpq = np.einsum('xaq,qi->xai', np.einsum('pa,xpq->xaq', mo_coeff, ao_dipole), mo_coeff)
            dpq = dpq.transpose(1,2,0)

        for q in range(ntot):
            for s in range(ntot):
                if (np.all(np.abs(dpq[q,s,:]) < self.dpq_tol)): continue
                e_vector += np.einsum('x,v->xv',dpq[q,s,:],eom_rccsd.tdm_r_imd(self,q,s))
                e0s += dpq[q,s,:]*eom_rccsd.tdm_r0_imd(self,q,s)

        spectrum = np.zeros((3,len(omega_list)),dtype=np.complex)
        b_vector = np.zeros((3,n),dtype=np.complex)

        for p in range(ntot):
            for r in range(ntot):
                if (np.all(np.abs(dpq[p,r,:]) < self.dpq_tol)): continue
                b_vector += np.einsum('x,v->xv',dpq[p,r,:],eom_rccsd.tdm_l_imd_bvec(self,p,r))
                
        if self.verbose > 4: print 'Vectors built'

        for iomega in range(len(omega_list)):
            curr_omega = omega_list[iomega]
            def matr_multiply(vector,args=None):
                return greens_func_multiply(eom_rccsd.eeccsd_matvec_singlet, self, imds, vector, -curr_omega-1j*broadening)

            counter = gmres_counter()
            H = scipy.sparse.linalg.LinearOperator((n,n), matvec = matr_multiply)
            
            # Preconditioner
            P_diag = np.ones(n, dtype = np.complex)*-curr_omega-1j*broadening
            P_diag += eom_rccsd.eeccsd_diag(self,imds=imds)[0]
            P = scipy.sparse.diags(P_diag,format='csc')
            M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
            M = scipy.sparse.linalg.LinearOperator((n, n), M_x)
            
            for i in range(3):
                sol, info = scipy.sparse.linalg.gcrotmk(H, -b_vector[i], x0 = x0[i], tol = self.gmres_tol, M = M, callback = counter)
                if self.verbose > 4 and info == 0: print 'Frequency', np.round(curr_omega,3), 'converged in', counter.niter, 'iterations'
                if self.verbose > 4 and info != 0: print 'Frequency', np.round(curr_omega,3), 'not converged'

                s0 = -np.dot(sol,self._cc.amplitudes_to_vector(self._cc.l1,self._cc.l2))
                x0[i] = sol
                spectrum[i,iomega] = np.dot(e_vector[i],sol) + e0s[i]*s0

        return -2*spectrum.imag


    def solve_ipgf(self,ps,qs,omega_list,broadening):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        print "Calculating IP Greens Function"

        nocc, nvir = self._cc.t1.shape
        cc_dtype = self._cc.t1.dtype
        x0 = eom_rccsd.amplitudes_to_vector_ip(np.zeros((nocc),dtype=cc_dtype),np.zeros((nocc,nocc,nvir),dtype=cc_dtype))
        p0 = 0.0*x0 + 1.0
        imds = eom_rccsd._IMDS(self._cc)
        imds.make_ip()

        e_vector = list()
        n = len(x0)
        for q in qs:
            e_vector.append(eom_rccsd.ip_tdm_r_imd(self,q))
        gfvals = np.zeros((len(ps),len(qs),len(omega_list)),dtype=np.complex)
        for ip,p in enumerate(ps):
            if self.verbose > 3: print "Solve Linear System for p =",p
            b_vector = eom_rccsd.ip_tdm_l_imd(self,p)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(eom_rccsd.ipccsd_matvec, self, imds, vector, curr_omega-1j*broadening)
                counter = gmres_counter()
                H = scipy.sparse.linalg.LinearOperator((n,n), matvec = matr_multiply)
                
                # Preconditioner
                P_diag = np.ones(n, dtype = np.complex)*curr_omega-1j*broadening
                P_diag += eom_rccsd.ipccsd_diag(self,imds=imds)
                P = scipy.sparse.diags(P_diag,format='csc')
                M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
                M = scipy.sparse.linalg.LinearOperator((n, n), M_x)
                
                sol, info = scipy.sparse.linalg.gcrotmk(H,-b_vector, x0 = x0, tol = self.gmres_tol, M = M, callback = counter)

                if self.verbose > 4 and info == 0: print 'Frequency', np.round(curr_omega,3), 'converged in', counter.niter, 'iterations'
                if self.verbose > 4 and info != 0: print 'Frequency', np.round(curr_omega,3), 'not converged'

                x0  = sol
                for iq,q in enumerate(qs):
                    gfvals[ip,iq,iomega]  = -np.dot(e_vector[iq],sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals
    
    def get_ip_spec(self,ws,eta):
        nocc, nvir = self._cc.t1.shape
        ntot = nocc+nvir
        IPgf = self.solve_ipgf(range(ntot),range(ntot),ws,eta)
        spectrum = np.zeros(len(ws))
        for p in range(ntot):
            spectrum -= IPgf[p,p,:].imag
            
        return spectrum

    def solve_2ppegf(self,ps,qs,omega_list,broadening,r_ee,l_ee,e):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        print "Calculating 2PPE Greens Function"

        nocc, nvir = self._cc.t1.shape
        nmo = nocc+nvir
        cc_dtype = self._cc.t1.dtype
        x0 = eom_rccsd.amplitudes_to_vector_ip(np.zeros((nocc),dtype=cc_dtype),np.zeros((nocc,nocc,nvir),dtype=cc_dtype))
        p0 = 0.0*x0 + 1.0
        imds = eom_rccsd._IMDS(self._cc)
        imds.make_ip()

        n = len(x0)
        e_vector = list()
        l1_ee,l2_ee = eom_rccsd.vector_to_amplitudes_ee(l_ee,nmo,nocc)
        r1_ee,r2_ee = eom_rccsd.vector_to_amplitudes_ee(r_ee,nmo,nocc)

        for q in qs:
            e_vector.append(eom_rccsd.tppe_tdm_r_imd(self,q,l1_ee,l2_ee))

        gfvals = np.zeros((len(ps),len(qs),len(omega_list)),dtype=np.complex)
        for ip,p in enumerate(ps):
            if self.verbose > 3: print "Solve Linear System for p =",p
            b_vector = eom_rccsd.tppe_tdm_l_imd(self,p,r1_ee,r2_ee)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]
                def matr_multiply(vector,args=None):
                    return greens_func_multiply(eom_rccsd.ipccsd_matvec, self, imds, vector, curr_omega-e-1j*broadening)

                counter = gmres_counter()
                H = scipy.sparse.linalg.LinearOperator((n,n), matvec = matr_multiply)
                
                # Preconditioner
                P_diag = np.ones(n, dtype = np.complex)*curr_omega-e-1j*broadening
                P_diag += eom_rccsd.ipccsd_diag(self,imds=imds)
                P = scipy.sparse.diags(P_diag,format='csc')
                M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
                M = scipy.sparse.linalg.LinearOperator((n, n), M_x)
                
                sol, info = scipy.sparse.linalg.gcrotmk(H,-b_vector, x0 = x0, tol = self.gmres_tol, M = M, callback = counter)
                if self.verbose > 4 and info == 0: print 'Frequency', np.round(curr_omega,3), 'converged in', counter.niter, 'iterations'
                if self.verbose > 4 and info != 0: print 'Frequency', np.round(curr_omega,3), 'not converged'

                x0  = sol
                for iq,q in enumerate(qs):
                    gfvals[ip,iq,iomega]  = np.dot(e_vector[iq],sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals

    def get_2ppe_spec(self,ws,eta,r_ee,l_ee,e_r):
        nocc, nvir = self._cc.t1.shape
        ntot = nocc+nvir
        TPPEgf = self.solve_2ppegf(range(ntot),range(ntot),ws,eta,r_ee,l_ee,e_r)
        spectrum = np.zeros(len(ws))
        for p in range(ntot):
            spectrum -= TPPEgf[p,p,:].imag
            
        return spectrum

    def solve_eagf(self,ps,qs,omega_list,broadening):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        print "Calculating EA Greens Function"

        nocc, nvir = self._cc.t1.shape
        cc_dtype = self._cc.t1.dtype
        x0 = eom_rccsd.amplitudes_to_vector_ea(np.zeros((nvir),dtype=cc_dtype),np.zeros((nocc,nvir,nvir),dtype=cc_dtype))
        p0 = 0.0*x0 + 1.0
        imds = eom_rccsd._IMDS(self._cc)
        imds.make_ea()

        e_vector = list()
        n = len(x0)
        for q in qs:
            e_vector.append(eom_rccsd.ea_tdm_r_imd(self,q))
        gfvals = np.zeros((len(ps),len(qs),len(omega_list)),dtype=np.complex)
        for ip,p in enumerate(ps):
            if self.verbose > 3: print "Solve Linear System for p =",p
            b_vector = eom_rccsd.ea_tdm_l_imd(self,p)
            for iomega in range(len(omega_list)):
                curr_omega = omega_list[iomega]
                def matr_multealy(vector,args=None):
                    return greens_func_multiply(eom_rccsd.eaccsd_matvec, self, imds, vector, -curr_omega-1j*broadening)
                counter = gmres_counter()
                H = scipy.sparse.linalg.LinearOperator((n,n), matvec = matr_multealy)
                
                # Preconditioner
                P_diag = np.ones(n, dtype = np.complex)*-curr_omega-1j*broadening
                P_diag += eom_rccsd.eaccsd_diag(self,imds=imds)
                P = scipy.sparse.diags(P_diag,format='csc')
                M_x = lambda x: scipy.sparse.linalg.spsolve(P, x)
                M = scipy.sparse.linalg.LinearOperator((n, n), M_x)

                sol, info = scipy.sparse.linalg.gcrotmk(H,b_vector, x0 = x0, tol = self.gmres_tol, M = M, callback = counter)
                if self.verbose > 4 and info == 0: print 'Frequency', np.round(curr_omega,3), 'converged in', counter.niter, 'iterations'
                if self.verbose > 4 and info != 0: print 'Frequency', np.round(curr_omega,3), 'not converged'
                x0  = sol
                for iq,q in enumerate(qs):
                    gfvals[ip,iq,iomega]  = -np.dot(e_vector[iq],sol)
        if len(ps) == 1 and len(qs) == 1:
            return gfvals[0,0,:]
        else:
            return gfvals
            
    def get_ea_spec(self,ws,eta):
        nocc, nvir = self._cc.t1.shape
        ntot = nocc+nvir
        EAgf = self.solve_eagf(range(ntot),range(ntot),ws,eta)
        spectrum = np.zeros(len(ws))
        for p in range(ntot):
            spectrum -= EAgf[p,p,:].imag
            
        return spectrum


class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
