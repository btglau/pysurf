# natural orbital iteration
# algorithm:
# 1) SCF
# 2) Frozen orbital calculation
# 3) Reassemble 1RDM
# 4) Regenerate Fock matrix with this 1RDM
# x) diagonalize for new set of MOs? NO?
# 5) Frozen NO calculation
# 6) Iterate to energy and NO orbital convergence

import sys
import surf
import ccg
import numpy

try:
    from regional_embedding import emb
    from pyscf import lib
except ImportError:
    print('Testing on a machine without pyscf')

if __name__ == '__main__':

    max_cycle = 20

    print(sys.version)

    # initiate
    args = surf.getArgs(sys.argv[1:])
    args.f = args.f[0]
    args.ft = args.ft[0]
    args,cell,mf = surf.init_calc(args)
    surfout = {'E':{},'C':{},'conv':{}}

    # first partitioning of orbital space
    surfout,args,ro_coeff,frozen = surf.embed(surfout,args,cell,mf)

    ro_coeff = None
    e_corr_last = 1
    e_corr_now = 0
    conv_tol = 1e-5
    s1e = mf.get_ovlp()
    NMg = None
    if 'mgo' in args.sys and 'gthccpv' in args.b.replace('-','').lower() and args.frcore:
        NMg = sum([1 for at in cell.atom if 'Mg' in at[0]]) * 4
    for a in range(max_cycle):
        # embed
        ro_coeff, frozen, emb_dat = emb.make_rdmet(cell,mf.mo_coeff,mf.mo_occ,NMg,args.ft,'!',args.minao)
        # canonicalize (assume there's a chk file)
        if a == 0:
            mf_or_fock = numpy.load(args.chk_path + '.npy')
        else:
            mf_or_fock = fock
        ro_coeff, ro_energy = emb.canonicalize(mf_or_fock,ro_coeff,frozen)

        # correlated calculation
        mycc = ccg.DFCCSD(mf,frozen=frozen,mo_coeff=ro_coeff)
        mycc.keep_exxdiv = args.ke
        eris = mycc.ao2mo()
        mycc.kernel(eris=eris)

        # check for convergence
        e_corr_last = e_corr_now
        e_corr_now = mycc.e_corr
        print(f'NOIT cycle = {a}, dE = {e_corr_now - e_corr_last}, E_corr = {e_corr_now} E_corr_last = {e_corr_last}')
        if abs(e_corr_now - e_corr_last) < conv_tol:
            print('converged')
            break
        
        # new set of orbitals
        print('update orbitals')
        mycc.solve_lambda(eris=eris)
        y = mycc.make_rdm1(ao_repr=True)
        fock = mf.get_fock(dm=y)
        mf.mo_energy, mf.mo_coeff = mf.eig(fock, s1e)
        with lib.temporary_env(mf, verbose=0):
            mf.mo_occ = mf.get_occ()

    print('End of cycle')
    print(f'E_corr final = {mycc.e_corr}')