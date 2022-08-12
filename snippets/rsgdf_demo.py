import sys
from pyscf.pbc import gto, df, scf
from pyscf.cc import eom_rccsd
import surf
from regional_embedding import emb
import ccg

if __name__ == "__main__":
    args = surf.getArgs(sys.argv[1:])
    fs = args.f[:]
    args.f = args.f[0]
    args.verbose = 6
    args,cell,mf = surf.init_calc(args)
    mf.direct_scf = True

    # RSGDF init
    # with_df.precision_R = 1e-10 # or 1e-12 etc. default is cell.precision which is 1e-8
    #with_df = df.RSDF(cell)
    """ RSGDF relies on a parameter "omega" to balance the cost between short-range (SR) and long-range (LR) parts. The general trend is larger omega --> faster SR and slower LR, and smaller omega has the opposite effect. It's recommended to test a few values of omega between 0.1 ~ 1 and see
        1. if the time vs omega plot show a minimum (indicate you find the optimum choice)
        2. if different choices of omega give you same answer (SCF energy, CIS excitation energy, CCSD correlation energy etc).
    """
    #with_df.omega = 0.5 # just an example
    #with_df.build()
    """ In the output file, grepping
            "CPU time for 3c2e" for building the SR part (same for GDF!)
            "CPU time for j3c"  for full DF init time (same for GDF!)
        The difference between them is the CPU time for building the LR part.
    """

    # HF with RSGDF
    #mf = scf.RHF(cell)
    #mf.with_df = with_df
    #mf.kernel()

    # HF with GDF
    #mf0 = scf.RHF(cell).density_fit()
    e_tot0 = mf.kernel()

    with_df = df.RSDF(cell)
    for omega in [0.3, 0.5, 0.7, 1.0]:
        with_df.omega = omega # just an example
        with_df.build()
        mf.with_df = with_df
        mf.kernel()

        print(f'omega: {omega}')
        print("HF/GDF   etot = %.10f" % (e_tot0))
        print("HF/RSGDF etot = %.10f" % (mf.e_tot))
        print("Difference    = %.10f" % (abs(mf.e_tot-e_tot0)))

        for f in fs:
            if len(fs) > 1:
                args.f = f
                A, atom, pseudo, basis = surf.init_geom(args)
                cell.build(parse_arg = False,
                            a = A,
                            atom = atom,
                            pseudo = pseudo,
                            basis = basis)
            frozen = None
            ro_coeff = None
            if f > 0:
                ro_coeff, frozen, emb_dat = emb.make_rdmet(cell,mf,args.ft,'!',args.minao)
                ro_coeff, ro_energy = emb.canonicalize(mf,ro_coeff,frozen)
            mycc = ccg.DFCCSD(mf,frozen=frozen,mo_coeff=ro_coeff)
            eris = mycc.ao2mo()
            mycc.kernel(eris=eris)
            e_corr = mycc.e_corr
            myeom = eom_rccsd.EOMEESinglet(mycc)
            eome, eomv = myeom.kernel(eris=eris)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Running surf/defects for omega: {omega} and {f} in {fs}: e_corr: {e_corr}, eom: {eome}')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
