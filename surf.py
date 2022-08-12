"""
Created on Fri Oct 25 10:10:57 2019
@author: blau

assumes:

^ ---> ceph ---> output_chk
|
| ---> output_surf/output_cube
|
folder with surf.py

"""

import os
import argparse
import sys
import time
import copy
import shutil
import itertools
#import traceback, signal
import scipy.io as sio
from scipy.linalg import sqrtm
import numpy

try:
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    from pyscf.pbc import gto
    from pyscf.pbc import mp
    from pyscf.pbc import scf
    from pyscf.pbc import df
    from pyscf.pbc import tdscf
    from pyscf.pbc import tools
    from pyscf import mp as mmp # molecular mp
    #from pyscf.data.elements import _std_symbol_without_ghost
    from pyscf import lib
    from functools import reduce
    # more memory efficient ccsd molecular solver for Gamma-point df object
    import ccg
    import rcis
    from regional_embedding import emb
    from pyscf.data.elements import _symbol
    BOHR = 0.52917721092
except ImportError:
    print('Testing on a machine without pyscf')

def getArgs(argv):
    '''
    get args from command line sys.argv[1:]
    '''
    parser = argparse.ArgumentParser(description="Setup a surface calculation")
    # electronic structure arguments
    parser.add_argument('-b',help="basis set (string)",default='gth-dzvp')
    parser.add_argument('--bw',help="water basis set (string)",default=None)
    parser.add_argument('--bf',help="fragment basis set (excludes water) (string)",default=None)
    parser.add_argument('-p',help="Pseudo basis (effective core potential)",default='gth-pade')
    parser.add_argument('--pr',help="precision of ewald and lattice sums",default=None,type=float)
    parser.add_argument('--prrs',help="precision for RSDF (relative to cell.precision / args.pr)",default=None,type=float)
    parser.add_argument('--df',help="Density fitting (default: GDF for pp/ecp)",default='GDF')
    parser.add_argument('--dflocal',help="Copy df tensor to local scratch",default=0,type=int)
    parser.add_argument('-d',help='Dimension (default:3), whether to use pyscf tricks for low-d',default=3,type=int)
    parser.add_argument('-e',help='''exxdiv type:
                                     None: V_(G=0)=0
                                     default 'ewald': point-charge Madelung correction
                                     'vcut_sph': truncate C on a sphere
                                     'vcut_ws': should be the best, truncate C at WS cell 
                                  ''',default='ewald')
    parser.add_argument('--ke',help="keep exxdiv",default=False,type=bool)
    parser.add_argument('-l',help='''low_dim_ft_type:
                                     default None: 2D analytic, inf_vacuum 0D and 1D
                                     'inf_vacuum': 0D and 1D
                                     ''',default=None)
    parser.add_argument('-n',help='''do a band calculation with n points per direction 
                                     (should only do with pristine surface)''',default=0,type=int)
    parser.add_argument('-T',
                        help='''
                        levels of post-HF theory to use, dash separated string:
                            - Coupled-cluster (CCSD)
                            -- Add 'T'for CCSD(T)
                            -- Add "EE" for EOM-EE-CCSD
                            - Coupled-cluster (CCT)
                            - Moller-Plesset 2 (MP)
                            - Self-consistent regional embedding (SCRE)
                            - 'STOP' only builds the cell and mf
                            - default value of '' just sets the filepath for .scf and .df
                              - Also runs SCF (if it doesn't find a chk file)
                              - should be paired with load scf to get <r> and <r^2>
                        '''
                        ,default='')
    parser.add_argument('--corr',help="Comma-separated pair of PHF methods for SCRE: CCSD/MP2/HF(no PHF)",default=None)
    parser.add_argument('--minao',help="Default: MINAO/gth-szv/same (use computational basis for occ)",default='minao')
    parser.add_argument('-f',help='''size of the fragment for embedding
                                     atoms are selected based upon a region where a characteristic length scale
                                     is multiplied by f (fraction)
                                     can select by SPHERE or by UNIT CELL (see --emb)
                                     can give a list of values separated by ','
                                  ''',default='0')
    parser.add_argument('--ft',help='''embedding eigenvalue threshold tuple(#,#)
                                    = (decimal,decimal) selects based on eigenvalue cutoff
                                    = (int,int) directly selects orbitals starting from largest ev
                                        also is used for selecting orbitals to freeze in other bases
                                    can give a list of pairs separated by ',,'
                                    ''',default=None)
    parser.add_argument('--ghost',help='''ghost atom calculations
                                          0: substrate + absorbate / pristine crystal
                                          1: substrate with virtual defect
                                             if f == 0, essentially a way to build a g supercell
                                             if f != 0, define a fragment of g as if water was there
                                          2: build substrate, include ghost absorbate / crystal w/ defect
                                          3: build ghost substrate, include absorbate
                                        ''',default=0,type=int)
    parser.add_argument('--etd',help="exponent to discard",default=0,type=float)
    parser.add_argument('--dfccsd',help="dfccsd with exxdiv=None",default=1,type=int)
    parser.add_argument('--can',help='''canonicalize fragment orbitals
                                        the default is ON because  the number of iterations to converge
                                        fraction calculations is restored to a the same number
                                        as CCSD with a HF reference - since pyscf assumes
                                        a canonical form of the Fock matrix''',default=1,type=int)
    parser.add_argument('--remlindep',help="Use remove_linear_dep_",default=0,type=int)
    parser.add_argument('--stab',help="Stability analysis",default=0,type=int)
    parser.add_argument('-r',help="Number of roots (excited state)",default=None,type=int)
    parser.add_argument('--nr',help="Which root to visualize for eom nto",default=0,type=int)
    parser.add_argument('--ericis',help="Enable restricted, ERI-based CIS",default=0,type=int)
    parser.add_argument('--nto',help="CIS NTO or EOM R1 NTO (also will make NTO cube files)",default=0,type=int)
    parser.add_argument('--eomnto',help="Also make EOMEE TDM vs. r1 eomnto",default=0,type=int)
    parser.add_argument('--frcore',help="Freeze core for MgO (specifically Mg, cc-HZ)",default=1,type=int)
    parser.add_argument('--frcore_nto',help="Freeze core for MgO CIS NTOs",default=0,type=int)
    parser.add_argument('--mc',help="Max cycle",default=0,type=int)
    parser.add_argument('--emb',help='''Type of embedding
                                        RES and REC can be combined with NTO, NO, or DSCF
                                        RES : regional embedding, select by sphere
                                        REC : regional embedding, select by unit cell
                                        FE : frozen energy
                                        NTO : natural transition orbitals
                                        NO : natural orbitals
                                        DSCF : delta-SCF
                                        FEX : frozen energy, ordered by contribution to defect''',default='')
    parser.add_argument('--blockdim',help="blockdim for pyscf pbc/df (default:240)",default=0,type=int)
    parser.add_argument('--resymb',help="regional embedding symbol",default='!')

    # chemical system arguments
    parser.add_argument('-v',help="Vacuum (angstrom)",default=20,type=float)
    parser.add_argument('-z',help="Height of adsorbate (angstrom)",default=0,type=float)
    parser.add_argument('-w',help="Place a water (int: substrate supercell tiling)",default=0,type=int)
    parser.add_argument('-a',help="only adsorbate calculation (1 - water, 2 - carbon)",default=0,type=int)
    parser.add_argument('-g',help="gruneis alternate geometry",default=0,type=int)
    parser.add_argument('--hbn',help="water @ h-BN (deprecated)",default=0,type=int)
    parser.add_argument('--lih',help="water @ LiH (deprecated)",default=0,type=int)
    parser.add_argument('--sys',help="systems: g, hbn, lih; mgo, lif, d, si",default='g')

    # other
    parser.add_argument('--log',help='''advanced logging: cpu usage''',default=0,type=int)
    parser.add_argument('-j',help="file name to save (if called with nothing doesn't save)",default=None)
    parser.add_argument('--js',help="file name supplement",default=None)
    parser.add_argument('-c',help='cubegen, +c orbitals from the nocc or comma separated list',default='0')
    parser.add_argument('--verbose',help='pyscf verbosity',default=5,type=int)
    parser.add_argument('--chk',help='''chk name for both .scf and .df if either file is found
                                        it will stream it from disk, otherwise it will save a 
                                        .scf or .df file''',default='')
    parser.add_argument('--loadcell',help='load geometry from QE (filename)',default='')
    parser.add_argument('--slim',help='slim output (generally, don''t save coeffs)',default=1,type=int)
    parser.add_argument('--forceuhf',help='force UHF for debugging',default=False,type=bool)

    args = parser.parse_args(argv)
    args.JOBID = os.getenv('SLURM_JOB_ID',default=None)

    if args.js:
        args.j += args.js

    args.f = tuple([float(e) for e in args.f.split(',')])
    
    # Change any string None to python None
    for a in [a for a in dir(args) if not a.startswith('_')]:
        if getattr(args,a) == 'None':
            setattr(args,a,None)
    
    if ('gth' not in args.b.lower()) and ('ecp' not in args.b.lower()):
        args.p = None
        print('GTH or ECP not selected, pseudo is turned off!')
        print('Basis: ' + args.b)
            
    if args.ft is None:
        args.ft = ['0.1,0.1']
    else:
        args.ft = args.ft.split(',,')
    args.ft = [list(map(float,e.split(','))) for e in args.ft]
    args.ft = tuple([[int(x) if x >=1 else x for x in lst] for lst in args.ft])

    if args.corr is None:
        args.corr = ('HF','MP2')

    if args.c.split(',')[0] == args.c:
        args.c = int(args.c)
    else:
        args.c = tuple(map(int,args.c.split(',')))

    outdir = '../ceph/output_chk/'
    if args.chk:
        args.chk_path = os.path.normpath(outdir + args.chk)

    if 'GDF' in args.chk:
        args.df = 'GDF'
    if 'RSDF' in args.chk:
        args.df = 'RSDF'

    print(args)
    return args

def init_surf(args):
    '''
    Initialize the geometry of a surface + adsorbate
    
    Atoms in the embedding cell are suffixed with '!'

    '''
    
    # unpack args for build() parameters
    pseudo = args.p
    basis = {'default':args.b}
    
    if args.sys == 'g':
        # bond length in graphene: 1.42 A
        # c = 6.708 (graphite) 3.354 (A-B layer distance)
        # total energy: 1037 eV/atom
        a12 = 2.46 # in-plane lattice constant (angstroms)
        r3 = numpy.sqrt(3)
        # lattice vectors of graphene
        a1 = [a12*r3/2, -a12/2, 0.]
        a2 = [a12*r3/2,  a12/2, 0.]
        # carbon atom basis
        c1 = [0.,         0.,    0.]
        c2 = [a12/(2*r3), a12/2, 0.]
        if args.g:
            ### gruneis
            a1 = [2.465012,0.,0.]
            a2 = [-1.232506,2.134753,0.]
            c2 = [0.,1.423169,0.]
        a1 = numpy.asarray(a1)
        a2 = numpy.asarray(a2)
        c1 = numpy.asarray(c1)
        c2 = numpy.asarray(c2)
        
        # primitive graphene cell
        atom = [['C',c1],
                ['C',c2]]
        
        # water in 0 leg inside a benzene motif
        h2o = [['O', numpy.asarray([1.42  , 0., 3.1000 + args.z])],
               ['H', numpy.asarray([0.6537, 0., 3.6863 + args.z])],
               ['H', numpy.asarray([2.1862, 0., 3.6863 + args.z])]
              ]
        if args.g:
            ### gruneis
            h2o = [['O', numpy.asarray([1.232506,  0.711584,  3.1000000 + args.z])],
                   ['H', numpy.asarray([0.568865,  0.328513,  3.6863446 + args.z])],
                   ['H', numpy.asarray([1.896094,  1.094656,  3.6863446 + args.z])]
                  ]
    elif args.sys == 'hbn':
        # h-BN
        a1 = [2.512,0,0]
        a2 = [-1.256,2.17545581430651,0]
        a1 = numpy.asarray(a1)
        a2 = numpy.asarray(a2)
        c1 = a1/3 + a2*2/3
        c2 = a1*2/3 + a2/3
        
        atom = [['B',c1],
                ['N',c2]]
        # H of water on top of N
        h2o = [['H', numpy.asarray([1.326,   0.755151938102170,  2.27 + args.z])],
               ['H', numpy.asarray([0.816,   0.585151938102170,  3.72 + args.z])],
               ['O', numpy.asarray([1.596,   0.855151938102170,  3.2 + args.z])]
              ]
    elif args.sys == 'lih':
        ac = 4.084
        a = numpy.sqrt(ac**2/2)
        a1 = [a,0,0]
        a2 = [0,a,0]
        a1 = numpy.asarray(a1)
        a2 = numpy.asarray(a2)
        c1 = numpy.zeros(3)
        c2 = a1/2 + a2/2
        
        atom = [['H',c1],
                ['Li',c2]]
        h2o = [['O', numpy.asarray([1.532717715, 1.443911943, 4.193270986 + args.z - ac/2])],
               ['H', numpy.asarray([2.127302827, 0.674412056, 4.089802120 + args.z - ac/2])],
               ['H', numpy.asarray([2.127302827, 2.213411788, 4.089802120 + args.z - ac/2])]
              ]
    
    # label water: !! to distingish H2O from LiH for basis set
    # search_ao_label will still pick up !! as !
    for at in h2o:
        at[0] = at[0] + '!!'
            
    if args.ghost:
        #    pseudo = {'default':pseudo}
        if args.ghost == 2:
            # h2o ghost
            for at in h2o:
                at[0] = 'ghost-' + at[0]
                #pseudo[at[0]] = args.p
            
        if args.ghost == 3:
            # lattice ghost
            for at in atom:
                at[0] = 'ghost-' + at[0]
                #pseudo[at[0]] = args.p
                
    if args.bw:
        for at in h2o:
            basis[at[0]] = args.bw
    
    if args.bf:
        for at in atom:
            basis[at[0] + '!'] = args.bf
            
    # generate the supercell
    if args.w:
        # change: use the first atom of H2O to define center
        h2oc = numpy.zeros((4,3))
        h2oc[0,:] = h2o[0][1]
        h2oc[1,:] = h2oc[0,:] + args.w*a1
        h2oc[2,:] = h2oc[0,:] + args.w*a2
        h2oc[3,:] = h2oc[0,:] + args.w*(a1+a2)
        h2oc[:,2] = 0 # set z coord to 0
        
        # select fragment based upon fractional unit cell
        if 'REC' in args.emb:
            frag_poly = numpy.zeros((4,3))
            frag_poly[1,:] = a1
            frag_poly[2,:] = a1 + a2
            frag_poly[3,:] = a2
            frag_poly2 = abs(args.f)*(frag_poly - (a1+a2)/2)
            if args.f < 0:
                fragc = frag_poly*args.w
            elif args.f > 0:
                fragc = h2oc
            frag_polys = []
            for ix in range(4): # omit z-axis
                tmp = frag_poly2 + fragc[ix,:]
                frag_polys.append(Polygon([tuple(coord) for coord in tmp[:,:2]]))
        
        # radius of circle equal to a1 (or a2)
        sr = numpy.linalg.norm(args.w*a1)

        c1_str = atom[0][0]
        c2_str = atom[1][0]
        atom = []
        for R1 in range(args.w):
            for R2 in range(args.w):
                c1_str_suffix = ''
                c2_str_suffix = ''
                # R = x1*a1 + x2*a2
                R = R1*a1 + R2*a2
                # each lattice point has 2 C motif
                if 'RES' in args.emb:
                    # select C atoms based on distance from h2o centroid
                    # check distance against h2oc and periodic copies at a1, a2, a1+a2
                    if min([numpy.linalg.norm(h2oc_-(R+c1)) for h2oc_ in h2oc]) <= args.f*sr:
                        c1_str_suffix = '!'
                    if min([numpy.linalg.norm(h2oc_-(R+c2)) for h2oc_ in h2oc]) <= args.f*sr:
                        c2_str_suffix = '!'
                elif 'REC' in args.emb:
                    for polygon in frag_polys: # omit z-axis
                        if polygon.contains(Point(tuple((R+c1)[:2]))):
                            c1_str_suffix = '!'
                        if polygon.contains(Point(tuple((R+c2)[:2]))):
                            c2_str_suffix = '!'
                        
                atom.append([c1_str + c1_str_suffix,R+c1])
                atom.append([c2_str + c2_str_suffix,R+c2])
                
                if args.sys == 'lih':
                    # second layer, swap positions of Li and H translate z by ac/2
                    atom.append([c2_str + c1_str_suffix,R+c1+[0,0,-ac/2]])
                    atom.append([c1_str + c2_str_suffix,R+c2+[0,0,-ac/2]])
                
        if args.ghost != 1:
            atom.extend(h2o)
        
        # set the dimensions of the supercell
        a1 *= args.w
        a2 *= args.w
    
    Lz = args.z + args.v
    if args.z > args.v:
        Lz = 2*args.z
    a3 = numpy.asarray([0., 0., Lz])
    a = numpy.vstack((a1,a2,a3))
    
    if args.a:
        # override periodic calculation with an atom/molecule in vacuum
        if args.a == 1:
            atom = h2o # orientation doesn't really matter in a vacuum
        if args.a == 2:
            atom = [['C',0.,0.,0.]]
        a = numpy.eye(3)

    return a, atom, pseudo, basis

def init_fcen(args):
    '''
    F-center in LiF, MgO (not sure why it's called F-center in MgO)
    Update: F - Farbe center
    '''
    
    # unpack args for build() parameters
    pseudo = args.p
    
    # see matlab code for references on the value of a
    if 'lif' in args.sys:
        a = 4.03
        b = a/2
        cat = 'Li'
        ani = 'F'
    elif 'mgo' in args.sys:
        a = 4.2568
        b = a/2
        cat = 'Mg'
        ani = 'O'
    elif 'd' in args.sys:
        a = 3.567095
        b = a/4
        cat = 'C'
        ani = 'C'
    basis = {ani:args.b,cat:args.b}
    
    # Mg and Li are in MOLOPT-SR, not MOLOPT
    # diamond can use any of the pbc bases
    if 'mgo' in args.sys or 'lif' in args.sys:
        if 'molopt' in args.b:
            # check for diffuse functions (from the UCL MOLOPT series)
            if 'pd-molopt' in args.b:
                ani_basis = args.b.split('-')
                ani_basis[0] = ani_basis[0][:-1] # remove 'd'
                ani_basis = ''.join(ani_basis)
                basis = {ani:ani_basis,cat:args.b+'sr'}
            else:
                basis = {ani:args.b,cat:args.b+'sr'}
    
    if 'p' in args.sys:
        # rhombohedral unit cell
        A = a/2*numpy.asarray([
            [1,0,1],
            [1,1,0],
            [0,1,1]]).T
        uc = [[ani,numpy.asarray([0.,0.,0.])],
              [cat,numpy.asarray([b,b,b])]]
    else:
        if 'd' in args.sys:
            # zinc blende
            uc = [[ani,numpy.asarray([0.,0.,0.])],
              [ani,numpy.asarray([a/2,a/2,0.])],
              [ani,numpy.asarray([0.,a/2,a/2])],
              [ani,numpy.asarray([a/2,0.,a/2])],
              [cat,numpy.asarray([b ,b ,b])],
              [cat,numpy.asarray([3*b,3*b,b])],
              [cat,numpy.asarray([3*b,b,3*b])],
              [cat,numpy.asarray([b,3*b,3*b])]]
        else:
            # rock salt
            uc = [[ani,numpy.asarray([0.,0.,0.])],
              [ani,numpy.asarray([a/2,a/2,0.])],
              [ani,numpy.asarray([0.,a/2,a/2])],
              [ani,numpy.asarray([a/2,0.,a/2])],
              [cat,numpy.asarray([b ,b ,b])],
              [cat,numpy.asarray([b ,0.,0.])],
              [cat,numpy.asarray([0.,b ,0.])],
              [cat,numpy.asarray([0.,0.,b])]]
        A = numpy.eye(3)*a
    
    atom = uc
    if args.w:
        atom = []
        corners = [
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1]]
        corners = numpy.asarray(corners) @ A * args.w
        # for each atom, the vector pointing to the nearest corner
        acorner = []
        
        for R1 in range(args.w):
            for R2 in range(args.w):
                for R3 in range(args.w):
                    R = R1*A[0,:] + R2*A[1,:] + R3*A[2,:]
                    for at in uc:
                        attmp = copy.deepcopy(at)
                        attmp[1] += R
                        atom.append(attmp)
                        # find the closest vector pointing from atom to the F-center
                        acorner.append(corners[numpy.linalg.norm(corners-attmp[1],axis=1).argmin()] - attmp[1])

        # ghost = 0 is the full lattice
        # ghost = 3 isolated vacancy atom with ghost lattice basis
        if args.ghost == 3:
            for at in atom[1:]:
                at[0] = 'ghost-' + at[0]
            basis['ghost-' + cat] = basis[cat]
            basis['ghost-' + ani] = basis[ani]
        
        # ghost 2: F-center; first entry in atom is ghost
        if args.ghost == 2:
            atom[0][0] = 'ghost-' + atom[0][0]
            if args.bw:
                basis[atom[0][0]] = args.bw
            else:
                basis[atom[0][0]] = basis[ani]

            if 'r' in args.sys:
                # split atom in cation and anion, sort them by distance, take the
                # first 6 entries for cation, and entries 1:13 (12 next nearest) for anion,
                # shift them, then splice them back into a list. The actual order of
                # atom doesn't matter
                acorner_norm = numpy.asarray([numpy.linalg.norm(ac) for ac in acorner])
                atom = [[x[0],x[1],y,z] for x,y,z in zip(atom,acorner,acorner_norm)]
                cat_ind = numpy.asarray([cat in x[0] for x in atom])

                cat_atom = list(itertools.compress(atom,cat_ind))
                ani_atom = list(itertools.compress(atom,~cat_ind))

                cat_norm = acorner_norm[cat_ind]
                ani_norm = acorner_norm[~cat_ind]

                cat_atom = [cat_atom[i] for i in cat_norm.argsort()]
                ani_atom = [ani_atom[i] for i in ani_norm.argsort()]

                if 'mgo' in args.sys:
                    # outward movement, numbers from Rinke, Van de Walle 10.1103/PhysRevLett.108.126404
                    cat_shift = 0.12
                    ani_shift = 0.008
                elif 'd' in args.sys:
                    # it is unclear whether there is an outward or inward shift
                    cat_shift = 0
                    ani_shift = 0
                elif 'lif' in args.sys:
                    cat_shift = 0.04
                    ani_shift = 0.01

                for i in range(6):
                    # cation nearest neighbour
                    cat_atom[i][1] -= cat_shift * cat_atom[i][2]/cat_atom[i][3]
                for i in range(1,13):
                    # anion next nearest neighbours
                    ani_atom[i][1] -= ani_shift * ani_atom[i][2]/ani_atom[i][3]

                atom = ani_atom + cat_atom
                atom = [x[0:2] for x in atom]

        A *= args.w

    return A, atom, pseudo, basis

def init_fcen2(args):
    '''
    Load geometry from a file. Assumes file is in the same folder as this script

    Ghost atom is labeled in the json, so args.ghost has no meaning in this fn
    '''
    import json
    with open(os.path.normpath('./geometries/' + args.loadcell) + '.json') as f:
        loadcell = json.load(f)
        ani = loadcell['ani']
        cat = loadcell['cat']
        atom = loadcell['atom']
        # the unit cell in QE and the unit cell I use to generate the geometries do
        # not match, they differ by a reflection in the yz plane. I use the pyscf a
        # even though the optimization is done with the QE unit cell - pyscf does
        # not like the QE with the negative basis vector (check MATLAB code)
        A = loadcell['A']

    if args.z:
        A[-1][-1] += args.z

    # convert from string representation to ['atom',ndarray(coords)]
    for idx in range(len(atom)):
        at = atom[idx].split()
        atom[idx] = [at[0],numpy.asarray(at[1:],dtype=numpy.float64)]

    # unpack args for build() parameters
    pseudo = args.p
    basis = dict()
    if ani:
        basis[cat] = args.b
        basis[ani] = args.b
        basis['ghost-' + ani] = basis[ani]
    else:
        for at in cat:
            basis[at] = args.b
    
    # Stitch together MOLOPT bases if specified - DEPRECATED
    if 'mgo' in args.sys or 'lif' in args.sys:
        if 'molopt' in args.b:
            # check for diffuse functions (from the UCL MOLOPT series)
            if 'pd-molopt' in args.b:
                ani_basis = args.b.split('-')
                ani_basis[0] = ani_basis[0][:-1] # remove 'd'
                ani_basis = ''.join(ani_basis)
                basis = {ani:ani_basis,cat:args.b+'sr'}
            else:
                basis = {ani:args.b,cat:args.b+'sr'}

    return A, atom, pseudo, basis

def sphere_selector(f,corners,atom,label='!'):
    '''
    select atoms based on sphere set at origin and equivalent points
    '''
    ind = numpy.zeros(len(atom))
    atom_coords = numpy.asarray([at[1] for at in atom])
    # length of unit cell, any row works
    # should work if there is a vacuum (for slab/2D) - should ignore vacuum length
    s = numpy.linalg.norm(corners[1:4,:],axis=1).min()
    for c in corners:
        b = numpy.linalg.norm(atom_coords - c,axis=1)
        ind += b <= s*f
    for idx,i in enumerate(ind):
        if i:
            atom[idx][0] += label
    print(f'{sum(ind>0)} atoms in fragment')
    return atom

def cell_selector(f,corners,atom,label='!'):
    '''
    select atoms based on a mini-supercell set at the origin and equivalent points
    solves Ax=b, where
        b : positions of atoms
        A : basis vector scaled by f
    if 0 <= x <= 1, the atom is selected
    '''
    ind = numpy.zeros(len(atom))
    atom_coords = numpy.asarray([at[1] for at in atom])
    A = corners[1:4,:]
    # reverse corners
    rcorners = [
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1]]
    rcorners = -numpy.asarray(rcorners)
    rcorners[rcorners==0] = 1
    for idx,c in enumerate(corners):
        b = atom_coords - c
        x = numpy.linalg.solve(A*rcorners[idx,None]*f,b.T)
        # x close to zero is sometimes on the order of -(machine epsilon)
        x = numpy.absolute(x)
        ind += numpy.all((x >= 0) & (x <= 1),axis=0)
    for idx,i in enumerate(ind):
        if i:
            atom[idx][0] += label
    print(f'{sum(ind>0)} atoms in fragment')
    return atom

def init_geom(args):
    # setup geometry
    if args.loadcell:
        args.sys = args.loadcell.split('_')[0]
        cell_func = init_fcen2
    elif args.sys in ('g','hbn','lih'):
        cell_func = init_surf
    elif 'mgo' in args.sys or 'lif' in args.sys or 'd' in args.sys:
        # systems could have 'p' for primitive cell
        cell_func = init_fcen
    else:
        raise NotImplementedError(args.sys)
    A, atom, pseudo, basis = cell_func(args)

    if args.loadcell or 'mgo' in args.sys or 'lif' in args.sys or 'd' in args.sys or 'si' in args.sys:
        # select atoms to be in the fragment, label with '!'
        corners = [
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [1,0,1],
            [0,1,1],
            [1,1,1]]
        corners = numpy.asarray(corners) @ A

        if 'RE' in args.emb:
            # set basis for labeled atoms to be same as unlabeled counterparts
            for k in list(basis): # cast to list, or else 'dictionary changed in size'
                basis[k + '!'] = basis[k]
            if 'RES' in args.emb:
                atom = sphere_selector(args.f,corners,atom)
            elif 'REC' in args.emb:
                atom = cell_selector(args.f,corners,atom)
        
        if args.bf:
            f = 0
            for k in list(basis):
                basis[k + '1'] = args.bf
            while True:
                aug_atom = sphere_selector(f,corners,copy.deepcopy(cell.atom),label='1')
                f += 0.011 # prime number to avoid a multiple of unit cell

                aug_shell = 0
                for at in aug_atom:
                    if '1' in at[0]:
                        aug_shell += 1
                if 'd' in args.sys or 'si' in args.sys:
                    if aug_shell == 5: # aug_shell includes ghost atom
                        break
                else:
                    if aug_shell == 7:
                        break
                if aug_shell > 7:
                    raise RuntimeError('Did not identify a shell for basis augmentation')
            atom = aug_atom

    return A, atom, pseudo, basis

def init_cell(args):
    '''
    Gather all cell initiation code in one spot so I can update cell without
    discarding the mf object.
    '''
    cell = gto.Cell()
    # max_memory is set by the slurm submission script
    # these values are rounded down as I believe taking all the memory causes
    # flatiron nodes to fail (linux oom killer)
    if cell.max_memory == 768000:
        cell.max_memory = 750000
    elif cell.max_memory == 512000:
        cell.max_memory = 500000
    elif cell.max_memory == 1024000:
        cell.max_memory = 1000000
    if args.etd:
        cell.exp_to_discard = args.etd

    A, atom, pseudo, basis = init_geom(args)

    # internally, units are Bohr (a.u.) but input is angstrom
    cell.build(parse_arg = False,
               a = A, # rows are the basis vectors
               atom = atom,
               dimension = args.d,
               charge = 0,
               low_dim_ft_type = args.l,
               pseudo = pseudo,
               basis = basis,
               precision = args.pr,
               spin = None,
               verbose = args.verbose)
    args.nelectron = cell.nelectron

    return args,cell

def init_scf(args,cell,scf_method='HF'):
    '''
    Initialize the SCF objects
    '''

    with_df = getattr(df,args.df)
    with_mf = getattr(scf,scf_method)
    mf = with_mf(cell,exxdiv=args.e)
    mf.with_df = with_df(cell)
    if args.blockdim != 0:
        mf.with_df.blockdim = args.blockdim

    if 'RSDF' in args.df and args.prrs is not None:
        mf.with_df.precision_R = cell.precision * args.prrs
        mf.with_df.precision_G = cell.precision * args.prrs

    if args.chk:
        # .df and .scf files are streamed/saved to ceph
        chk_path = args.chk_path

        #scf
        if scf_method == 'HF': # ignore ROHF (for delta-SCF)
            if os.path.isfile(chk_path + '.scf'):
                print('*** scf chkfile found')
                # loads e_tot, mo_energy, mo_occ, mo_coeff
                mf.__dict__.update(scf.chkfile.load(chk_path + '.scf', 'scf'))
                mf.converged = True
            else:
                print('*** No scf chkfile found, scf will begin from scratch')
                mf.chkfile = chk_path + '.scf'

        #df
        if os.path.isfile(chk_path + '.df'):
            print('*** df tensor found')
            if args.dflocal:
                mf.with_df._cderi = os.path.join(lib.param.TMPDIR,args.chk + '.df')
                if os.path.isfile(mf.with_df._cderi):
                    print('df tensor already exists in local scratch')
                else:
                    print('copy df tensor to local scratch')
                    t0 = time.time()
                    shutil.copy(chk_path + '.df',mf.with_df._cderi)
                    print(f'    copy time: {(time.time()-t0)/60:.1f}m')
            else:
                mf.with_df._cderi = chk_path + '.df'
            print('   ' + mf.with_df._cderi)
        else:
            print('*** No df tensor found, df will build from scratch')
            mf.with_df._cderi_to_save = chk_path + '.df'
        print(mf.with_df)

    if args.remlindep:
        from pyscf.scf import addons
        mf = mf.apply(addons.remove_linear_dep_)

    return mf

def init_calc(args):
    '''
    This function is probably unncessary, should refactor it away eventually
    '''
    args,cell = init_cell(args)
    mf = init_scf(args,cell)
    return args,cell,mf

def make_kpts_path(kpts,nband):
    '''
    if args.n and args.a == 0:
        print('Band structure')
        kpts_band = make_kpts_path(args.n)
        mo_energy_bands, mo_coeff_bands = mf.get_bands(kpts_band)
        surfout['E']['HF'].update({'mo_energy_bands':mo_energy_bands})
        surfout['C']['HF'].update({'mo_coeff_bands':mo_coeff_bands})
        args.kpts_band = kpts_band

        # i.e. pure graphene, not absorbate or with water, grab bands
        # K-G
        kpts_band = numpy.zeros((3*nband-2,3))
        kpts_band[0:nband,0] = 2*numpy.linspace(1/3,0,nband)
        kpts_band[0:nband,1] = numpy.linspace(1/3,0,nband)
        kpts_band[0:nband,2] = 0
        # G-M
        kpts_band[nband-1:2*nband-1,0] = numpy.linspace(0,1/2,nband)
        kpts_band[nband-1:2*nband-1,1] = 0
        kpts_band[nband-1:2*nband-1,2] = 0
        # M-K
        kpts_band[2*nband-2:,0] = 1/2 + numpy.linspace(0,1/3,nband)/2
        kpts_band[2*nband-2:,1] = numpy.linspace(0,1/3,nband)
        kpts_band[2*nband-2:,2] = 0
    '''
    kpts_band = numpy.zeros((nband,3))

    return kpts_band

def save_surfout(surfout):
    '''
    save as a mat file
    '''
    args = surfout['args']
    output_path = os.path.normpath('../output_surf/' + args.j)
    # None -> 'None' (None is not defined in matlab / sio)
    for a in [a for a in dir(args) if not a.startswith('_')]:
        if getattr(args,a) is None:
            setattr(args,a,'None')
    sio.savemat(output_path + '.mat',surfout,oned_as='column')
    return

def boys_cost(cell,mo_coeff):
    # calculate the boys cost function, <r^2> - <r>^2
    # charge_center = numpy.einsum('z,zx->x', cell.atom_charges(), cell.atom_coords())
    # defect is at the center

    origin = (0., 0., 0.)

    charges = cell.atom_charges()
    coords  = cell.atom_coords()
    origin = numpy.einsum('i,ix->x', charges, coords) / charges.sum()

    # use the center of the h2o if adsorption
    # F-center - use origin
    if args.sys in ('g','hbn','lih'):
        origin = cell.atom[-3][1]

    with cell.with_common_orig(origin):
        dip = numpy.asarray([reduce(lib.dot, (mo_coeff.conj().T, x, mo_coeff))
                         for x in cell.pbc_intor('int1e_r')])
        r2 = reduce(lib.dot, (mo_coeff.conj().T, cell.pbc_intor('int1e_r2'), mo_coeff))

    dip2 = numpy.einsum('xij,xij->ij', dip, dip)
    dip = numpy.einsum('xij->ij',dip)
    # <r> and <r^2> are in the MO basis - get the diagonals, which corresponds 
    # to the <r>_ii matrix element
    return numpy.diag(dip), numpy.diag(dip2), numpy.diag(r2)

def frozen_mf(args,cell,mf,frozen,ro_coeff=None,ro_energy=None):
    '''
    make a frozen mf object

    regenerate mo_energy for exxdiv
    '''
    # create a whole new mf object instead of copy
    mfcis = init_scf(args,cell)

    if ro_coeff is None:
        # for freeze by energy
        ro_coeff = mf.mo_coeff
        ro_energy = mf.mo_energy
    
    if isinstance(mf, scf.hf.RHF):
        nmo = mfcis.mo_coeff.shape[1]
        active = ~numpy.isin(numpy.arange(nmo),frozen)
        mfcis.mo_occ = mfcis.mo_occ[active]
        mfcis.mo_coeff = ro_coeff[:,active]
        mfcis.mo_energy = ro_energy[active]
    elif isinstance(mf, scf.uhf.UHF):
        for idx in range(2):
            nmo = mfcis.mo_coeff[idx].shape[1]
            active = ~numpy.isin(numpy.arange(nmo),frozen[idx])
            mfcis.mo_occ[idx] = mfcis.mo_occ[idx][active]
            mfcis.mo_coeff[idx] = ro_coeff[idx][:,active]
            mfcis.mo_energy[idx] = ro_energy[idx][active]
            
    return mfcis

def get_nto(mo_coeff,tdm):
    # get NTOs from a transition density matrix of shape ia
    # borrows code from pyscf/tdscf/rhf:get_nto()
    # tdm is expected to be in shape "ia"

    # renormalize tdm (could be eom-ccsd r1)
    tdm *= 1. / numpy.linalg.norm(tdm)
    nocc,nvir = tdm.shape

    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]

    nto_o, w, nto_vT = numpy.linalg.svd(tdm)
    nto_v = nto_vT.conj().T
    weights = w**2

    idx = numpy.argmax(abs(nto_o.real), axis=0)
    nto_o[:,nto_o[idx,numpy.arange(nocc)].real<0] *= -1
    idx = numpy.argmax(abs(nto_v.real), axis=0)
    nto_v[:,nto_v[idx,numpy.arange(nvir)].real<0] *= -1

    occupied_nto = numpy.dot(orbo, nto_o)
    virtual_nto = numpy.dot(orbv, nto_v)
    nto_coeff = numpy.hstack((occupied_nto, virtual_nto))

    return weights, nto_coeff

def embed(surfout,args,cell,mf):
    '''
    Selects an active space, based upon parameters given in args
    '''

    # HZ correlation consistent basis requires first 4 Mg core orbitals to be frozen
    mgo_cc = False
    NMg = 0
    if 'mgo' in args.sys and 'gthccpv' in args.b.replace('-','').lower() and args.frcore:
        mgo_cc = True
        NMg = sum([1 for at in cell.atom if 'Mg' in at[0]]) * 4
        
    ro_coeff = None
    ro_energy = None
    frozen = None
    surfout['C']['emb'] = dict()

    if 'FE' in args.emb:
        print('******** Energy frozen orbitals ******** ')
        if 'FEX' in args.emb:
            print('        ordered by overlap with defect')
            if 'hbn' in args.sys:
                idx = cell.search_ao_label('C')
            else:
                with lib.temporary_env(args, verbose=0, bf='gthszv'):
                    _,tmp_cell = init_cell(args)
                idx = tmp_cell.search_ao_label('1')

            S = mf.get_ovlp()
            X = sqrtm(S) @ mf.mo_coeff
            X = numpy.absolute(X)**2
            defect_ovlp = X[idx,:].sum(axis=0)

            nocc = cell.nelectron // 2
            # ascending
            idx2 = numpy.argsort(defect_ovlp[NMg:nocc]) + NMg
            # descending
            idx3 = numpy.flip(numpy.argsort(defect_ovlp[nocc:])) + nocc
            idx4 = numpy.concatenate((idx2,idx3))
            ro_coeff = mf.mo_coeff[:,numpy.concatenate((numpy.r_[0:NMg],idx2,idx3))]

            surfout['C']['emb']['ovlp'] = defect_ovlp[idx4]
            surfout['C']['emb']['idx'] = idx4 + 1

        print(args.ft)
    elif 'NTO' in args.emb:
        print('******** NTO frozen orbitals ********')
        print(args.ft)

        if args.frcore_nto and mgo_cc:
            if isinstance(mf, scf.hf.RHF):
                frozen = list(range(NMg))
            elif isinstance(mf, scf.uhf.UHF):
                frozen = numpy.array((list(range(NMg)),list(range(NMg))))
            mfcis = frozen_mf(args,cell,mf,frozen)
        else:
            mfcis = mf

        # check for existing NTOs
        if args.chk:
            # try to load a previously saved fock matrix, file format numpy .npy
            NTO_path = args.chk_path + '_NTO'
            wts_path = args.chk_path + '_wts'
            if args.frcore_nto:
                NTO_path += '_frcore'
                wts_path += '_frcore'
            NTO_path += '.npy'
            wts_path += '.npy'
            if os.path.isfile(NTO_path):
                print('Found a previously saved NTO.npy')
                ro_coeff = numpy.load(NTO_path)
                weights = numpy.load(wts_path)
            else:
                print('No saved NTO coeffs, making one now')
                mytda = tdscf.TDA(mfcis)
                mytda.kernel(nstates=args.r)
                weights, ro_coeff = mytda.get_nto()
                numpy.save(NTO_path,ro_coeff)
                numpy.save(wts_path,weights)

        surfout['C']['emb']['nto_weights'] = weights
        surfout['C']['emb']['nto_coeff'] = ro_coeff.shape
        if not args.slim: surfout['C']['emb']['nto_coeff'] = ro_coeff.copy()

        # weights from the SVD are sorted in descending order (same with the
        # singular vectors). flip the occupied block so it looks like a
        # quantum chemistry HOMO-LUMO ordering
        if isinstance(mf, scf.hf.RHF):
            nocc = len(weights)
            ro_coeff[:,:nocc] = numpy.flip(ro_coeff[:,:nocc],axis=1)
        elif isinstance(mf, scf.uhf.UHF):
            # ro_coeff is returned as a tuple
            ro_coeff = list(ro_coeff)
            ro_coeff[0][:,:len(weights[0])] = numpy.flip(ro_coeff[0][:,:len(weights[0])],axis=1)
            ro_coeff[1][:,:len(weights[1])] = numpy.flip(ro_coeff[1][:,:len(weights[1])],axis=1)
    elif 'NO' in args.emb:
        print('******** Frozen NOs (from MP2) ********')
        print(args.ft)



        raise NotImplementedError
    elif 'DSCF' in args.emb:
        print('******** delta-SCF frozen orbitals ********')
        print(args.ft)
        from pyscf.scf.addons import mom_occ
        # although true DSCF should have a ROHF for both GS and ES, we only 
        # want the modified ES orbitals. Closed-shell ROHF and RHF
        # should be the same though "emf - excited mf"
        mo0 = mf.mo_coeff
        occ = mf.mo_occ
        setocc = numpy.zeros((2, occ.size))
        setocc[:, occ==2] = 1

        nocc = numpy.count_nonzero(occ)
        setocc[0][nocc-1] = 0 # HOMO
        setocc[0][nocc] = 1 # LUMO, due to python indexing
        ro_occ = setocc[0][:] + setocc[1][:]

        d = init_scf(args,cell,'ROHF')
        dm_ro = d.make_rdm1(mo0, ro_occ)
        d = mom_occ(d, mo0, setocc) # only works for single k-point/gamma point?
        d.scf(dm_ro)
        ro_coeff = d.mo_coeff

        surfout['C']['emb']['dscf_coeff'] = d.mo_coeff.shape
        if not args.slim: surfout['C']['emb']['dscf_coeff'] = d.mo_coeff
        surfout['C']['emb']['dscf_occ'] = d.mo_occ
        surfout['conv']['emb']['dscf_occ'] = d.converged
        surfout['E']['emb']['dscf_energy'] = d.mo_energy
        surfout['E']['emb']['dscf_etot'] = d.e_tot
    elif 'RE' not in args.emb:
        raise NotImplementedError('You did''t specify a valid embedding method')

    with lib.temporary_env(mf, verbose=0):
        mo_occ = mf.get_occ()

    if 'RE' not in args.emb:
        # set frozen indices for any method that isn't regional embedding
        # regional embedding, if enabled, will override this frozen index
        if not all([e >= 1 for e in args.ft]):
            print('You didn''t give a valid range for ft (you chose freezing orbitals)')
            print('Assuming no frozen orbitals')
            frozen = None
        else:
            ft = args.ft.copy()
            if NMg and 'FE' in args.emb:
                # only freeze core orbitals if in the canonical basis
                nocc = cell.nelectron // 2
                ft[0] = min(nocc - NMg,ft[0])

            if isinstance(mf, scf.hf.RHF):
                nocc = numpy.count_nonzero(mo_occ)
                nmo = mo_occ.size
                frozen = list(range(nocc - ft[0])) + list(range(nocc+ft[1], nmo))
            elif isinstance(mf, scf.uhf.UHF):
                nocc = numpy.count_nonzero(mo_occ,axis=1)
                nmo = mo_occ.shape[1]
                frozen_a = list(range(nocc[0] - ft[0])) + list(range(nocc[0]+ft[1], nmo[0]))
                frozen_b = list(range(nocc[1] - ft[0])) + list(range(nocc[1]+ft[1], nmo[1]))
                frozen = numpy.array((frozen_a,frozen_b))
            if len(frozen) == 0:
                frozen = None
    else: # 'RE' in args.emb
        if ro_coeff is None:
            ro_coeff = mf.mo_coeff
        print('******** Regional embedding ********')
        print(args.f,args.ft)
        ro_coeff, frozen, emb_dat = emb.make_rdmet(cell,ro_coeff,mo_occ,NMg,args.ft,args.resymb,args.minao)
        surfout['C']['emb'].update(emb_dat)
    
    if not args.slim: surfout['C']['emb']['ro_coeff'] = ro_coeff
    if frozen is not None:
        surfout['C']['emb']['frozen'] = frozen
    if NMg:
        surfout['C']['emb']['nmgfr'] = NMg

    if args.can and 'FE' not in args.emb:
        # canonicalize active space for any embedding method that is not frozen energy
        # (which is frozen in the canonical basis)
        print('******** Canonicalize the active fragment orbitals ********')
        mf_or_fock = mf
        if args.chk:
            # try to load a previously saved fock matrix, file format numpy .npy
            if os.path.isfile(args.chk_path + '.npy'):
                print('Found a previously saved fock.npy')
                mf_or_fock = numpy.load(args.chk_path + '.npy')
            else:
                print('There was a .scf but no fock.npy, making one now')
                mf_or_fock = mf.get_fock()
                numpy.save(args.chk_path,mf_or_fock) # np auto adds '.npy'
        ro_coeff, ro_energy = emb.canonicalize(mf_or_fock,ro_coeff,frozen)
        if not args.slim: surfout['C']['emb']['can_coeff'] = ro_coeff
        surfout['E']['emb'] = {'can_energy':ro_energy}
    
    return surfout,args,ro_coeff,frozen

def main(args,cell,mf):
    # output dictionary
    surfout = {'E':dict(),'C':dict(),'conv':dict()}

    if args.r is None:
        if 'CIS' in args.T:
            # CIS sensitive to nroots, EOM does not appear to be sensitive
            # Diamond: doubly degenerate GR1 state (two roots total?)
            if 'd' in args.sys:
                args.r = 2
            # MgO: from HF, LUMO:1-fold, LUMO+1:3-fold (four roots total?)
            elif 'mgo' in args.sys:
                args.r = 3
            else:
                # pyscf default
                args.r = 3
        else:
            args.r = 1

    print('')
    print('*******************************************************')
    print('surfin'' USA - graphene (+water) surface pyscf - pysurf')
    print('and also defects lalalala ~~~')
    print('*******************************************************')
    print('{0} threads'.format(lib.num_threads()))
    print('{0} memory assigned'.format(cell.max_memory))
    print('{0} scratch directory'.format(lib.param.TMPDIR))

    t0 = time.time()
    if args.forceuhf:
        mf = scf.addons.convert_to_uhf(mf)

    # preload fock matrix for later canonicalization of regional embedding
    if not mf.converged:
        mf.kernel()
    else:
        print('\n===== Using converged SCF result from chk file =====')
        mf._finalize()

    surfout['E']['HF'] = {'e_tot':mf.e_tot,'mo_energy':mf.mo_energy}
    C_dict = {'mo_coeff':[],'mo_occ':mf.mo_occ,'S':[]}
    if not args.slim:
        C_dict['mo_coeff'] = mf.mo_coeff
        C_dict['S'] = mf.get_ovlp()
    surfout['C']['HF'] = C_dict
    surfout['conv']['HF'] = {'conv':mf.converged,'time':time.time() - t0}

    if not mf.converged:
        print('SCF did not converge! post-HF will not be applied; .scf and .df contain unconverged results!')
        return surfout
    
    if args.stab:
        from pyscf.ao2mo.incore import iden_coeffs
        print('Stability analysis (internal)')
        
        mo1 = mf.stability()[0]
        for cycle in range(10):
            print(f'Stability analysis, cycle {cycle}')
            if iden_coeffs(mo1,mf.mo_coeff):
                print('RHF is internally stable')
                surfout['E']['HF']['e_tot_stab'] = mf.e_tot
                surfout['E']['HF']['mo_energy_stab'] = mf.mo_energy
                surfout['conv']['HF']['conv_stab'] = mf.converged
                break
            else:
                print('Unstable solution found')
                dm1 = mf.make_rdm1(mo1, mf.mo_occ)
                # new solution should automatically be saved as run() calls the
                # kernel of the existing mf object, which will dump_chk at conv
                mf = mf.run(dm1)
                mo1 = mf.stability()[0]
            
    # force building mo eris from df
    mf._eri = None
    # build any j/k matrices from here on out with df to avoid repopulating _eri
    # edge case: _eri is None, there is enough memory to rebuild _eri for get_veff
    # from ccsd common_init, then out of memory assigning vvvv to feri via Lvv.T @ Lvv
    # NOTE: this may not be  a problem anymore as dfccsd avoids building vvvv
    mf.direct_scf = True

    # generate frozen/active space
    frozen = None
    ro_coeff = None
    if args.emb:
        surfout,args,ro_coeff,frozen = embed(surfout,args,cell,mf)

    if 'MP' in args.T:
        if isinstance(mf, scf.hf.RHF):
            #mymp = mp.RMP2(mf,frozen=frozen,mo_coeff=ro_coeff)
            mymp = mmp.dfmp2.DFMP2(mf,frozen=frozen,mo_coeff=ro_coeff)
        elif isinstance(mf, scf.uhf.UHF):
            mymp = mp.UMP2(mf,frozen=frozen,mo_coeff=ro_coeff)
        mymp.kernel(with_t2=(args.slim==0))

        surfout['E']['MP2'] = {'e_corr':mymp.e_corr}
        if not args.slim: surfout['C']['MP2'] = {'y':mymp.make_rdm1()}
        surfout['conv']['MP2'] = {'conv':1,'time':time.time() - t0}
    
    if 'CCSD' in args.T:
        if isinstance(mf, scf.hf.RHF):
            if args.dfccsd:
                mycc = ccg.DFCCSD(mf,frozen=frozen,mo_coeff=ro_coeff)
            else:
                # memory optimized incore ccsd (vs rccsd)
                mycc = ccg.CCSD(mf,frozen=frozen,mo_coeff=ro_coeff)
        elif isinstance(mf, scf.uhf.UHF):
            mycc = ccg.UCCSD(mf,frozen=frozen,mo_coeff=ro_coeff)

        if args.mc:
            mycc.max_cycle = args.mc
        
        mycc.keep_exxdiv = args.ke
        eris = mycc.ao2mo()
        if args.log:
            import cProfile
            mycc.max_cycle = 3
            cProfile.run('mycc.kernel(eris=eris)')
            print('Finished profiling CCSD, exiting now...')
            sys.exit()
        else:
            mycc.kernel(eris=eris)
        surfout['E']['CCSD'] = {'e_corr':mycc.e_corr}
        if not args.slim:
            surfout['C']['CCSD'] = {'y':mycc.make_rdm1()}
        else:
            surfout['C']['CCSD'] = {}
        surfout['conv']['CCSD'] = {'conv':mycc.converged,'time':time.time() - t0}
        
        if 'T' in args.T:
            et = mycc.ccsd_t(eris=eris)
            surfout['E']['CCSD']['et'] = et
            surfout['conv']['CCSD']['timet'] = time.time() - t0
            
        if 'EE' in args.T:
            if isinstance(mf, scf.hf.RHF):
                from pyscf.cc import eom_rccsd
                myeom = eom_rccsd.EOMEESinglet(mycc)
            elif isinstance(mf, scf.uhf.UHF):
                from pyscf.cc import eom_uccsd
                myeom = eom_uccsd.EOMEESpinKeep(mycc)
            myeom.kernel(nroots=args.r,eris=eris)

            surfout['E']['CCSD']['ee'] = myeom.e
            if not args.slim: surfout['C']['CCSD'] = {'eomv':myeom.v}
            surfout['conv']['CCSD']['timeee'] = time.time() - t0
            surfout['conv']['CCSD']['convee'] = myeom.converged

            if args.nto:
                print('R1 NTO')
                if args.r > 1:
                    v = myeom.v[args.nr]
                    e = myeom.e[args.nr]
                    c = myeom.converged[args.nr]
                else:
                    v = myeom.v
                    e = myeom.e
                    c = myeom.converged
                r1 = myeom.vector_to_amplitudes(v)[0]
                if r1.ndim == 1:
                    # nocc == 1
                    r1 = numpy.expand_dims(r1,0)
                r1_weights, r1_nto = get_nto(eris.mo_coeff,r1)

                if args.eomnto:
                    nocc,nvir = r1.shape
                    print('EOM NTO (Alan)')
                    print('solve lambda')
                    mycc.solve_lambda(eris=eris)
                    from eom_rdm import eom_rccsd as eom_rccsd_alan
                    mydummyeom = eom_rccsd_alan.EOMEESinglet(mycc)
                    mydummyeom.v = [v]
                    mydummyeom.e = [e]
                    mydummyeom.converged = [c]
                    print('gen tdm')
                    eom_tdm = mydummyeom.gen_tdms(v,prange=range(nocc),qrange=range(nocc,nocc+nvir),eris=eris)
                    # dimensions of (nstate,prange,qrange)
                    eom_tdm = eom_tdm[0]
                    eom_weights, eom_nto = get_nto(eris.mo_coeff,eom_tdm)
                else:
                    eom_weights = numpy.zeros(r1.shape[0])
                    eom_nto = numpy.zeros(r1.shape)

                surfout['C']['CCSD']['r1_nto_weights'] = r1_weights
                surfout['C']['CCSD']['eom_nto_weights'] = eom_weights
                surfout['C']['CCSD']['r1_nto_coeff'] = r1_nto.shape
                surfout['C']['CCSD']['eom_nto_coeff'] = eom_nto.shape
                if not args.slim:
                    surfout['C']['CCSD']['r1_nto_coeff'] = r1_nto
                    surfout['C']['CCSD']['eom_nto_coeff'] = eom_nto

        eris = None

    if 'CIS' in args.T:
        # to grab nocc, nmo
        mymp = mmp.dfmp2.DFMP2(mf,frozen=frozen,mo_coeff=ro_coeff)
        nocc = mymp.nocc
        nvir = mymp.nmo - nocc
        mem_eri = 2*nocc**2*nvir**2*8/1024**2
        mem_eri = nocc*nvir*mymp.nmo**2*8/1024**2
        mem_now = lib.current_memory()[0]
        max_memory = max(0, mymp.max_memory - mem_now)*.8
        mymp = None

        if (args.ericis or mem_eri < max_memory) and not args.c:
            print('******** ERI frozen CIS ********')
            if isinstance(mf, scf.hf.RHF):
                mytda = ccg.CIS(mf,frozen=frozen,mo_coeff=ro_coeff)
            elif isinstance(mf, scf.uhf.UHF):
                raise NotImplementedError('ERI UCIS')
            mytda.kernel(nroots=args.r)
        else:
            # make a new mf because its attributes will be modified if frozen
            #args.dflocal = 1
            print('******** Hacked frozen response CIS ********')
            mytda = ccg.TDA(init_scf(args,cell),frozen=frozen,mo_coeff=ro_coeff)
            mytda.kernel(nstates=args.r)

        surfout['E']['CIS'] = mytda.e
        surfout['C']['CIS'] = {'xy':[]}
        surfout['conv']['CIS'] = mytda.converged
        surfout['C']['CIS']['xy'] = mytda.xy[0][0].shape # nocc,nvir - xy is in (x,0) tuple
        if not args.slim:
            surfout['C']['CIS']['xy'] = mytda.xy
        if args.nto:
            weights, nto_coeff = mytda.get_nto()
            surfout['C']['CIS']['nto_weights'] = weights
            surfout['C']['CIS']['nto_coeff'] = nto_coeff.shape
            if not args.slim:
                surfout['C']['CIS']['nto_coeff'] = nto_coeff
        
    if 'SCRE' in args.T:
        myre = emb.SCRE(mf,aolabels,args.ft,args.corr,args.minao)
        myre.kernel()
        surfout['E']['SCRE'] = {'e_corr':myre.e_corr}
        if not args.slim: surfout['C']['SCRE'] = {'rdm_corr':myre.rdm_corr,'rdm_mf':myre.rdm_mf,'U':myre.U,'Fo':myre.Fo}
        surfout['conv']['SCRE'] = myre.conv
    
    # cube output modifies args to trick code inside, make a copy of args to 
    # preserve the original intent of args
    surfout['args'] = copy.copy(args)
    if args.c:
        import cubegen
        def make_cubes():
            #cput0 = (time.clock(), time.time())
            #logger.timer(cell,'molden dump', *cput0)            
            # cube file (args.c specifies orbitals beyond occupied level)
            if args.j is not None:
                cube_path = os.path.normpath('../output_cube/' + args.j)
            else:
                cube_path = os.path.normpath('../output_cube/' + 'test')
                
            trailing_zeros = int(numpy.log10(len(mf.mo_occ))) + 1
            
            if numpy.isscalar(args.c):
                if args.nto:
                    cube_range = list(range(min(args.c,nocc))) + list(range(nocc,min(nocc+args.c,2*nocc)))
                else:
                    cube_range = range(max(0,nocc-args.c),min(len(mf.mo_occ),nocc+args.c))
            else:
                cube_range = args.c
            
            for ind in cube_range:
                cube_path_mo = cube_path + '_mo' + '{0}'.format(ind).zfill(trailing_zeros)
                if ind < nocc:
                    cube_path_mo += '_occ'
                else:
                    cube_path_mo += '_vir'
                if args.emb:
                    if ind in frozen:
                        cube_path_mo += '_fr'
                    else:
                        cube_path_mo += '_as'
                        if 'RE' in args.emb:
                            cube_path_mo += str(emb_dat['wlist'][ind])
                if args.nto:
                    cube_path_mo += f'_{weights[ind % nocc]:.3e}' + '_nto'
                
                cube_path_mo += '.cube'
                cubegen.orbital(args,cell,cube_path_mo,cube_coeff[:,ind])
                print(cube_path_mo)

        print ('******** Cubegen ********')
        #cell.dump_input()

        # emb: print out the embedding orbitals
        # superceded by ntos, which are analyzed based on results of a possible embedding
        # if nothing, then print canonical orbitals
        nocc = numpy.count_nonzero(mf.mo_occ)
        if args.nto:
            if 'CIS' in args.T:
                nocc = len(weights)
                cube_coeff = nto_coeff
                make_cubes()
            elif 'EE' in args.T and args.nto:
                args.nto = True
                args.emb = False
                nocc = mycc.nocc

                j = args.j
                cube_coeff = r1_nto
                weights = r1_weights
                args.j = 'r1' + j
                make_cubes()

                if args.eomnto:
                    cube_coeff = eom_nto
                    weights = eom_weights
                    args.j = 'eom' + j
                    make_cubes()
        elif args.emb:
            cube_coeff = ro_coeff
            if 'NTO' in args.emb:
                nocc = len(weights)
                # flip from HOMO-LUMO ordering back to pyscf default 
                # (i weights increasing ->),(a weights increasing ->)
                # (<- i weights increasing),(a weights increasing ->)
                cube_coeff[:,:nocc] = numpy.flip(cube_coeff[:,:nocc],axis=1)
                args.nto = True
                args.emb = False
            make_cubes()
        else:
            cube_coeff = mf.mo_coeff
            make_cubes()

    return surfout

if __name__ == '__main__':
    print(sys.version)

    args = getArgs(sys.argv[1:])
    # name of script surf.py -> surf
    j1 = os.path.splitext(sys.argv[0])[0]
    # meshgrid of f and ft
    fs = []
    for ft in args.ft:
        for f in args.f:
            fs.append([f,ft])
    args.f = args.f[0]
    args.ft = args.ft[0]
    
    args,cell,mf = init_calc(args)

    for f,ft in fs:
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'Running surf/defects for {f} and {ft} in {fs}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # don't rebuild cell for a single f
        if len(fs) > 1:
            # rebuild cell for each f. The cell object is referenced by all
            # subsequent mf, with_df, post-mf, so modifying the original cell 
            # is enough
            argv = sys.argv[1:]

            try:
                fidx = argv.index('-f') + 1
                argv[fidx] = str(f)
            except ValueError:
                pass
            args.f = f

            try:
                fidx = argv.index('--ft') + 1
                argv[fidx] = str(f)
            except ValueError:
                pass
            args.ft = ft

            try:
                # try to remove any -j arguments given
                jidx = sys.argv[1:].index('-j')
                del argv[jidx:jidx+1+1]
            except ValueError:
                pass
            args.j = j1 + ''.join(argv).replace('-','_')
            print(args.j)

            A, atom, pseudo, basis = init_geom(args)
            cell.build(parse_arg = False,
                        a = A,
                        atom = atom,
                        pseudo = pseudo,
                        basis = basis)
            # borrow code from mole dump_input(), print out new labeled atoms
            for ia,atom in enumerate(cell._atom):
                coorda = tuple([x * BOHR for x in atom[1]])
                coordb = tuple([x for x in atom[1]])
                cell.stdout.write('[INPUT]%3d %-4s %16.12f %16.12f %16.12f AA  '
                                '%16.12f %16.12f %16.12f Bohr\n'
                                % ((ia+1, _symbol(atom[0])) + coorda + coordb))
        
        surfout = main(args,cell,mf)
        # save results
        if args.j is not None:
            save_surfout(surfout)
    
    scratch_df_path = os.path.join(lib.param.TMPDIR,args.chk + '.df')
    if os.path.isfile(scratch_df_path):
        print('found a .df file in scratch at: ' + scratch_df_path)
        os.remove(scratch_df_path)
    else:
        print('no df in scratch to clean up')

'''
def sigterm_handler(sig,frame):
    sys.exit('SIGTERM recieved. Did you scancel?')

can't use a sigterm handler with multiprocessing, because multiprocessing
manages workers with sigterm and other signals
signal.signal(signal.SIGTERM, sigterm_handler)

cube_list.append((cell,cube_path_mo,mf.mo_coeff[:,ind]))
from multiprocessing import Pool
pool = Pool(lib.num_threads()-1)
pool.starmap(cubegen.orbital_pool,cube_list)
pool.close()
pool.join()
'''