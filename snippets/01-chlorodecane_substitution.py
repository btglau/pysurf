"""
Created on Fri Oct 25 10:10:57 2019

@author: blau

The substitution reaction:

Chlorodecane + -OH -> Decanol + Cl-

Change the basis for the calculation at the end of the 'init_mol' function
Change the basis that defines the fragment at the start of __main__
Illustrates the trickiness of thresholds for the overlap eigenvalues, as well as
sensitivity to the choice of the fragment AO basis
"""

import numpy
from pyscf import gto
from pyscf import mp
from pyscf import scf
import emb

def init_mol(mol_str):
    if mol_str == 'cd':
        # chlorodecane
        atom = [
        ['1H',       (-6.4206104583,      0.5725387852,     -0.2230454498)],         
        ['1H',       (-5.6407076534,      1.5647519991,      1.0202831615)],        
        ['1C',       (-5.5345312520,      1.1737968759,      0.0033182150)],       
        ['1H',       (-5.5125473268,      2.0210849090,     -0.6896428878)],  
        ['1H',       (-4.3048029843,     -0.4730430339,      0.6250321783)],     
        ['1C',       (-4.2769717604,      0.3279518646,     -0.1228457498)],    
        ['1H',       (-4.2627915809,     -0.1490947308,     -1.1095922178)],   
        ['2H',       (-3.0500372625,      1.6892985548,      1.0231181948)],  
        ['2C',       (-3.0158260076,      1.1721152028,      0.0566694956)], 
        ['2H',       (-2.9841418864,      1.9450779224,     -0.7213684710)],
        ['2C',       (-1.7499955650,      0.3155211943,     -0.0191723371)],
        ['2H',       (-1.7469213695,     -0.4067793283,      0.8065081434)],
        ['2H',       (-1.7464078670,     -0.2593229231,     -0.9527917262)],
        ['2H',       (-0.5055451063,      1.7757620998,      0.9739128363)],
        ['2H',       (-0.4977335103,      1.8954337133,     -0.7864792169)],
        ['2C',       (-0.4938428569,      1.1852907079,      0.0500078318)],
        ['3C',       ( 0.7832651261,      0.3444221161,     -0.0030423929)],
        ['3H',       ( 0.7904812099,     -0.2604925932,     -0.9174661256)],
        ['3H',       ( 0.8022152053,     -0.3500774228,      0.8458852170)],
        ['3H',       ( 2.0024202592,      1.9284068511,     -0.8185148067)],
        ['3H',       ( 2.0114386136,      1.8470787153,      0.9440582443)],
        ['3C',       ( 2.0254132281,      1.2368754395,      0.0330277075)],
        ['3C',       ( 3.3158313532,      0.4171415805,     -0.0123668087)],
        ['3H',       ( 3.3338224911,     -0.1927778622,     -0.9234608569)],
        ['3H',       ( 3.3444487405,     -0.2723773006,      0.8403271474)],
        ['3H',       ( 4.5114393866,      2.0177422465,     -0.8308798445)],
        ['3H',       ( 4.5219231228,      1.9384403672,      0.9323440347)],
        ['3C',       ( 4.5456244807,      1.3275366309,      0.0215025888)],
        ['4C',       ( 5.8347231605,      0.5184884176,     -0.0228256625)],
        ['4H',       ( 5.8982652495,     -0.0747182233,     -0.9397918508)],
        ['4H',       ( 5.9082673152,     -0.1556894990,      0.8357763028)],
        ['5Cl',      ( 7.2462700056,      1.6006565771,      0.0186181670)]
        ]
        '''
        atom = [
        ['5Cl',(1.1643273123,-9.6662407950,-0.0103951879)],
        ['4H',(-0.5453851379,-8.3065902533,-0.8881536299)],
        ['4H',(-0.5329580550,-8.3050312849,0.8889111687)],
        ['4C',(0.0962278868,-8.2439260897,-0.0041344079)],
        ['3C',(0.9186323676,-6.9622477608,-0.0111645937)],
        ['3H',(1.5621452270,-6.9384087569,-0.8994076352)],
        ['3H',(1.5772801705,-6.9383385115,0.8658708353)],
        ['3H',(-0.6452153904,-5.7491598775,-0.8782202918)],
        ['3H',(-0.6252319626,-5.7467621024,0.8869174981)],
        ['3C',(0.0155603623,-5.7273268422,-0.0030915990)],
        ['3C',(0.8386971306,-4.4382762011,-0.0145560140)],
        ['3H',(1.5043306419,-4.4200296811,0.8569706702)],
        ['3H',(1.4751169984,-4.4175341015,-0.9076867893)],
        ['3H',(-0.7286767282,-3.2188707263,-0.8664804547)],
        ['3H',(-0.6927739644,-3.2184227121,0.8983898368)],
        ['3C',(-0.0600243183,-3.2006079587,0.0026492898)],
        ['2C',(0.7687146231,-1.9151015281,-0.0148798347)],
        ['2H',(1.4434487592,-1.9015233889,0.8496997816)],
        ['2H',(1.3959336621,-1.8956213185,-0.9145398880)],
        ['2H',(-0.7477364144,-0.6902135563,0.9158700614)],
        ['2H',(-0.8012485255,-0.6867782896,-0.8485349711)],
        ['2C',(-0.1238563567,-0.6733647665,0.0139005332)],
        ['2C',(0.7102553511,0.6086974009,-0.0090526417)],
        ['2H',(1.3923963949,0.6187176899,0.8497946197)],
        ['2H',(1.3301566030,0.6269836488,-0.9138193349)],
        ['1H',(-0.7938844043,1.8426761531,0.9347329586)],
        ['1H',(-0.8600750765,1.8492078952,-0.8290511494)],
        ['1C',(-0.1772257289,1.8523740522,0.0284334225)],
        ['1C',(0.6507817278,3.1273424315,0.0022203008)],
        ['1H',(1.3238374754,3.1759072178,0.8641918504)],
        ['1H',(1.2547242432,3.1840825013,-0.9090700422)],
        ['1H',(-0.0028365383,4.0047068888,0.0317136192)],
        ]'''
        
        charge = 0
    elif mol_str == 'do':
        # decanol
        atom = [
        ['1H',       (-6.3915339418,      0.5135560065,     -0.1266068112)],
        ['1H',       (-5.5759451539,      1.6268604868,      0.9847756715)],
        ['1H',       (-5.5336618885,      1.9272190749,     -0.7636611566)],
        ['1C',       (-5.5082281804,      1.1465555280,      0.0033764198)],
        ['1H',       (-4.2546369935,     -0.4670883824,      0.6618892364)],                 
        ['1C',       (-4.2413590731,      0.3130516405,     -0.1079638091)],                 
        ['1H',       (-4.2273477731,     -0.1911193091,     -1.0813049604)],                 
        ['2H',       (-3.0176099186,      1.6973879664,      1.0105133884)],                 
        ['2C',       (-2.9898990310,      1.1765954756,      0.0455263865)],                 
        ['2H',       (-2.9791320260,      1.9461143351,     -0.7360537600)],                 
        ['2C',       (-1.7149432748,      0.3358792908,     -0.0438036514)],                 
        ['2H',       (-1.7078843099,     -0.4071992211,      0.7628613943)],                 
        ['2H',       (-1.7018910570,     -0.2152029635,     -0.9919726637)],                 
        ['2H',       (-0.4867321492,      1.7795126123,      0.9935364067)],                 
        ['2H',       (-0.4722832737,      1.9469813639,     -0.7631866462)],                 
        ['2C',       (-0.4669798299,      1.2147365849,      0.0534801331)],                 
        ['3C',       ( 0.8141262239,      0.3817846137,     -0.0145574511)],                 
        ['3H',       ( 0.8238850967,     -0.2056890397,     -0.9406431365)],                 
        ['3H',       ( 0.8347589417,     -0.3291271423,      0.8203581676)],                 
        ['3H',       ( 2.0352378183,      1.9779885571,     -0.8034884711)],                 
        ['3H',       ( 2.0391415317,      1.8720069055,      0.9581118742)],                 
        ['3C',       ( 2.0549641584,      1.2748748985,      0.0381999240)],                 
        ['3C',       ( 3.3423422066,      0.4510337476,     -0.0140326993)],                 
        ['3H',       ( 3.3553142887,     -0.1550670581,     -0.9280186193)],                 
        ['3H',       ( 3.3690824146,     -0.2427197929,      0.8350021502)],                 
        ['3H',       ( 4.5575692899,      2.0491946438,     -0.8264556134)],                 
        ['3H',       ( 4.5669939782,      1.9749674663,      0.9226006501)],                 
        ['3C',       ( 4.5757303875,      1.3512952242,      0.0199907780)],                 
        ['4C',       ( 5.8652491970,      0.5421996986,     -0.0193640166)],                 
        ['4H',       ( 5.9288616715,     -0.1314700343,      0.8407647533)],                 
        ['4H',       ( 5.9319084877,     -0.0586311472,     -0.9320214525)],                 
        ['5H',       ( 6.9310923110,      2.0020988812,     -0.7483192270)],                 
        ['5O',       ( 6.9844390797,      1.4171748810,      0.0267715145)]                 
        ]
        '''
        atom = [
        ['5O',(0.8054735974,-9.3420382053,0.0353087362)],
        ['5H',(1.3697769291,-9.3088185004,-0.7559193892)],
        ['4H',(-0.6755047413,-8.2671061332,-0.8900774814)],
        ['4H',(-0.6932729600,-8.2425548007,0.8838214096)],
        ['4C',(-0.0452011880,-8.2038709186,0.0027854300)],
        ['3H',(1.4573882730,-6.9350758580,-0.8721378670)],
        ['3C',(0.7930542795,-6.9323021030,0.0009886700)],
        ['3H',(1.4513960414,-6.9310142181,0.8786803493)],
        ['3H',(-0.7341687447,-5.6927061785,-0.8875514224)],
        ['3H',(-0.7319404956,-5.6826008554,0.8774764308)],
        ['3C',(-0.0818569702,-5.6806372246,-0.0058932003)],
        ['3C',(0.7675442633,-4.4087137461,-0.0149012253)],
        ['3H',(1.4292643921,-4.4025733808,0.8598219987)],
        ['3H',(1.4082871108,-4.4016422246,-0.9050758547)],
        ['3H',(-0.7747489278,-3.1620329276,-0.8729820752)],
        ['3H',(-0.7423591656,-3.1587151694,0.8918222234)],
        ['3C',(-0.1074800621,-3.1541755000,-0.0025427322)],
        ['2C',(0.7433818604,-1.8832140314,-0.0212567011)],
        ['2H',(1.4218493465,-1.8822812014,0.8405414495)],
        ['2H',(1.3668352441,-1.8724947105,-0.9235874513)],
        ['2H',(-0.7469241413,-0.6358392457,0.9208158260)],
        ['2H',(-0.8136590737,-0.6305818342,-0.8430166077)],
        ['2C',(-0.1295541871,-0.6276996893,0.0142828181)],
        ['2H',(1.4148901432,0.6403246829,0.8359075732)],
        ['2C',(0.7222512572,0.6424446028,-0.0145758031)],
        ['2H',(1.3310737799,0.6553314776,-0.9268606058)],
        ['1H',(-0.7516277534,1.8937042882,0.9536061281)],
        ['1C',(-0.1483703477,1.8975655502,0.0382882599)],
        ['1H',(-0.8439728367,1.9049520548,-0.8089000372)],
        ['1C',(0.6949235125,3.1621396561,0.0023646528)],
        ['1H',(1.3824817176,3.1997072608,0.8533631155)],
        ['1H',(1.2844291406,3.2143159919,-0.9185646251)],
        ['1H',(0.0527192771,4.0473747624,0.0451562090)]
        ]'''
        
        charge = 0
    elif mol_str == 'cl':
        atom = 'Cl 0 0 0'
        charge = -1
    elif mol_str == 'oh':
        atom = '''
        H       -0.8699925594      1.0777749203      0.9131409488                 
        O       -0.5995440252      1.1886505771      0.0199429032                 
        '''
        charge = -1
    
    mol = gto.Mole()
    mol.build(parse_arg = False,
              verbose = 1, # set to 5 to see the make_dmet output
              atom = atom,
              basis = 'def2-TZVPP',
              #basis = 'def2-TZVP',
              charge = charge)
    
    return mol

if __name__ == '__main__':
    # embedding goes from f to 5
    # e.g. f = 5 selects only atoms numbered 5
    # script runs through all fragment sizes
    results = dict()

    # the choice of the basis to represent the fragment can have significant
    # effects. With my work I normally use the single zeta version. The SZ def2
    # set is still polarized, which isn't good because the polarization functions
    # spatially extend beyond what you would normally think of the fragment
    minao = 'minao'
    #minao = 'def2-svp'
    #minao = 'def2-svpd'

    # do the substituting/leaving groups, Cl- and -OH
    for piece in ('cl','oh'):
        # setup calculation
        mol = init_mol(piece)
        mf = scf.HF(mol)    
        mf.kernel()
        mymp = mp.MP2(mf)
        e_corr,t2 = mymp.kernel(with_t2=False)
        results[piece] = [mymp.e_hf,mymp.e_tot]

    for piece in ('cd','do'):
        # setup calculation
        mol = init_mol(piece)
        mf = scf.HF(mol)    
        mf.kernel()

        emb_coeff = None
        frozen = None
        frag_energy = []
        # no embedding
        mymp = mp.MP2(mf,frozen=frozen,mo_coeff=emb_coeff)
        e_corr,t2 = mymp.kernel(with_t2=False)
        frag_energy.append(mymp.e_tot)
        for f in range(1,6): # fragments from 1 to 5 (see init_mol)
            aolabels = list(range(f,5+1))
            aolabels = [str(fr) for fr in aolabels]

            # embed
            ro_coeff, frozen, emb_dat = emb.make_rdmet(mol,mf,ft=(0.1,0.1),aolabels=aolabels,minao=minao)
            print(emb_dat['wocc'])
            # canonicalize
            # if the condition number is low, you can try building the Fock matrix
            # from the mo_energy in the diagonal basis, i.e. the matrix multiply SCFCS, 
            # where S is the overlap matrix, C has indices AOxMO, and F is MOxMO. If S
            # is close to singular, F in the AO basis reconstructed this way will 
            # be quite different from F built from the ground up from the 1RDM + AO ERIs;
            # the eigenvalues will be the same, but the coefficients different. In my
            # experience I used fock_ao=False with no change in the final results, but
            # the setting is not extensively tested
            emb_coeff, co_energy = emb.canonicalize(mf,ro_coeff,frozen,fock_ao=True)

            mymp = mp.MP2(mf,frozen=frozen,mo_coeff=emb_coeff)
            e_corr,t2 = mymp.kernel(with_t2=False)
            frag_energy.append(mymp.e_tot)
        frag_energy.insert(0,mf.e_tot)
        results[piece] = frag_energy

    HA2KCAL = 627.5

    print('')
    print('cl',results['cl'])
    print('do',results['do'])
    print('oh',results['oh'])
    print('cd',results['cd'])

    fullhf = results['cl'][0] + results['do'][0] - results['oh'][0] - results['cd'][0]
    print('Full system RHF: ',fullhf*HA2KCAL)

    fullmp = results['cl'][1] + results['do'][1] - results['oh'][1] - results['cd'][1]
    print('Full system MP: ',fullmp*HA2KCAL)

    for idx in range(2,7):
        mpe = results['cl'][1] + results['do'][idx] - results['oh'][1] - results['cd'][idx]
        print(f'Fragment MP from {idx-1} to 5: ',mpe*HA2KCAL)

    '''
    There is more to this example after the results shown here.

    Notice how the results track the reference data pretty well except for a large discrepancy
    in the 3-5 embedding. The fragments were generated with overlap cutoffs of 0.1 for both
    the occupied and virtual space, while Gerald used a different formula: any overlap above machine
    epsilon for occupied, and for virtuals, as many orbitals as the free fragment would have. Let's test
    that out below.

    Gerald also have used relaxed geometries, and also used IAOs to define the fragment

    ***************** REFERENCE DATA
    def2-TZVPP (from Gerald Knizia: Dmet regions presentation)
    The single particle fragment basis is unknown (assumed to be IAOs, but I do not have
    an IAO implementation for regional embedding)
    Full system RHF: -70.01 kcal/mol
    Full system MP2: -58.83 kcal/mol (same as 1-5 embedding)
    2-5 embedding: -58.83 kcal/mol
    3-5 embedding: -58.83 kcal/mol
    4-5 embedding: -59.55 kcal/mol
    5-5 embedding: -60.39 kcal/mol

    ***************** with "minao" as the basis for the fragment
    def2 TZVPP
    [blau@rusty1 ~]$ seff 999447
    Job ID: 999447
    Cluster: slurm
    User/Group: blau/blau
    State: COMPLETED (exit code 0)
    Nodes: 1
    Cores per node: 28
    CPU Utilized: 07:30:09
    CPU Efficiency: 73.80% of 10:09:56 core-walltime
    Job Wall-clock time: 00:21:47
    Memory Utilized: 371.79 GB
    Memory Efficiency: 74.36% of 500.00 GB
    Full system RHF:  -68.75764607669083
    Full system MP:  -57.863321856619905
    Fragment MP from 1 to 5:  -57.86327976020459
    Fragment MP from 2 to 5:  -57.8646844097355
    Fragment MP from 3 to 5:  -153.66104571245302
    Fragment MP from 4 to 5:  -58.62743450516149
    Fragment MP from 5 to 5:  -61.688253834254

    def2-TZVP
    Job ID: 999549
    Cluster: slurm
    User/Group: blau/blau
    State: COMPLETED (exit code 0)
    Nodes: 1
    Cores per node: 128
    CPU Utilized: 09:44:22
    CPU Efficiency: 82.01% of 11:52:32 core-walltime
    Job Wall-clock time: 00:05:34
    Memory Utilized: 110.85 GB
    Memory Efficiency: 11.08% of 1000.00 GB
    cl [-459.55562241678035, -459.84976128771905]
    do [-466.5259789637478, -468.6588154525862, -468.65881555243465, -468.6465812057628, -468.21695424674436, -467.28741280793963, -467.0290459725071]
    oh [-75.39815634976748, -75.67935234367455]
    cd [-850.5733052946549, -852.7350595347253, -852.7350596801332, -852.7228288756917, -852.1555324666176, -851.3624916395347, -851.0995007591933]
    Full system RHF:  -69.11268440631403
    Full system MP:  -59.08845084561932
    Fragment MP from 1 to 5:  -59.08842225707588
    Fragment MP from 2 to 5:  -59.08619950757867
    Fragment MP from 3 to 5:  -145.473779417448
    Fragment MP from 4 to 5:  -59.819645562036214
    Fragment MP from 5 to 5:  -62.721233742331606

    ***************** with "def2-svpd" as the basis for the fragment
    def2-TZVP
    cl [-459.5556224167799, -459.8497612877186]
    do [-466.52597896374687, -468.6588154525853, -468.6588155524308, -468.64658120576235, -468.43516223867437, -467.8118096583208, -467.1387320063902]
    oh [-75.39815634976735, -75.67935234367442]
    cd [-850.5733052946536, -852.7350595347241, -852.7350596801357, -852.7228288756895, -852.4515424938136, -851.8880473235891, -851.2028676096833]
    Full system RHF:  -69.11268440645671
    Full system MP:  -59.08845084561932
    Fragment MP from 1 to 5:  -59.088422252866906
    Fragment MP from 2 to 5:  -59.08619950843473
    Fragment MP from 3 to 5:  -96.65300228787913
    Fragment MP from 4 to 5:  -59.0924774317989
    Fragment MP from 5 to 5:  -66.68652132135207

    ***************** with "def2-svp" as the basis for the fragment
    def2-TZVPP
    cl [-459.55562241678035, -459.9175126205242]
    do [-466.53435239135865, -468.7366768536234, -468.7366769610637, -468.72574731301796, -468.55193407901805, -467.5482830324114, -467.11743178342397]
    oh [-75.40009928417328, -75.68644329047555]
    cd [-850.5803015859973, -852.8755337185644, -852.8755338930786, -852.8646020065685, -852.6350772959942, -851.6866465791962, -851.249152514187]
    Full system RHF:  -68.75764607519272
    Full system MP:  -57.86332185505046
    Fragment MP from 1 to 5:  -57.863279766125686
    Fragment MP from 2 to 5:  -57.864684402530315
    Fragment MP from 3 to 5:  -92.82363595296744
    Fragment MP from 4 to 5:  -58.17287899803176
    Fragment MP from 5 to 5:  -62.341246051699386

    def2-TZVP
    cl [-459.55562241677995, -459.84976128771865]
    do [-466.52597896374823, -468.6588154525867, -468.6588155524349, -468.6465812057656, -468.43939168109546, -467.50707646058305, -467.09727692956994]
    oh [-75.39815634976739, -75.67935234367447]
    cd [-850.5733052946536, -852.7350595347241, -852.735059680133, -852.722828875692, -852.4570572691559, -851.5829026190635, -851.1665862329781]
    Full system RHF:  -69.11268440724143
    Full system MP:  -59.08845084647538
    Fragment MP from 1 to 5:  -59.088422257147215
    Fragment MP from 2 to 5:  -59.08619950886276
    Fragment MP from 3 to 5:  -95.84645587971067
    Fragment MP from 4 to 5:  -59.35069794123194
    Fragment MP from 5 to 5:  -63.44002449901467
    '''

    '''
    Threshold a different way:
    Occupied: all orbitals above machine epsilon
    Virtual: number of orbitals the fragment would have had
    = (number of orbitals on fragment) - (number of electrons on fragment) / 2
    '''

    for piece in ('cd','do'):
        # setup calculation
        mol = init_mol(piece)
        mf = scf.HF(mol)
        mf.kernel()
        
        emb_coeff = None
        frozen = None
        frag_energy = []
        # no embedding
        mymp = mp.MP2(mf,frozen=frozen,mo_coeff=emb_coeff)
        e_corr,t2 = mymp.kernel(with_t2=False)
        frag_energy.append(mymp.e_tot)
        for f in range(1,6): # fragments from 1 to 5 (see init_mol)
            aolabels = list(range(f,5+1))
            aolabels = [str(fr) for fr in aolabels]

            # change is here ***************************************************
            # virtual orbitals
            nfo = len(mol.search_ao_label(aolabels))
            nfe = emb.get_frag_nelec(mol,aolabels)/2
            ft = (numpy.finfo(float).eps,int(nfo-nfe))
            # ******************************************************************
            
            # embed
            ro_coeff, frozen, emb_dat = emb.make_rdmet(mol,mf,ft=ft,aolabels=aolabels,minao=minao)
            print(emb_dat['wocc'])
            # canonicalize
            emb_coeff, co_energy = emb.canonicalize(mf,ro_coeff,frozen,fock_ao=True)

            mymp = mp.MP2(mf,frozen=frozen,mo_coeff=emb_coeff)
            e_corr,t2 = mymp.kernel(with_t2=False)
            frag_energy.append(mymp.e_tot)
        frag_energy.insert(0,mf.e_tot)
        results[piece] = frag_energy

    print('')
    print('cl',results['cl'])
    print('do',results['do'])
    print('oh',results['oh'])
    print('cd',results['cd'])

    fullhf = results['cl'][0] + results['do'][0] - results['oh'][0] - results['cd'][0]
    print('Full system RHF: ',fullhf*HA2KCAL)

    fullmp = results['cl'][1] + results['do'][1] - results['oh'][1] - results['cd'][1]
    print('Full system MP: ',fullmp*HA2KCAL)

    for idx in range(2,7):
        mpe = results['cl'][1] + results['do'][idx] - results['oh'][1] - results['cd'][idx]
        print(f'Fragment MP from {idx-1} to 5: ',mpe*HA2KCAL)

    '''
    def2-SVP basis for the fragment

    original way:
    cl [-459.55562241678, -459.91751262052384]
    do [-466.5343523913589, -468.7366768536236, -468.73667696106105, -468.72574731302603, -468.5519340790258, -467.5482830324109, -467.11743178342397]
    oh [-75.40009928417334, -75.68644329047561]
    cd [-850.5803015859962, -852.8755337185632, -852.8755338930787, -852.8646020065704, -852.6350772959977, -851.6866465791944, -851.2491525141858]
    Full system RHF:  -68.75764607583477
    Full system MP:  -57.863321855621166
    Fragment MP from 1 to 5:  -57.86327976419955
    Fragment MP from 2 to 5:  -57.86468440616858
    Fragment MP from 3 to 5:  -92.82363595539294
    Fragment MP from 4 to 5:  -58.17287899860247
    Fragment MP from 5 to 5:  -62.341246052198755

    gerald's way:
    cl [-459.55562241678, -459.91751262052384]
    do [-466.53435239135956, -468.7366768536243, -468.7366769610687, -468.7366769610635, -468.69870997939114, -468.07947157116365, -467.2830796160987]
    oh [-75.40009928417334, -75.68644329047561]
    cd [-850.5803015859952, -852.8755337185623, -852.8755338930756, -852.875533893082, -852.8110071703042, -852.220216745312, -851.4129953903828]
    Full system RHF:  -68.75764607683351
    Full system MP:  -57.863321856619905
    Fragment MP from 1 to 5:  -57.8632797709767
    Fragment MP from 2 to 5:  -57.86327976370018
    Fragment MP from 3 to 5:  -74.52951730734867
    Fragment MP from 4 to 5:  -56.67840782719168
    Fragment MP from 5 to 5:  -63.47385624196363

    As you can see, it's better but not fixed. If we print the overlap eigenvalues, even for the smallest
    fragment (5-5, Cl/OH only) we see that all of the eigenvalues are above the threshold of eps:
    [4.60528387e-10 9.88878496e-08 2.88661339e-06 6.57675602e-06
    1.03771601e-05 7.11283549e-05 1.42426395e-04 4.53252184e-04
    1.04413371e-03 2.06235682e-03 2.62869724e-03 3.20005811e-03
    7.74456753e-03 1.24324317e-02 1.26813881e-02 1.37723564e-02
    1.56435025e-02 3.44389176e-02 5.24769412e-02 5.75979727e-02
    7.31331511e-02 8.70774409e-02 1.20584814e-01 1.37452653e-01
    1.47003853e-01 2.24079492e-01 2.95688974e-01 3.28830306e-01
    4.65481020e-01 8.72936727e-01 8.78477120e-01 8.92187198e-01
    9.02567356e-01 9.13315343e-01 9.25982944e-01 9.26618399e-01
    9.38923214e-01 9.58967729e-01 9.59625140e-01 9.93719770e-01
    9.95866007e-01 9.98996050e-01 9.99905469e-01 9.99906859e-01
    9.99932087e-01]
    So all of the occupied orbitals (i.e. the whole molecule) are being selected to be in the fragment, 
    and the restriction of the VIRTUAL orbitals is what ends up defining the fragment! I find it fascinating
    that you only need to restrict the virtual orbitals to define a spatial fragment, maybe because the
    virtual orbitals are the main ingredient for post-HF.

    Next, I try the 'minao' basis (with def2-TZVP):
    Full system RHF:  -69.11268440524395
    Full system MP:  -59.088450844477904
    Fragment MP from 1 to 5:  -59.08842225258155
    Fragment MP from 2 to 5:  -59.0884222597154
    Fragment MP from 3 to 5:  -80.13057320632441
    Fragment MP from 4 to 5:  -78.95075727676499
    Fragment MP from 5 to 5:  -64.3382545432118

    Which gets close to fixing the problem , but still not entirely. At this point I am out of ideas. 
    I thought this would be a quick fix and a great example, but solving this problem looks like
    it will quickly grow in scope into an actual research project, so I'll leave it here for now!
    '''
