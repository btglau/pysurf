#!/usr/bin/python
# -*- coding: utf-8 -*-

########################
#
# Calculation GS Urea
#
########################

#import libraries

import numpy as np
import pyscf
from pyscf import scf,gto,ao2mo,cc
from pyscf.cc import ccsd_t

#####################
# Defining Molecule #
#####################

# Creating new molecule
mol=gto.Mole()

# Geometry (optimized B3LYP/6-31G*)
mol.atom="""
C1  0.0000   0.0000   0.1449
O2  0.0000   0.0000   1.3650
N3     -0.1309   1.1569  -0.6170
N4  0.1309  -1.1569  -0.6170
H5  0.0000   1.9959  -0.0667
H6  0.3478   1.1778  -1.5093
H7  0.0000  -1.9959  -0.0667
H8     -0.3478  -1.1778  -1.5093
"""

#Defining basis set and ECP
mol.basis={'C': 'cc-pVDZ', 'O': 'cc-pVDZ', 'N':'cc-pVDZ', 'H':'cc-pVDZ'}

# unit for the distance
mol.unit='angstrom'

# print level: 0--> almost nothing, 10-->detailed infos 
mol.verbose=4

# build molecule
mol.build()

# HF calculation
# Create HF object for the "mol" attribute 
mf=scf.RHF(mol).run()

# CCSD calculation
# Create cc object from mf attribute
mycc = cc.RCCSD(mf)
# Run RCCSD 
mycc.run()

#---------- SPECTRA SECTION --------------

# Convergence parameters
gmres_tol = 1e-1 # The tolerance parameter for converging the linear equations
dpq_tol = 1e-3 # Elements of the dipole matrix smaller than this are discarded
n_frozen = 40 # Number of virtual orbitals to freeze

# Spectrum details - the frequency range of interest, the spacing of the freq. grid
# and the broadening eta. Larger eta *should* mean faster convergence. All values in a.u.
wmin = 0.0
wmax = 1.0
dw = 0.1
eta = 0.07

nw = int((wmax-wmin)/dw) + 1
ws = np.linspace(wmin, wmax, nw)

# Setup cc with frozen orbitals if necessary
nmo = len(mf.mo_energy)
freeze_range = range(nmo-n_frozen,nmo)
if n_frozen != 0: mycc = cc.RCCSD(mf,frozen = freeze_range)
if n_frozen != 0: mycc.run()

# Calculating the spectrum requires the lambda amplitudes
mycc.solve_lambda()

# Setup spectrum object
from pyscf.cc import gf
mygf = gf.SpectralFunctions(mycc)
mygf.verbose = 4
mygf.dpq_tol = dpq_tol
mygf.gmres_tol = gmres_tol

# Calculate and output the absorption spectrum
spectrum = mygf.get_abs_spec(ws,eta)
spectrum = np.sum(spectrum,axis = 0)
np.savetxt("abs_spectrum.txt", np.column_stack([ws,spectrum]))
        
#---------- SPECTRA SECTION --------------
