#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:07:01 2020

@author: blau

A modification of pyscf's cubegen to allow sheared volumetric cells

Takes a supercell, replicates it back by -1 (for a total of 2x2 super-supercell)
Then places the center of the grid on the origin (0,0,0)
This geometry is chosen to better visualize the adsorbate/defect orbitals

"""

from pyscf import __version__ 
import time
import numpy
from pyscf.tools import cubegen
from pyscf.pbc import gto, tools
from pyscf.lib import prange

RESOLUTION = cubegen.RESOLUTION

def orbital(args, mol, outfile, coeff, nx=80, ny=80, nz=80, resolution=RESOLUTION):
    """
    copied from cubegen.orbital()
    """
    cc = Cube(args, mol, nx, ny, nz, resolution)

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = numpy.empty(ngrids)
    for ip0, ip1 in prange(0, ngrids, blksize):
        ao = mol.eval_gto('PBCGTOval', coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = numpy.dot(ao, coeff)
    orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)

    # Write out orbital to the .cube file
    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')
    
    return orb_on_grid

class Cube(cubegen.Cube):
    
    '''  Read-write of the Gaussian CUBE files  '''
    def __init__(self, args, mol, nx=80, ny=80, nz=80, resolution=RESOLUTION):
        self.mol = mol
        self.args = args

        # If the molecule is periodic, use lattice vectors as the box
        # and ignore margin, origin, and extent arguments
        a = mol.lattice_vectors() # returns Bohrs
        
        self.boxorig = numpy.zeros(3)
        # shift origin back by half a unit cell to center around the defect
        self.boxorig -= a[0,:]/2
        self.boxorig -= a[1,:]/2

        # setup calculation
        if args.sys in ('g','hbn','lih'):
            # only shift down by 1/4 for vacuum height
            self.boxorig -= a[2,:]/4
            # last 3 entries are water - take first entry of water
            defect_shift = mol.atom[-3][1]
            defect_shift = 0
        else: # f-center, center lower left corner at origin
            self.boxorig -= a[2,:]/2
            defect_shift = 0
        self.boxorig += defect_shift
        
        if resolution is not None:
            nx, ny, nz = numpy.ceil(numpy.linalg.norm(a,axis=1) / resolution).astype(int)
        a /= numpy.asarray([nx,ny,nz])
        
        # generate the coordinates for density evaluation by tiling an atom over
        # the number of grid points nx ny nz
        coord_cell = gto.Cell()
        coord_cell.build(False,False,
                         unit='B',
                         a = a,
                         atom = [['H',self.boxorig]])
        self.coord_cell = tools.super_cell(coord_cell,[nx,ny,nz])

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.xs = a[0,:]
        self.ys = a[1,:]
        self.zs = a[2,:]
            
    def get_coords(self) :
        """  Result: set of coordinates to compute a field which is to be stored
        in the file.
        """
        # atom_coords returns in units of bohrs
        coords = numpy.ascontiguousarray(self.coord_cell.atom_coords())
        return coords

    def write(self, field, fname, comment=None):
        """  Result: .cube file with the field in the file fname.  """
        assert(field.ndim == 3)
        assert(field.shape == (self.nx, self.ny, self.nz))
        if comment is None:
            comment = 'Generic field? Supply the optional argument "comment" to define this line'

        # make a 222 supercell for cubegen to print out atoms
        # the orbital values which are centered either at the origin or the water
        # then have extra atoms to make their interpretation less ambiguous (vs
        # a 111 cell)
        mol = self.mol
        a = mol.lattice_vectors()

        args = self.args
        if args.sys in ('g','hbn','lih'):
            mol = tools.super_cell(mol,[2,2,1]) # 2D
        else:
            mol = tools.super_cell(mol,[2,2,2]) # 3D
        
        coords = mol.atom_coords()
        coords -= a[0,:]
        coords -= a[1,:]
        if args.sys not in ('g','hbn','lih'):
            coords -= a[2,:] # 3D
        with open(fname, 'w') as cf:
            cf.write(comment+'\n')
            cf.write('PySCF Version: %s  Date: %s\n' % (__version__, time.ctime()))
            cf.write('%5d' % mol.natm)
            cf.write('%12.6f%12.6f%12.6f\n' % tuple(self.boxorig.tolist()))
            cf.write('%5d%12.6f%12.6f%12.6f\n' % (self.nx, self.xs[0], self.xs[1], self.xs[2]))
            cf.write('%5d%12.6f%12.6f%12.6f\n' % (self.ny, self.ys[0], self.ys[1], self.ys[2]))
            cf.write('%5d%12.6f%12.6f%12.6f\n' % (self.nz, self.zs[0], self.zs[1], self.zs[2]))
            for ia in range(mol.natm):
                atmsymb = mol.atom_symbol(ia)
                cf.write('%5d%12.6f'% (gto.charge(atmsymb), 0.))
                cf.write('%12.6f%12.6f%12.6f\n' % tuple(coords[ia]))

            for ix in range(self.nx):
                for iy in range(self.ny):
                    for iz0, iz1 in prange(0, self.nz, 6):
                        fmt = '%13.5E' * (iz1-iz0) + '\n'
                        cf.write(fmt % tuple(field[ix,iy,iz0:iz1].tolist()))