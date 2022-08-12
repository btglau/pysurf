#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:08:55 2020

@author: blau

An overloaded molden part to output [pseudo] for molden instead of [core] for
Molden2Aim

[Pseudo]
Fe 1 16
elementname number_of_atom effective_core_charge
"""

from pyscf.tools.molden import *

def header(mol, fout, ignore_h=IGNORE_H):
    logger.info(mol,'Pseudo output was chosen')
    if ignore_h:
        mol = remove_high_l(mol)[0]
    fout.write('[Molden Format]\n')
    fout.write('made by pyscf v[%s]\n' % pyscf.__version__)
    fout.write('[Atoms] (AU)\n')
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        chg = mol.atom_charge(ia)
        fout.write('%s   %d   %d   ' % (symb, ia+1, chg))
        coord = mol.atom_coord(ia)
        fout.write('%18.14f   %18.14f   %18.14f\n' % tuple(coord))

    fout.write('[GTO]\n')
    for ia, (sh0, sh1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
        fout.write('%d 0\n' %(ia+1))
        for ib in range(sh0, sh1):
            l = mol.bas_angular(ib)
            nprim = mol.bas_nprim(ib)
            nctr = mol.bas_nctr(ib)
            es = mol.bas_exp(ib)
            cs = mol.bas_ctr_coeff(ib)
            for ic in range(nctr):
                fout.write(' %s   %2d 1.00\n' % (lib.param.ANGULAR[l], nprim))
                for ip in range(nprim):
                    fout.write('    %18.14g  %18.14g\n' % (es[ip], cs[ip,ic]))
        fout.write('\n')

    if mol.cart:
        fout.write('[6d]\n[10f]\n[15g]\n')
    else:
        fout.write('[5d]\n[7f]\n[9g]\n')

    # overwriting this part to output [Pseudo] instead of [core] according
    # to the molden format
    if mol.has_ecp():  # See https://github.com/zorkzou/Molden2AIM
        fout.write('[Pseudo]\n')
        for ia in range(mol.natm):
            symb = mol.atom_pure_symbol(ia)
            nelec_ecp_core = mol.atom_nelec_core(ia)
            if nelec_ecp_core != 0:
                fout.write('%s   %d   %d\n' % (symb, ia+1, nelec_ecp_core))
    fout.write('\n')