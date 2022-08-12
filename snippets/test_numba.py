#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:20:22 2020

@author: blau
"""

import numpy
from numba import jit
import scipy.linalg
import timeit


def make_rdm1_grad(u,mu,nu,occ_coeff,vir_coeff,Y,Dgrad):
    '''
    evaluate gradient for one position of the correlation potential
    '''
    #u, mu, nu = umunu
    
    X = u*vir_coeff.conj().T[:,mu,None]*occ_coeff[None,nu,:]
    X += u.conj()*vir_coeff.conj().T[:,nu,None]*occ_coeff[None,mu,:]
    Z = -X*Y
    Z = vir_coeff @ Z @ occ_coeff.conj().T
    dD = Z + Z.conj().T
    
    return (Dgrad*dD).sum()

#@jit(nopython=True)
def make_rdm1_grad_virH(u,mu,nu,occ_coeff,vir_coeff,Y,Dgrad):
    '''
    evaluate gradient for one position of the correlation potential
    
    vir_coeff is actually vir_coeff^H, but I preapply H because numba does not like
    reshaping a noncontiguous array (formed from conj and T which create views)
    '''
    #u, mu, nu = umunu
    
    X = u*vir_coeff[:,mu].reshape((-1,1))*occ_coeff[nu,:].reshape((1,-1))
    X += u.conj()*vir_coeff[:,nu].reshape((-1,1))*occ_coeff[mu,:].reshape((1,-1))
    Z = -X*Y
    Z = vir_coeff.conj().T @ Z @ occ_coeff.conj().T
    dD = Z + Z.conj().T
    
    return (Dgrad*dD).sum()

#@jit(nopython=True)
def make_rdm1_grad_slice(u,mu,nu,occ_coeff_slice,vir_coeff_slice,occ_coeff,vir_coeff,Y,Dgrad):
    '''
    evaluate gradient for one position of the correlation potential
    
    vir_coeff is actually vir_coeff^H, but I preapply H because numba does not like
    reshaping a noncontiguous array (formed from conj and T which create views)
    '''
    #u, mu, nu = umunu
    X = u*vir_coeff_slice[mu]*occ_coeff_slice[nu]
    X += u.conj()*vir_coeff_slice[nu]*occ_coeff_slice[mu]
    Z = -X*Y
    Z = vir_coeff @ Z @ occ_coeff.conj().T
    dD = Z + Z.conj().T
    
    return (Dgrad*dD).sum()

def make_rdm1_grad_full(U,occ_coeff,vir_coeff,Y,Dgrad):
    '''
    evaluate gradient for one position of the correlation potential
    '''
    #u, mu, nu = umunu
    
    Z = -(vir_coeff.conj().T @ U @ occ_coeff)*Y
    Z = vir_coeff @ Z @ occ_coeff.conj().T
    dD = Z + Z.conj().T
    
    return (Dgrad*dD).sum()

if __name__ == '__main__':
    print('test various gradient functions, compare them to numerical evaluation')
    
    rng = numpy.random.default_rng()
    
    N = 50
    Npair = N*(N+1)//2
    nocc = 10
    
    U = rng.random((N,N))
    U += U.conj().T
    Ucopy = U.copy()
    
    # Dcorr - Dmf
    Dgrad = rng.random((N,N))
    Dgrad += Dgrad.conj().T
    
    U[numpy.diag_indices(N)] *= 0.5
    u = U[numpy.tril_indices(N)]
    ur, uc = numpy.tril_indices(N)

    grad = numpy.zeros(Npair)
    
    F = rng.random((N,N))
    F += F.conj().T
    F += U
    
    e,c = scipy.linalg.eigh(F,None)
    occ_coeff = c[:,:nocc]
    vir_coeff = c[:,nocc:]
    Y = 1/(e[nocc:,None] - e[None,:nocc])
    
    Dmf = occ_coeff @ occ_coeff.conj().T
    
    # pre-slice the coeffs into contiguous row/column vectors
    occ_coeff_slice = []
    vir_coeff_slice = []
    for a in range(N):
        occ_coeff_slice.append(occ_coeff[None,a,:].copy())
        vir_coeff_slice.append(vir_coeff.conj().T[:,a,None].copy())
    vir_coeffH = vir_coeff.conj().T.copy()    
    
    # vanilla implementation
    start_time = timeit.default_timer()
    a = 0
    for umn,mu,nu in zip(u,ur,uc):
        grad[a] = make_rdm1_grad(umn,mu,nu,occ_coeff,vir_coeff,Y,Dgrad)
        a += 1
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    # vir_coeff -> vir_coeff^H
    grad_virH = numpy.zeros(Npair)
    start_time = timeit.default_timer()
    a = 0
    for umn,mu,nu in zip(u,ur,uc):
        grad_virH[a] = make_rdm1_grad_virH(umn,mu,nu,occ_coeff,vir_coeffH,Y,Dgrad)
        a += 1
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    print('grad to grad_virH',numpy.allclose(grad,grad_virH))
    
    # preslice and apply ^H
    grad_slice = numpy.zeros(Npair)
    start_time = timeit.default_timer()
    a = 0
    for umn,mu,nu in zip(u,ur,uc):
        grad_slice[a] = make_rdm1_grad_slice(umn,mu,nu,occ_coeff_slice,vir_coeff_slice,occ_coeff,vir_coeff,Y,Dgrad)
        a += 1
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    print('grad to pre-slicing grad',numpy.allclose(grad,grad_slice))
    
    u = Ucopy[numpy.tril_indices(N)]
    # build U
    grad_full = numpy.zeros(Npair)
    start_time = timeit.default_timer()
    a = 0
    for umn,mu,nu in zip(u,ur,uc):
        dU = numpy.zeros(U.shape)
        dU[mu,nu] = umn
        dU[nu,mu] = umn.conj()
        grad_full[a] = make_rdm1_grad_full(dU,occ_coeff,vir_coeff,Y,Dgrad)
        a += 1
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    print('grad to full U',numpy.allclose(grad,grad_full))
    
    # check if numerical gradient is the same as the analytical gradient
    # 
    grad_num = numpy.zeros(Npair)
    a = 0
    h = 1e5
    for umn,mu,nu in zip(u,ur,uc):   
        dU = numpy.zeros(U.shape)
        step = umn/h
        dU[mu,nu] = step
        dU[nu,mu] = step.conj()
        e,c = scipy.linalg.eigh(F+dU,None)  
        dD = c[:,:nocc] @ c[:,:nocc].conj().T
        
        dU *= -1
        e,c = scipy.linalg.eigh(F+dU,None)  
        dD2 = c[:,:nocc] @ c[:,:nocc].conj().T
        
        grad_num[a] = (Dgrad*(dD-dD2)).sum()/(2*step)
        a += 1
    print(elapsed)
    print('grad to finite diff',numpy.allclose(grad,grad_num))
    print('grad/umn to finite diff',numpy.allclose(grad/u,grad_num))
    print('gets signs? {}'.format(numpy.all((numpy.sign(grad)-numpy.sign(grad_num))==0)))
    