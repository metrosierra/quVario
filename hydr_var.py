#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:42:53 2020

Variational Method for finding ground state energy of Hydrogen
The Hamiltonian is fixed at the moment

@author: kenton
"""
#%% Import packages and initialisation of spaces
#from scipy.misc import derivative
from sympy.vector import CoordSys3D, Del
from sympy import diff, exp, conjugate, symbols, integrate, solve

#initialisation of coordinate systems for vector calculus
R = CoordSys3D('R', transformation = 'spherical')
#rlen = r.magnitude()
delop = Del()

'''
Sympy help

Note that R.r, R.theta, R.phi are the variables called base scalars
    Base scalars can be treated as symbols
and R.i, R.j, R.k are the r, theta, phi basis vectors

https://docs.sympy.org/latest/tutorial/basic_operations.html

'''
#%%
#Symbolic parameters and numerical infinity

#N, alpha = symbols('N alpha', real=True)
alpha = symbols('alpha', real = True)
#N = 1
#alpha = 1
inf = 10

#%% Define functions
#Hamiltonian operator H operating on psi
def H(psi):
    '''
    The Hamiltonian operator operating on some trial function
    input: some trial function psi(r)
    output: another function
    '''
    H = - 1 / 2 * del2(psi) - 1 / R.r * psi
    
    return H

#Del-squared, the Laplacian for the Schrodinger equation
def del2(f):
    '''
    Laplacian operator in spherical polar coordinates
    input: some function f and point r
    output: 
    '''
    
    del2 = delop.dot(delop(f)).doit()
    
    return del2

#Trial function psi
#def psi():
#    '''
#    The trial function
#    input:
#    output: 
#    '''
#    return N * exp(- alpha * R.r **2)

psi = exp(-alpha * R.r **2)

#%% Testing area
def f (r):
    '''
    Random scalar field f
    '''
    return 4* R.i * R.r
#    try:
#        f =  1/ (r) ** 2
#    except ZeroDivisionError:
#        f = 9e9999
#        pass
#    return f
r = 1.0
#print(derivative((r ** 2) * derivative(f, 1, dx= 1e-6), r))
    
#print(del2(f, 1.0))


#%%
#The expectation value integral <psi|H|psi>

expectation = conjugate(psi) * H(psi)
#print(expectation)

#the normalisation value <psi|psi> 
absolute = integrate(conjugate(psi) * psi, (R.r,0,100)).evalf()

solve(diff(integrate(expectation, (R.r ,0.001,inf)), alpha), alpha) 
