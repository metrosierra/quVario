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
from sympy import (exp, conjugate, symbols, simplify, integrate, lambdify)
from scipy.optimize import fmin
from scipy.integrate import quad
import scipy as sp

#initialisation of coordinate systems for vector calculus
#we work in Spherical coordinates because of the nature of the problem
R = CoordSys3D('R', transformation = 'spherical')
delop = Del() #by convention

'''
Sympy help

Note that R.r, R.theta, R.phi are the variables called base scalars
    Base scalars can be treated as symbols
and R.i, R.j, R.k are the r, theta, phi basis vectors

https://docs.sympy.org/latest/tutorial/basic_operations.html

Symplify may be of use later

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
    The Hamiltonian operator operating on some trial function psi
    Input: some trial function psi(r)
    Output: another scalar field Sympy expression
    '''
    H = - 1 / 2 * del2(psi) - 1 / R.r * psi
    
    return H

#Del-squared, the Laplacian for the Schrodinger equation
def del2(f):
    '''
    Laplacian operator in spherical polar coordinates
    Input: some Sympy expression f to which the Laplacian is applied 
    Output: A Sympy expression
    '''
    
    del2 = delop.dot(delop(f)).doit()
    
    return del2

#Trial function psi defined using SymPy
psi = exp(-alpha * R.r **2)

#def psi():
#    '''
#    The trial function
#    input:
#    output: 
#    '''
#    return N * exp(- alpha * R.r **2)

#%% Testing area
#def f (r):
#    '''
#    Random scalar field f
#    '''
#    return 4* R.i * R.r
#    try:
#        f =  1/ (r) ** 2
#    except ZeroDivisionError:
#        f = 9e9999
#        pass
#    return f
#r = 1.0
#print(derivative((r ** 2) * derivative(f, 1, dx= 1e-6), r))
#print(del2(f, 1.0))


#%% This part executes the variation method
#The expectation value  <psi|H|psi>
expectation = simplify(conjugate(psi) * H(psi) * R.r **2)
#print(expectation)

#the normalisation value <psi|psi> 
absolute = integrate(conjugate(psi) * psi, (R.r,0,100)).evalf()

#The lambdify function converts Sympy expressions into something more suitable
#for numerical evaluation with SciPy
expectation_lamb = lambdify((R.r, alpha), expectation, 'scipy')

#now define a function to be optimsied
#def expectationfunc(r, alpha):
#    '''
#    This function concerts the SymPy expression with base scalars and vectors 
#        into a Python function that can be inputted and integrated using SciPy
#    Input:
#    Output:
#    '''
#    
#    expectationfunc = expectation.subs([(R.r, a), (r, alpha)])
#    return expectationfunc
#solve(diff(integrate(expectation, (R.r ,0.001,inf)), alpha), alpha) 

# definining a function such that SciPy Optimisation can be used
def expectation_intgrl(a):
    '''
    
    returns 
    '''
    expectation_integral = quad(expectation_lamb, 0, 16 ,args=(a))
    return expectation_integral[0]


fmin(expectation_intgrl, 5)