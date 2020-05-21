#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:42:53 2020

Variational Method for finding ground state energy of Hydrogen.
Exercise 1.19 in Quantum Chemistry. 
The Hamiltonian is fixed at the moment.

@author: kenton
"""
#%% Import packages and initialisation of spaces

from sympy.vector import CoordSys3D, Del
from sympy import (exp, conjugate, symbols, simplify, lambdify)
from scipy.optimize import fmin
from scipy.integrate import quad
#import scipy as sp
#import matplotlib.pyplot as plt 

# initialisation of coordinate systems for vector calculus
# we work in Spherical coordinates because of the nature of the problem
# this simplifies the need for the Laplacian as SymPy has sphericals built in
R = CoordSys3D('R', transformation = 'spherical')
delop = Del() #by convention


#Sympy help
#
#Note that R.r, R.theta, R.phi are the variables called base scalars
#    Base scalars can be treated as symbols
#and R.i, R.j, R.k are the r, theta, phi basis vectors
#
#https://docs.sympy.org/latest/tutorial/basic_operations.html

#%% Define symbolic parameters and numerical infinity

alpha = symbols('alpha', real = True)

inf = 1000

#%% Define functions

#Hamiltonian operator H operating on psi
def H(psi):
    ''' Function: the Hamiltonian operator operating on some trial function psi.
    This is in atomic units. 
   
    Input: 
        some trial function psi(r)
    
    Output: 
        another scalar field Sympy expression
    '''
    H = - 1 / 2 * del2(psi) - 1 / R.r * psi
    
    return H

#Del-squared, the Laplacian for the Schrodinger equation
def del2(f):
    '''Laplacian operator in spherical polar coordinates
    
    Input: 
        some Sympy expression f to which the Laplacian is applied 
    
    Output: 
        A Sympy expression
    '''
    
    del2 = delop.dot(delop(f)).doit()
    
    return del2

# Trial function psi defined using SymPy
# A decaying Gaussian 
psi = exp(-alpha * R.r **2)

# We need to supply the Jacobian manually because we are no longer integrating 
# with SymPy.
# The Jacobian of the spherical integral is r**2 sin(theta)
# ignore sin(theta) because of symmetry 
jacobian = R.r ** 2 

#%% This part defines function related to the variational method

#The lambdify function converts Sympy expressions into something more suitable
#for numerical evaluation with SciPy

#The expectation value  <psi|H|psi>
expectation = simplify(conjugate(psi) * H(psi) ) 

expectation_lamb = lambdify((R.r, alpha), expectation * jacobian, 'scipy')

#the normalisation value <psi|psi> 
norm = conjugate(psi) * psi 

norm_lamb = lambdify((R.r, alpha), norm * jacobian, 'scipy')


# definining a function such that SciPy Optimisation can be used
def expectation_intgrl(a, stpt, endpt):
    ''' Formualtes an integral of the lambdified expectation function
   
    inputs:
        a (float): parameter
            
        stpt (float): starting point of the integral 
        
        endpt (float): endpoint of the integral
    
    output: 
        expectation_integral[0]: numerical result of the integral
    
    '''
    
    expectation_integral = quad(expectation_lamb, 0, endpt,args=(a))
    
    return expectation_integral[0]

def norm_intgrl(a, stpt, endpt):
    ''' Finds the square of the Normalisation constant of the trial function.
    
    Formualtes an integral of the lambdified normalisation function and uses
    the fact that the trial functions must be normalised to 1. 
   
    inputs:
        a (float): 
            
        stpt (float): starting point of the integral
        
        endpt (float): endpoint of the integral
    
    output: 
        N_squared: normalisation
    
    '''
    norm_intgrl = quad(norm_lamb, stpt, endpt, args = (a)) 
    
    N_squared = 1 / norm_intgrl[0]
    
    return N_squared


def var_intgrl(a, expectation_intgrl, norm_intgrl, stpt = 0, endpt = 1000):
    ''' Multiplies the normalisation with the expectation integral
    
    inputs: 
        expectation_intgrl (function of a): 
            
        norm_intgrl (function of a): N_squared, normalisation for trial function
        
        stpt (float): starting point of the integral (0 by default)
        
        endpt (float): endpoint of the integral (1000 effectively infinity)
        
    outputs: 
        var_intgrl (function of a): numerical result of the 
            variation integral
    '''
    
    var_intgrl = (norm_intgrl(a, stpt, endpt) * 
                  expectation_intgrl(a, stpt, endpt))
    
    return var_intgrl


#%% Execution of the minimisation of the variation integral

#supply the initial guess for the variation intergral
initial_guess = 0.1

#minimise the variation integral with respect to the parameter
alpha_op, energy_min = fmin(var_intgrl, initial_guess, (expectation_intgrl, norm_intgrl), 
              full_output = 1)[0:2]

print('\nIn atomic units, \nThe optimised parameter is %f. \nThe minimum energy is %f.' % (float(alpha_op),energy_min))
