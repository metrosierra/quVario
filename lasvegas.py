#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vegas example from the Basic Integrals section
of the Tutorial and Overview section of the
vegas documentation.
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import numpy as np
import sys
import time
from numba import jit, njit
from scipy.optimize import fmin, minimize
import matplotlib.pyplot as plt


#if sys.argv[1:]:
#    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
#else:
#    SHOW_GRID = True
    
SHOW_GRID = False

PLOT = True

@njit
def f(x, alpha):
    ''' Local energy term
    '''
    r1 = np.array([x[0], x[1], x[2]])
    r2 = np.array([x[3], x[4], x[5]])
    
    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)

    r1_hat = r1 / r1_len
    r2_hat = r2 / r2_len

    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

    EL =  (-4 + np.dot(r1_hat - r2_hat, r1 - r2 ) / (r12 * (1 + alpha*r12)**2)
            - 1/ (r12*(1 + alpha*r12)**3)
            - 1/ (4*(1 + alpha*r12)**4)
            + 1/ r12 )
    
    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
            * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
    
    return psisq * EL
  
@njit
def psisq(x, alpha):
    ''' Squared trial wavefunction
    '''
    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
    
    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)
    
    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
            * np.exp(r12 / (2 * (1+ alpha* r12)))) ** 2
    
    return psisq

def evalenergy(alpha): 
    @njit
    def expec(x):
        return f(x,alpha)
    @njit
    def norm(x):
        return psisq(x, alpha)

#    @njit
    def main():
        # seed the random number generator so results reproducible
        np.random.seed(1)
    
        # assign integration volume to integrator
        bound = 10 
        dims = 6
        
        # creates symmetric bounds specified by [-bound, bound] in dims dimensions
        symm_bounds= np.full((dims, 2), np.array([-bound, bound]))
        
        # simultaneously initialises expectation and normalisation integrals
        expinteg = vegas.Integrator(symm_bounds)
        norminteg = vegas.Integrator(symm_bounds)
        
        # adapt to the integrand; discard results
        expinteg(expec, nitn=5, neval=1000)
        norminteg(norm,nitn=5, neval=1000)
    
        # do the final integrals
        expresult = expinteg(expec, nitn=10, neval=100000)
        normresult = norminteg(norm, nitn=10, neval=100000)
        
### Code for printing the grid
### and diagnostics
#        print(expresult.summary())
#        print('expresult = %s    Q = %.2f' % (expresult, expresult.Q))
#        if SHOW_GRID:
#            expinteg.map.show_grid(20)
#        
#        print(normresult.summary())
#        print('normresult = %s    Q = %.2f' % (normresult, normresult.Q))
#        if SHOW_GRID:
#            norminteg.map.show_grid(20)
        
        # obtain numerical result
        
        ### Different expressions for plotting/ minimisation algorithm
        if PLOT:
            E = expresult.mean/normresult.mean
        else: 
            E = expresult[0].mean/ normresult[0].mean
        
        
        print('Energy is %f when alpha is %f' %(E, alpha))
#        print(expresult/normresult)
        return E
        
#    if __name__ == '__main__':
    start_time = time.time()
    E = main()
    print("--- Iteration time: %s seconds ---" % (time.time() - start_time))
    
    return E
    
start_time = time.time() 
### Run plotting function
PLOT = True
alpha = np.linspace(0.001, 0.3, 30)
energies = []
for i in alpha:
    energies.append(evalenergy(i))
plt.plot(alpha, energies)

alpha = np.linspace(0.3, 2, 10)
energies2 = []
for i in alpha:
    energies2.append(evalenergy(i))

plt.xlabel('alpha')
plt.ylabel('energy')
plt.grid()
plt.plot(alpha, energies2)

PLOT = False

result = fmin(evalenergy, .2, ftol = 0.01, xtol = 0.001, full_output=True)
plt.plot(result[0], result[1], 'ro', ms=5)

print("--- Total time: %s seconds ---" % (time.time() - start_time))

#%%

alpha = np.linspace(0.001, 0.3, 30)
plt.plot(alpha, energies)

plt.xlabel('alpha')
plt.ylabel('energy')
plt.grid()
alpha = np.linspace(0.3, 2, 10, color = 'blue')
plt.plot(alpha, energies2)

plt.plot(result[0], result[1], 'ro', ms=5)