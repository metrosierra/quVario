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
import time
from numba import jit, njit
from scipy.optimize import fmin
import matplotlib.pyplot as plt


#if sys.argv[1:]:
#    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
#else:
#    SHOW_GRID = True
    
SHOW_GRID = False
OPTIM = True

#@njit
#def f(x, alpha):
#    ''' Expectation value function, used as numerator for variational integral
#    '''
#    r1 = np.array([x[0], x[1], x[2]])
#    r2 = np.array([x[3], x[4], x[5]])
#    
#    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
#    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
#
#    r1_hat = r1 / r1_len
#    r2_hat = r2 / r2_len
#
#    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)
#
#    EL =  (-4 + np.dot(r1_hat - r2_hat, r1 - r2 ) / (r12 * (1 + alpha*r12)**2)
#            - 1/ (r12*(1 + alpha*r12)**3)
#            - 1/ (4*(1 + alpha*r12)**4)
#            + 1/ r12 )
#    
#    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
#            * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
#    
#    return psisq * EL
#
#@njit
#def psisq(x, alpha):
#    ''' Squared trial wavefunction, used as denominator for variational integral
#    '''
#    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
#    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
#    
#    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)
#    
#    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
#            * np.exp(r12 / (2 * (1+ alpha* r12)))) ** 2
#    
#    return psisq


def evalenergy(alpha): 
#    @njit
#    def expec(x):
#        return f(x, alpha)
#    @njit
#    def norm(x):
#        return psisq(x, alpha)
    
    @vegas.batchintegrand
    def expec(x):
        ''' Expectation value function, used as numerator for variational integral
        '''
#        x = np.reshape(x, (1,-1))
        
        r1 = np.array(x[:,0:3])
        r2 = np.array(x[:,3:])
        
        r1_len = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
        r2_len = np.sqrt(x[:,3]**2 + x[:,4]**2 + x[:,5]**2)
        
        r1_hat = r1 / np.reshape(r1_len, (-1,1))
        r2_hat = r2 / np.reshape(r2_len, (-1,1))
    
        r12 = np.sqrt((x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
    
        EL =  (-4 + np.sum((r1_hat-r2_hat)*(r1-r2), 1) / (r12 * (1 + alpha*r12)**2)
                - 1/ (r12*(1 + alpha*r12)**3)
                - 1/ (4*(1 + alpha*r12)**4)
                + 1/ r12 )
        
        psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
                * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2   
        return psisq * EL

    @vegas.batchintegrand
    def norm(x):
        ''' Squared trial wavefunction, used as denominator for variational integral
        '''
#        x = np.reshape(x, (1,-1))
        
        r1_len = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
        r2_len = np.sqrt(x[:,3]**2 + x[:,4]**2 + x[:,5]**2)
        
        r12 = np.sqrt((x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
        
        psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
                * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
        return psisq

    def main():
        # seed the random number generator so results reproducible
#        np.random.seed(1)
        start_time = time.time()
        
        # assign integration volume to integrator
        bound = 8 
        dims = 6
        # creates symmetric bounds specified by [-bound, bound] in dims dimensions
        symm_bounds = dims * [[-bound,bound]]
        
        # simultaneously initialises expectation and normalisation integrals
        expinteg = vegas.Integrator(symm_bounds)
        norminteg = vegas.Integrator(symm_bounds)
        
        # adapt to the integrands; discard results
        expinteg(expec, nitn=5, neval=1000)
        norminteg(norm,nitn=5, neval=1000)
        # do the final integrals
        expresult = expinteg(expec, nitn=10, neval=1000000)
        normresult = norminteg(norm, nitn=10, neval=1000000)
        
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
        
        ### obtain numerical result
        ### Different expressions for plotting/ minimisation algorithm
        if not OPTIM:
            E = expresult.mean/normresult.mean
        else: 
            E = expresult[0].mean/ normresult[0].mean
        #print('Energy is %f when alpha is %f' %(E, alpha))
        #print("--- Iteration time: %s seconds ---" % (time.time() - start_time))
        return E
    E = main()
    return E
    

#%% This is for one-parameter plotting
#### Run plotting function, plot alpha against energies
start_time = time.time()
OPTIM = False

plt.figure(figsize=(16,10))
#high resolution bit closer to the minimum
#alpha0 = np.linspace(0.001, 0.1, 5)
#energies0 = []
#for i in alpha0:
#    energies0.append(evalenergy(i))
  
print('Plotting function initialised!')
energies = []
alpha = np.linspace(0.1, 0.2, 20)
for i in alpha:
    energy_samples = []
    for j in range(1):
        energy_samples.append(evalenergy(i))
    avg = np.average(energy_samples)
    print('Averaged energy for %f is %f' %(i, avg))
    energies.append(avg)
plt.plot(alpha, energies, color='dodgerblue')
print("--- Total time: %s seconds ---" % (time.time() - start_time))

plt.savefig('Vegas')

#%%
##low resolution bit
#alpha = np.linspace(0.25, 2, 10)
#energies2 = []
#for i in alpha:
#    energies2.append(evalenergy(i))
#plt.plot(alpha, energies2, color = 'dodgerblue')
#
#
#
#### Use optimisation to obtain the minimum
#### and plot a point
##OPTIM = False
##result = fmin(evalenergy, 0.2, ftol = 0.01, xtol = 0.001, full_output=True)
##plt.plot(result[0], result[1], 'ro', ms=5)
#
#plt.xlabel('alpha')
#plt.ylabel('energy')
#plt.grid()
#plt.show()
#print("--- Total time: %s seconds ---" % (time.time() - start_time))

#%% Extension to multiple parameters 
### We plot a histogram
OPTIM = False
start_time = time.time()
result = fmin(evalenergy, 0.15, ftol = 0.01, xtol = 0.001, full_output=True)
#e = evalenergy(0.5)
#print(e)
print("--- Total time: %s seconds ---" % (time.time() - start_time))