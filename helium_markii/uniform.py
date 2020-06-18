#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:24:56 2020

@author: kenton
"""

import time

import numpy as np
import scipy as sp

from scipy.optimize import fmin

import matplotlib.pyplot as plt

from numba import jit, njit

@njit
def Uint(integrand, sampler, bounds, n, *alpha):
    ''' Obtains integral result and its variance
    Inputs:
        integrand: function
        sampler: uniform sampler, function
        bounds: integral bounds, array
        n: number of samples per dimension
    Outputs:
    '''

    samples = sampler(bounds, n)

    measure = get_measure(bounds)

    vals = integrand(samples, alpha)
    vals_sq = np.sum(vals**2)

    # this is the value and variance of the sampled integrand values
    result = measure * np.sum(vals) / n
    var = (vals_sq / n - ((result) ** 2))

    return np.array([result, var])

@njit
def iter_Uint(integrand, sampler, bounds, n, iters, *alpha):
        '''
        '''
        result = np.zeros(iters)

#        times = []

        for i in range(iters):
#            start_time = time.time()

            x = Uint(integrand, sampler, bounds, n, alpha)


#            duration = (time.time() - start_time)

            result[i] = x[0]
#            times.append(duration)

        avg = np.sum(result) / iters
#        print(avg)
        result_squared = result ** 2

        var = (np.sum(result_squared) - avg **2) / (iters - 1)
#        print(var)
#        return np.array([avg, var, times])
        return np.array([avg, var, result])

@njit
def get_energy(alpha, psiHpsi, psisq, integrand, sampler, bounds, n, iters):

    exp = iter_Uint(psiHpsi, sampler, bounds, n, iters, alpha)[0]
    norm = iter_Uint(psisq, sampler, bounds, n, iters, alpha)[0]

    return exp/ norm

@njit
def get_measure(bounds):
    ''' obtains n dimensional 'measure' for the integral, which effectively
        is the volume in n dimensional space of the integral bounds.
        Used for multiplying the expected value.

        inputs:
            bounds: list of size n, indicating the n bounds of the definite
                integral
        outputs:
            measure: float
    '''
    measure = 1.

#    for i in bounds:
#        dimlength = float(i[1] - i[0])
#        measure *= dimlength

    for i in range(len(bounds)):
        dimlength = float(bounds[i][1] - bounds[i][0])
        measure *= dimlength

    return measure

@njit
def sampler(bounds, iters):
    ''' generates a tuple of n input values from a random uniform distribution
        e.g. for three dimensions, outputs tuple = (x,y,z) where x,y,z are
        floats from a uniorm distribution
    '''
    dims = len(bounds)
    samples = np.zeros((dims, iters))

#    for i,b in enumerate(bounds):
#        dim_sample = np.random.uniform(b[0], b[1], iters)
#        samples[i,:] = dim_sample

    for i in range (len(bounds)):
        dim_sample = np.random.uniform(bounds[i][0], bounds[i][1], iters)
        samples[i,:] = dim_sample
    return samples

@njit
def integrand(x):
    ''' this is the integrand function
    '''
    return  np.exp(-(x[0]) ** 2) * x[0]**2 

### Helium atom variational method

@njit
def psiHpsi(x, alpha):
    ''' Local energy term
    '''
#    r1 = np.array([x[0], x[1], x[2]])
#    r2 = np.array([x[3], x[4], x[5]])

    r1 = x[0:2]
    r2 = x[3:5]

    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)

    r1_hat = r1 / r1_len
    r2_hat = r2 / r2_len
#    print(x[0])

    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

#    try:
#        EL =  (-4 + np.dot(r1_hat - r2_hat, r1 - r2 ) / (r12 * (1 + alpha*r12)**2)
#                - 1/ (r12*(1 + alpha*r12)**3)
#                - 1/ (4*(1 + alpha*r12)**4)
#                + 1/ r12 )
#    except:
#        EL =  (-4 + np.sum((r1_hat - r2_hat) * (r1 - r2), 0) / (r12 * (1 + alpha*r12)**2)
#                - 1/ (r12*(1 + alpha*r12)**3)
#                - 1/ (4*(1 + alpha*r12)**4)
#                + 1/ r12 )

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

#%%
alpha = .1


bound = 5
dims = 6
# creates symmetric bounds specified by [-bound, bound] in dims dimensions
symm_bounds= np.full((dims, 2), np.array([-bound, bound]))

n = int(100000)
iters = 10

#fmin(get_energy, 0.2)
energies = []
for i in range(500):
    e = get_energy(alpha, psiHpsi, psisq, integrand, sampler, symm_bounds, n, iters)
    if i % 25 == 0:
        print(i,e)
    energies.append(e)
plt.hist(energies,50)

#%%
#high resolution bit
alpha = np.linspace(0.001, 0.3, 30)
energies = []
n = int(100000)
iters = 10

for i in alpha:
    energies.append(get_energy(i, psiHpsi, psisq, integrand, sampler, symm_bounds, n, iters))
plt.plot(alpha, energies, color='dodgerblue')

args = (psiHpsi, psisq, integrand, sampler, symm_bounds, n, iters)

#fmin(get_energy, 0.1,  (psiHpsi, psisq, integrand, sampler, symm_bounds, n, iters),full_output = 1, ftol = 0.1,)


#%%%%%%%%%%%%%%%%%%%%%%%%
### Graph Plotting stuff
def plot_iter_graphs(itegrand, bounds, iter_range, lit_val, plotval=1, plotnormval=1, plottime=1):
    ''' This function is used for plotting graphs
    '''

    results = []
    variances = []
    times = []

    def get_lit(x):
        return float(lit_val) + 0*x

    for iters in iter_range:
        iters = int(iters)
        start_time = time.time()
        x = Uint(integrand, sampler, bounds, iters)
        duration = (time.time() - start_time)
        results.append(x[0])
        variances.append(x[1])
        times.append(duration)

    if plotval:
        plt.plot(np.log10(iter_range), results)
        plt.xscale = 'log'
        plt.ylabel('Value')
        plt.xlabel('Iterations log scale')
        plt.grid()
        plt.plot()
        plt.show()

    if plotnormval:
        xrange = np.log10(iter_range)
        plt.plot(xrange, results/get_lit(np.log10(iter_range)))
        plt.xscale = 'log'
        plt.ylabel('Normalised Value')
        plt.xlabel('Iterations log scale')
        plt.grid()
        plt.plot()
        plt.show()

    if plottime:
        plt.plot(np.log10(iter_range), times)
        plt.xscale = 'log'
        plt.ylabel('Time taken (s)')
        plt.xlabel('Iterations log scale')
        plt.grid()
        plt.show()
    pass

#%%
#bounds = np.array([[-4,4]])
#iter_range = np.logspace(5,7,20)
#lit_val = 0.8862925
#
#plot_iter_graphs(itegrand, bounds, iter_range, lit_val, 0,0,0)
