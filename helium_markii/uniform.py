#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:24:56 2020

Basic Uniform Monte Carlo Integrator
Applied to Variational Calculations
"""

import time
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(parallel = True)
def Uint(integrand, sampler, bounds, measure, n, alpha):
    ''' Obtains integral result
    Inputs:
        integrand: function
        sampler: uniform sampler, function
        bounds: integral bounds, array
        n: number of samples per dimension
    '''

    # this takes n samples from the integrand and stores it in values
    values = np.zeros(n)

    for i in prange(n):
        sample = sampler(bounds)
        val = integrand(sample, alpha)
        values[i] = val

    # this is the value of the sampled integrand values
    result = measure * np.sum(values) / n
    return result

@njit
def Uint_iter(integrand, sampler, bounds, measure, n, iters, alpha):
        ''' returns array of integral results
        '''
        # obtain iters integral results
        results = np.zeros(iters)
        for i in range(iters):
            results[i] = Uint(integrand, sampler, bounds, measure, n, alpha)
        return results

@njit
def get_measure(bounds):
    ''' obtains the volume of the dims dimensional hypercube.
        Used for multiplying the expected value.

        inputs:
            bounds: array of size dims, indicating the dims bounds of the definite
                integral
        outputs:
            measure: float
    '''
    measure = 1.
    dims = len(bounds)
    for i in range(dims):
        b = bounds[i]
        dimlength = float(b[1] - b[0])
        measure *= dimlength
    return measure

@njit
def sampler(bounds):
    ''' returns a uniformly distributed random array of size dims

    inputs:
        bounds is a 2D array, specifying the integration range of each dim
    '''
    dims = len(bounds)
    samples = np.zeros(dims)
    for i in range(dims):
        b = bounds[i]
        samples[i] = np.random.uniform(b[0], b[1])
    return samples

@njit
def psiHpsi(x, alpha):
    ''' Local energy term
    '''
    r1 = x[0:3]
    r2 = x[3:]

    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)


    r1_hat = r1 / r1_len
    r2_hat = r2 / r2_len

    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

    EL = ((-4 + np.dot(r1_hat - r2_hat, r1 - r2 ) / (r12 * (1+alpha[0]*r12)**2)
            - 1/ (r12*(1+alpha[0]*r12)**3)
            - 1/ (4*(1+alpha[0]*r12)**4)
            + 1/ r12 ))

    psisq = ((np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
            * np.exp(r12 / (2 * (1+ alpha[0]* r12)))) ** 2)

    return psisq* EL

@njit
def psisq(x, alpha):
    ''' Squared trial wavefunction
    '''
    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)

    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
            * np.exp(r12 / (2 * (1+ alpha[0]* r12)))) ** 2

    return psisq

# I just can't jit this part for some odd reason? All functions that it calls are
    # jitted though
@njit
def evalenergy(alpha):
    #initialise settings
    domain = 2.
    dims = 6

    bounds = []
    for i in range(dims):
        bounds.append([-domain, domain])
    bounds = np.array(bounds)
    n = 100000
    iters = 30
    measure  = get_measure(bounds)

    expresults = Uint_iter(psiHpsi, sampler, bounds, measure, n, iters, alpha)
    normresults = Uint_iter(psisq, sampler, bounds, measure, n, iters, alpha)

    #obtain average and variance
    vals = expresults/normresults
    avg = np.sum(vals) / iters
    vals_squared = np.sum(vals**2)
    var = (vals_squared/ iters - avg **2)
    std = np.sqrt(var)

    E = avg
    print(E, std)
    # print('When alpha is {}, the energy is {} with std {}' .format(alpha, E, std))
    return E


### Minimisation algorithm

start_time = time.time()
fmin(evalenergy, 0.1, full_output = 1, ftol = 1)
duration = (time.time() - start_time)
print('Time taken:', duration)

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
