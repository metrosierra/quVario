#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario


### This script provides the montypython object AND minimiss object that handles any calls to monte carlo integration functions and minimisation functions of our design. It interfaces with the python packages installed for basic functionalities (ie mcint)
### optipack should serve helium_markii.py which is the higher level script for direct user interface


### as of 27 may 2020 montypython uses mcint, and minimiss uses scipy fmin functions as main work horses.

### montypython removed, replaced with simple functions because this is the only sane way to be able to numba jit them

import math
import sys
import os
import time
from datetime import datetime


### using sympy methods for symbolic integration. Potentially more precise and convenient (let's not numerically estimate yet)
import numpy as np
import scipy as sp
import scipy.constants as sc
import sympy as sy
from sympy import conjugate, simplify, lambdify, sqrt
from sympy import *
from IPython.display import display
from scipy import optimize, integrate

import matplotlib.pyplot as plt

#### For future monte carlo integration work hopefully
import random
from numba import jit, njit, prange


@njit
def metropolis_hastings(pfunc, iter, alpha, dims):

    # we make steps based on EACH electron, the number of which we calculate from dims/3
    # 'scale' of the problem. this is arbitrary
    s = 3.
    # we discard some initial steps to allow the walkers to reach the distribution
    equi_threshold = 0
    initial_matrix = np.zeros((int(dims/3), 3))
    reject_ratio = 0.
    therm = 1
    test = []
    samples = []

    for i in range(int(dims/3)):
        initial_matrix[i] = 2.*s*np.random.rand(3) - s


    # now sample iter number of points
    for i in range(iter):

        # choose which electron to take for a walk:
        e_index = np.random.randint(0, dims/3)
        trial_matrix = initial_matrix.copy()
        trial_matrix[e_index] += (2.*s*np.random.rand(3) - s)/(dims/3)

        # trial_matrix[e_index] = hi
        proposed_pt = np.reshape(trial_matrix, (1, dims))[0]
        initial_pt = np.reshape(initial_matrix, (1, dims))[0]

        # print(initial_pt)
        p = pfunc(proposed_pt, alpha) / pfunc(initial_pt, alpha)
        if p > np.random.rand():
            initial_matrix = trial_matrix.copy()
            # print(initial_matrix == trial_matrix)

        else:
            reject_ratio += 1./iter

        if i > equi_threshold:
            if (i-therm)%therm == 0:
                test.append(np.reshape(initial_matrix, (1, dims))[0][3])
                samples.append(np.reshape(initial_matrix, (1, dims))[0])

    return samples, reject_ratio, test


@njit
def integrator_mcmc(pfunc, qfunc, sample_iter, walkers, alpha, dims):

    therm = 0
    vals = np.zeros(walkers)
    val_errors = 0.
    test = []
    for i in range(walkers):
        mc_samples, rejects, p  = metropolis_hastings(pfunc, sample_iter, alpha, dims)
        sums = 0.

        # obtain arithmetic average of sampled Q values
        for array in mc_samples[therm:]:
            sums += qfunc(array, alpha)
            test.append(qfunc(array, alpha))
        vals[i] = (sums/(sample_iter - therm))
    # also calculate the variance
    vals_squared = np.sum(vals**2)
    vals_avg = np.sum(vals) /walkers
    variance = vals_squared/walkers - (vals_avg) ** 2
    std_error = np.sqrt(variance/walkers)


    print('Iteration cycle complete, result = ', vals_avg, 'error = ', std_error, 'rejects = ', rejects, 'alpha current = ', alpha)
    return vals_avg, std_error, rejects, test, p


#%%%%%%%%%%%

@njit(parallel = True)
def Uint(integrand, samples, bounds, n, alpha):

    # this takes n samples from the integrand and stores it in values
    values = 0
    for i in prange(n):
        sample = np.random.uniform(bounds[0], bounds[1], len(bounds))
        val = integrand(sample, alpha)
        values += val

    # this is the value of the sampled integrand values
    result = values / n
    return result


@njit
def evalenergy(integrand, alpha, n = 100000, iters = 30):
    #initialise settings
    domain = 2.
    dims = 6

    #origin centered symmetrical volume
    bounds = []
    for i in range(dims):
        bounds.append([-domain, domain])
    bounds = np.array(bounds)

    measure = 1.
    for i in range(dims):
        dimlength = float(bounds[1] - bounds[0])
        measure *= dimlength

    results = np.zeros(iters)
    normresults = np.zeros(iters)
    for i in range(iters):

        results[i] = Uint(integrand, bounds, n, alpha) * measure
        normresults[i] = Uint(integrand, bounds, n, alpha) * measure

    #obtain average and variance
    vals = expresults/normresults
    avg = np.sum(vals) / iters
    vals_squared = np.sum(vals**2)
    var = (vals_squared/ iters - avg **2)
    std = np.sqrt(var)

    E = avg
    print(E, std)
    # print('When alpha is {}, the energy is {} with std {}' .format(alpha, E, std))
    return E, std

#%%%%%%%%%%%%%%%%%%%%%%%


class MiniMiss():

    def __init__(self):
        print('MiniMiss optimisation machine initialised and ready!\n')

    def minimise(self, func, guess, ftol):
        starttime = time.time()

        temp = optimize.fmin(func, guess, full_output = 1, ftol = ftol)

        endtime = time.time()
        elapsedtime = endtime - starttime
        now = datetime.now()
        date_time = now.strftime('%d/%m/%Y %H:%M:%S')
        # just returns datetime of attempt, elapsed time, optimised parameter, optimised value, number of iterations
        return [date_time, elapsedtime, guess, temp[0], temp[1], temp[3]]

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('\n\nMiniMiss object self-destructing\n\n')
        
#%%
#setup = 'import numpy as np'
#code = 'np.random.uniform((-5,5), 6)'
#print(timeit.timeit(setup = setup,stmt = code, number = N)/N)
#N= 100000
#
#setup = 'import random; import numpy as np; x = np.zeros(6)'
#code = 'for i in range(6): x[i] = random.uniform(-5,5)'
#print(timeit.timeit(setup = setup,stmt = code, number = N)/N)
#
#setup = 'import random; x = []'
#code = 'for i in range(6): x.append(random.uniform(-5,5))'
#print(timeit.timeit(setup = setup,stmt = code, number = N)/N)
#
#setup = 'import random; '
#code = 'x = [random.uniform(-5,5),random.uniform(-5,5),random.uniform(-5,5),random.uniform(-5,5),random.uniform(-5,5),random.uniform(-5,5)]'
#print(timeit.timeit(setup = setup,stmt = code, number = N)/N)
