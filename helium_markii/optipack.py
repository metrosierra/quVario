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
import statistics

### using sympy methods for symbolic integration. Potentially more precise and convenient (let's not numerically estimate yet)
import numpy as np
import scipy as sp
import scipy.constants as sc
from scipy import optimize, integrate

import sympy as sy
from sympy import conjugate, simplify, lambdify, sqrt
from sympy import *
from IPython.display import display

import vegas


import matplotlib.pyplot as plt

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
def integrator_mcmc(pfunc, qfunc, sample_iter, walkers, alpha, dims, verbose = True):

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

    if verbose:
        print('Iteration cycle complete, result = ', vals_avg, 'error = ', std_error, 'rejects = ', rejects, 'alpha current = ', alpha)
    return vals_avg, std_error, rejects, test, p


#%%%%%%%%%%%%%%%%%%%%%%%

@njit(parallel = True)
def Uint(integrand, bounds, n, alpha):


    # this takes n samples from the integrand and stores it in values
    values = 0
    for x in prange(n):
        sample = np.zeros(len(bounds))
        for i in range(len(bounds)):
            sample[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        val = integrand(sample, alpha)
        values += val

    # this is the value of the sampled integrand values
    result = values / n
    return result


@njit
def Ueval(normalisation, expectation, n, iters, alpha, dimensions):
    #initialise settings
    domain = 2.
    dims = dimensions
    #origin centered symmetrical volume
    bounds = []
    for i in range(dims):
        bounds.append([-domain, domain])
    bounds = np.array(bounds)

    measure = 1.
    for i in range(dims):
        dimlength = float(bounds[i][1] - bounds[i][0])
        measure *= dimlength

    results = np.zeros(iters)
    normresults = np.zeros(iters)
    for i in range(iters):
        results[i] = Uint(expectation, bounds, n, alpha) * measure
        normresults[i] = Uint(normalisation, bounds, n, alpha) * measure

    #obtain average and variance
    vals = results / normresults
    avg = np.sum(vals) / iters
    vals_squared = np.sum(vals**2)
    var = (vals_squared/ iters - avg **2)
    std = np.sqrt(var)

    E = avg
    print(E, std )
    # print('When alpha is {}, the energy is {} with std {}' .format(alpha, E, std))
    return E, std

#%%%%%%%%%%%%%%%%%%%%%%%

class LasVegas():

    def __init__(self):
        print('LasVegas up for business!')

    def vegas_int(self, expec, norm, evals, iter, dimensions, volumespan):

        self.final_results = {}

        start_time = time.time()

        # assign integration volume to integrator
        bound = volumespan
        dims = dimensions
        # creates symmetric bounds specified by [-bound, bound] in dims dimensions
        symm_bounds = dims * [[-bound,bound]]

        # simultaneously initialises expectation and normalisation integrals
        expinteg = vegas.Integrator(symm_bounds)
        norminteg = vegas.Integrator(symm_bounds)

        # adapt to the integrands; discard results
        expinteg(expec, nitn = 5, neval = 1000)
        norminteg(norm, nitn = 5, neval = 1000)
        # do the final integrals
        expresult = expinteg(expec, nitn = iter, neval = evals)
        normresult = norminteg(norm, nitn = iter, neval = evals)


        E = expresult.mean/normresult.mean
        print('Energy is {} when alpha is'.format(E), ' with sdev = ', [expresult.sdev, normresult.sdev])
        print("--- Iteration time: %s seconds ---" % (time.time() - start_time))

        self.final_results['energy'] = E
        self.final_results['dev'] = [expresult.sdev, normresult.sdev]
        self.final_results['pvalue'] = [expresult.Q, normresult.Q]

        return self.final_results


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('\n\nLasVegas object self-destructing\n\n')


#%%%%%%%%%%%%%%%%%%%%%%%

class MiniMiss():

    def __init__(self):
        print('MiniMiss optimisation machine initialised and ready!\n')

    def neldermead(self, func, guess, ftol, bounds = [(0., 3.),(0., 3.),(0., 0.3) ]):
        starttime = time.time()

        temp = optimize.fmin(func, guess, full_output = 1, ftol = ftol)
        # temp = optimize.differential_evolution(func, bounds)
        # temp = optimize.minimize(func, guess, method = 'BFGS', tol = 0.01)

        endtime = time.time()
        elapsedtime = endtime - starttime
        now = datetime.now()
        date_time = now.strftime('%d/%m/%Y %H:%M:%S')
        # just returns datetime of attempt, elapsed time, optimised parameter, optimised value, number of iterations
        return [date_time, elapsedtime, guess, temp]

    def gradient(self, func, guess, tolerance, convergence = 'fuzzy', args = None):

        print('\nGradient descent initiated, type of convergence selected as ' + convergence +'\n')
        cycle_count = 0
        position0 = guess
        ep = 0.00000001
        step = 0.2
        step_ratio = 0.5
        epsilons = np.identity(len(guess))*ep
        delta1 = 10.

        point_collection = []

        satisfied = False

        while satisfied == False:
            cycle_count+=1
            print('Cycle', cycle_count)
            value1 = func(position0)

            vector = np.zeros(len(guess))
            for i in range(len(guess)):
                vector[i] = (func(position0 + epsilons[i]) - func(position0))/ep

            vectornorm = vector/(np.linalg.norm(vector))
            print(vectornorm)
            position0 += -vectornorm * step
            value2 = func(position0)
            delta1 = value2 - value1

            positionprime = position0 + vectornorm*(step*step_ratio)
            value3 = func(positionprime)
            delta2 = value3 - value1
            if delta2 < delta1:
                print('shrink!')
                position0 = positionprime
                step = step*step_ratio
                delta1 = delta2
                value2 = value3

            point_collection.append(value2)
            if convergence == 'strict':
                satisfied = abs(delta1) < tolerance
                finalvalue = value2

            elif convergence == 'fuzzy':
                if len(point_collection) >= 5:
                    data_set = point_collection[-5:]
                    print('std', statistics.pstdev(data_set))
                    print('mean', statistics.mean(data_set))
                    satisfied = abs(statistics.pstdev(data_set)/statistics.mean(data_set)) < tolerance
                    finalvalue = statistics.mean(data_set)

        print('Convergence sucess!')
        return finalvalue, position0


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('\n\nMiniMiss object self-destructing\n\n')
