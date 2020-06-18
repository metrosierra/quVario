#!/usr/bin/env python3
#%%
import math
import sys
import os
import time
from datetime import datetime



import numpy as np
import scipy as sp
import scipy.constants as sc

#### For future monte carlo integration work hopefully
import mcint
import random
import numba as nb
from numba import jit, njit, prange

'''
https://jellis18.github.io/post/2018-01-02-mcmc-part1/
^ this is a really informative website about calibrating MCMC algorithms based on acceptance reject ratio
'''




#%%

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


    print('Iteration cycle complete, result = ', vals_avg, 'error = ', std_error, 'rejects = ', rejects)
    return vals_avg, std_error, rejects, test, p


@njit
def mcmc_q(x, alpha):
    ''' this is the Q part of the integrand function for mcmc
    inputs: x(array), denoting iter number of sample points, given by
    '''


    ### helium local energy
    r1 = x[0:2]
    r2 = x[3:]
    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)


    r1_hat = r1 / r1_len
    r2_hat = r2 / r2_len

    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

    return ((-4 + np.dot(r1_hat - r2_hat, r1 - r2 ) / (r12 * (1+alpha*r12)**2)
            - 1/ (r12*(1+alpha*r12)**3)
            - 1/ (4*(1+alpha*r12)**4)
            + 1/ r12 ))[0]

@njit
def mcmc_p(x, alpha):
    '''
    this is the integrand function for mcmc

    '''
    ### helium wavefunction squared
    r1 = x[0:3]
    r2 = x[3:]
    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

    return ((np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
            * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2)[0]
