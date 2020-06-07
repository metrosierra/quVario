#!/usr/bin/env python3

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
from numba import jit, njit

@njit
def metropolis_hastings(pfunc, iter, alpha, dims):

    initial_pt = 2*np.random.rand(dims) - 1.0
    step_scale = 0.4

    samples = np.zeros((iter, dims))
    reject_ratio = 0.
    test = []
    # now sample iter number of points
    for i in range(iter):
        # we propose a new point using a Symmetric transition distribution function: a Gaussian
        walk = step_scale * np.random.rand(6) - step_scale/2
        proposed_pt = initial_pt + walk

        p = pfunc(proposed_pt, alpha) / pfunc(initial_pt, alpha)
        # print(p)
        test.append(p)
        if p >= 1.:
            initial_pt = proposed_pt
        if np.random.rand() <= p:
            initial_pt = proposed_pt
        else:
            reject_ratio += 1./iter

        samples[i] = initial_pt
    return samples, reject_ratio, test


@njit
def integrator_mcmc(pfunc, qfunc, sample_iter, avg_iter, alpha, dims):

    therm = 0
    vals = np.zeros(avg_iter)
    val_errors = 0.
    alpha = np.array([alpha])

    test = []
    for i in range(avg_iter):
        mc_samples, rejects, p = metropolis_hastings(pfunc, sample_iter, alpha, dims)
        sums = 0.
        # obtain arithmetic average of sampled Q values
        for array in mc_samples[therm:]:
            sums += qfunc(array, alpha)
            test.append(qfunc(array, alpha))
        vals[i] = (sums/(sample_iter - therm))

    # also calculate the variance
    vals_squared = np.sum(vals**2)
    vals_avg = np.sum(vals) /avg_iter
    variance = vals_squared/avg_iter - (vals_avg) ** 2
    std_error = np.sqrt(variance/avg_iter)


    print('Iteration cycle complete, result = ', vals_avg, 'error = ', std_error, 'rejects = ', rejects)
    return vals_avg, std_error, test


@njit
def mcmc_q(x, alpha):
    ''' this is the Q part of the integrand function for mcmc
    inputs: x(array), denoting iter number of sample points, given by
    '''


    ### helium local energy
    r1 = x[0:3]
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
