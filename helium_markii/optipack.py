#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario


### This script provides the montypython object AND minimiss object that handles any calls to monte carlo integration functions and minimisation functions of our design. It interfaces with the python packages installed for basic functionalities (ie mcint)
### optipack should serve helium_markii.py which is the higher level script for direct user interface


### as of 27 may 2020 montypython uses mcint, and minimiss uses scipy fmin functions as main work horses.


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

#### For future monte carlo integration work hopefully
import mcint
import random
from numba import jit, njit

class MontyPython():

    ### initialisation protocol for monteCarlo object
    def __init__(self):
        self.hbar = sc.hbar
        self.mass_e = sc.electron_mass
        self.q = sc.elementary_charge
        self.pi = sc.pi
        self.perm = sc.epsilon_0
        self.k = (self.q**2)/(4*self.pi*self.perm)
        self.n = 1000

        self.bounds = [(-1,1),(-1,1)]
        self.dims = 6
        sy.init_printing()
        print('\n\nMontyPython integrator initialised and ready! Dimensions = {}\n\n'.format(self.dims))


#        print(self.integrator_basic())

#        print(self.sampler(self.bounds))

        print(self.integrator_mcmc(self.integrand_mcmc_p, self.integrand_mcmc_q, np.array([0]),
                                    sample_iter = 10000, avg_iter = 10))

### These helper functions concern the basic Monte Carlo Integrator
    def get_measure(self, bounds):
        ''' obtains n dimensional 'measure' for the integral, which effectively
            is the volume in n dimensional space of the integral bounds. used for
            multiplying the expected value.

            inputs:
                bounds: list of size n, indicating the n bounds of the definite
                    integral

            outputs:
                measure: float

        '''
        measure = 1

        for i in bounds:
            endpt = i[1]
            stpt = i[0]
            dimlength = endpt - stpt

            measure *= dimlength

        return measure


    def integrator_basic(self):
        ''' using mcint package to determine the integral via uniform distribution
        sampling over an interval

        '''
        # measure is the 'volume' over the region you are integrating over

        result,error = mcint.integrate(self.integrand,self.sampler(self.bounds),
                                        self.get_measure(self.bounds), self.n)

        return result, error


    def sampler(self, bounds):
        ''' generates a tuple of n input values from a random uniform distribution
            e.g. for three dimensions, outputs tuple = (x,y,z) where x,y,z are
            floats from a uniorm distribution

            inputs:
                bounds

            outputs:
                sample (tuple)
        '''
        while True:

            sample = ()

            for i in bounds:
                endpt = i[1]
                stpt = i[0]
                dimsample = random.uniform(stpt,endpt)

                x = list(sample)
                x.append(dimsample)
                sample = tuple(x)

            yield sample

    def integrand(self, x):
        ''' this is the integrand function

        inputs: x (array), where x[0] denotes the first variable

        '''

        return sp.exp(-(x[0]) ** 2)


### These helper functions concern the Metropolis algorithm implementation of the integral
    # @jit
    def integrator_mcmc(self, pfunc, qfunc, initial_point, sample_iter = 100000, avg_iter = 10):
        ''' fancy metropolis hastings integrator! where pfunc and qfunc give the
        function f you want to integrate over.

        inputs:
            pfunc: effective probability density function
            qfunc: some function where pfunc*qfunc = f

        outputs:
            result: result of integral
            error: error of integral

        '''
        vals = np.zeros(avg_iter)
        val_errors = np.zeros(avg_iter)

        for i in range(avg_iter):
            mc_samples = self.metropolis_hastings(pfunc, sample_iter, initial_pt = initial_point)
            mc_samples = np.array(mc_samples)

            func_vals = []
            
            # obtain arithmetic average of sampled Q values
            for array in mc_samples:
                func_vals.append(qfunc(array))

            sums = np.sum(func_vals)

            vals[i] = (sums/sample_iter)
        # also calculate the variance
        vals_squared = np.sum(vals**2)

        vals_avg = np.sum(vals)/ avg_iter

        for i in range (avg_iter):

            val_errors[i] = (np.sqrt(vals_squared/ avg_iter - (vals_avg) ** 2))

        result = vals_avg
        error = np.sum(val_errors)/ np.sqrt(avg_iter)

        return result, error

    # @jit
    def metropolis_hastings(self, pfunc, iter, initial_pt):
        ''' Metropolis algorithm for sampling from a function p

            inputs:
                pfunc: effective probability density function part of 
                    function f we want to integrate over
                iter: number of random walk iterations
                initial_pt: starting point
                dims: dimensions of the sample

            outputs:

        '''
        dims = np.size(initial_pt)
        
        # simple sanity check
        if len(initial_pt) != dims:
            raise Exception('Error with inputs')

        # note: initial point chosen in input
        samples = np.zeros((iter, dims))

        # now sample iter number of points
        for i in range(iter):

            # we propose a new point using a Symmetric transition distribution
            #function: a Gaussian
            proposed_pt = np.array(initial_pt) + np.random.normal(size=dims)

            # if the ratio is greater than one, accepept the proposal
            # else, accept with probability of the ratio
            if np.random.rand() < pfunc(proposed_pt) / pfunc(initial_pt):
                initial_pt = proposed_pt

            samples[i] = np.array(initial_pt)

        return samples


    def integrand_mcmc_q(self, x):
        ''' this is the Q part of the integrand function for mcmc

        inputs: x(array), denoting iter number of sample points, given by
            metropolis hastings

        output:
            array of function values

        CURRENTLY TESTING ON FUNCTION X**2 * GAUSSIAN 
        '''

        return x**2 

    def integrand_mcmc_p(self, x):
        ''' this is the integrand function for mcmc

        inputs: x(array), denoting iter number of sample points, given by
            metropolis hastings

        output:
            array of function values
        
        CURRENTLY TESTING ON FUNCTION X**2 * GAUSSIAN 
        '''

        return 1 / np.sqrt(2 * np.pi) * sp.exp(-(x) ** 2 /2)

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('MontyPython object self-destructing')



m = MontyPython()

class MiniMiss():

    def __init__(self):
        print('MiniMiss optimisation machine initialised and ready!')

    def minimise(self, expr, guess, args):
        starttime = time.time()

        temp = optimize.fmin(func, guess, args = (args), full_output = 1)

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
