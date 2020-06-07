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

import matplotlib.pyplot as plt

#### For future monte carlo integration work hopefully
import mcint
import random
from numba import jit, njit

class MontyPython():

    ### initialisation protocol for monteCarlo object
    def __init__(self, dimensions = 2, withplot = False):
        self.hbar = sc.hbar
        self.mass_e = sc.electron_mass
        self.q = sc.elementary_charge
        self.pi = sc.pi
        self.perm = sc.epsilon_0
        self.k = (self.q**2)/(4*self.pi*self.perm)


        # Variable used to control for one set of plots
        self.plotgraph = withplot

        # These are for the uniform sampling integrator
        self.bounds = [(-1,1),(-1,1)]
        self.dims = dimensions

        iterations = 100
        vals = np.zeros(iterations)
        for i in range(iterations):
            vals[i] = self.integrator_uniform(self.integrand, np.array([-10,10]), 100000)[0]
            print(i)
            
        print(np.average(vals))
        plt.hist(vals)
        plt.show()
#        print(self.sampler(self.bounds))

        # This the implementation for the Metropolis Algorithm of integration
#        print('\n\nMontyPython integrator initialised and ready! Dimensions = {}\n\n'.format(self.dims))
        # print(self.integrator_mcmc(self.integrand_mcmc_p, self.integrand_mcmc_q, np.zeros(1), 10000, 10, alpha= 1.))


### These helper functions concern the basic Monte Carlo Integrator
        
    def integrator_uniform(self, integrand, bounds, sample_iter):
        ''' using mcint package to determine the integral via uniform distribution
        sampling over an interval

        '''
        # measure is the 'volume' over the region you are integrating over

        result,error = mcint.integrate(integrand, self.sampler(bounds),
                                        self.get_measure(bounds), sample_iter)

        return result, error
    
    
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
        
        if bounds.ndim == 1:
            dimlength = bounds[1] - bounds[0]
            measure *= dimlength

        else:  
            for i in bounds:
            
                dimlength = i[1] - i[0]
                measure *= dimlength

        return measure


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
            
            if bounds.ndim == 1:
                    dimsample = random.uniform(bounds[0],bounds[1])
                    x = list(sample)
                    x.append(dimsample)
                    sample = tuple(x)
                
            else:
                for i in bounds:
                    dimsample = random.uniform(i[0],i[1])
        
                    x = list(sample)
                    x.append(dimsample)
                    sample = tuple(x)

            yield sample


    def integrand(self, x):
        ''' this is the integrand function

        inputs: x (array), where x[0] denotes the first variable

        '''

        return x[0]**2 * sp.exp(-(x[0]) ** 2)

    def combine_pq(self, pfunc, qfunc):
        ''' multiplies p and q for the uniform integrator! 
        '''
        return pfunc*qfunc
    
### These helper functions concern the Metropolis algorithm implementation of the integral
    def integrator_mcmc(self, pfunc, qfunc, initial_point, sample_iter, avg_iter, alpha):
        ''' fancy metropolis hastings integrator! where pfunc and qfunc give the
        function f you want to integrate over.

        inputs:
            pfunc: effective probability density function
            qfunc: some function where pfunc*qfunc = f
            alpha: variational parameter

        outputs:
            result: result of integral
            error: error of integral

        '''
        vals = np.zeros(avg_iter)
        val_errors = np.zeros(avg_iter)

        alpha = np.array([alpha])


        for i in range(avg_iter):
            mc_samples = self.metropolis_hastings(pfunc, sample_iter, initial_point, alpha)
            mc_samples = np.array(mc_samples)
            func_vals = []

            # obtain arithmetic average of sampled Q values
            for array in mc_samples:
                func_vals.append(qfunc(array, alpha))
                # print(qfunc(array, alpha))
            sums = np.sum(func_vals)

            vals[i] = (sums/sample_iter)
        # also calculate the variance
        vals_squared = np.sum(vals**2)
        vals_avg = np.sum(vals) / avg_iter

        for i in range (avg_iter):

            val_errors[i] = (np.sqrt(vals_squared/ avg_iter - (vals_avg) ** 2))

        result = vals_avg
        std = np.sqrt(np.sum(val_errors)/ np.sqrt(avg_iter))
        print('Iteration cycle complete, result = ', result, 'error = ', std)
        return result, std

    # @njit
    def metropolis_hastings(self, pfunc, iter, initial_pt, alpha):
        ''' Metropolis algorithm for sampling from a function q based on p

            inputs:
                pfunc: effective probability density function part of
                    function f we want to integrate over
                iter: number of random walk iterations
                initial_pt: starting point
                dims: dimensions of the sample
                alpha: variational parameter

            outputs:
                samples: array of points to input into qfunc
        '''

        dims = np.size(initial_pt)

        # simple sanity check
        if len(initial_pt) != dims:
            raise Exception('Error with inputs')

        # note: initial point chosen in input
        # keeping track of samples and initial_points
        samples = np.zeros((iter, dims))

        # now sample iter number of points
        for i in range(iter):
            # we propose a new point using a Symmetric transition distribution
            # function: a Gaussian
            proposed_pt = np.array(initial_pt) + np.random.normal(size=dims)

            # if the ratio is greater than one, accepept the proposal
            # else, accept with probability of the ratio


            if np.random.rand() < pfunc(proposed_pt, alpha) / pfunc(initial_pt, alpha):
                initial_pt = proposed_pt

            # add new point to array
            samples[i] = np.array(initial_pt)

        ### Some indicator for the running average of the points, looking at whether we should
        ### discard some of the initial samples or not
        if self.plotgraph:
            for i in range(dims):
                plt.plot(samples[:, i])
                mean_bin = 50
                plt.plot(self.running_mean(samples[:, i], mean_bin))
                plt.title('Sample histogram for random walk, dimension {}'.format(str(i+1)) )
                plt.ylabel('Number')
                plt.xlabel('Sample locations')
                plt.show(block = False)
                plt.pause(0.7)
                plt.close()

            ### Some graphs of the sample locations! Should look a lot like pfunc!
                plt.hist(samples[:, i], bins=np.arange(-5,5,0.2))
                plt.title('Sample histogram for random walk, dimension {}'.format(str(i+1)))
                plt.ylabel('Number')
                plt.xlabel('Sample locations')
                plt.show(block = False)
                plt.pause(0.7)
                plt.close()
            self.plotgraph = False

        return samples


    def mcmc_q(self, x, alpha):
        ''' this is the Q part of the integrand function for mcmc
        inputs: x(array), denoting iter number of sample points, given by
        '''


        ### helium local energy
        r1 = np.array(x[0:3])
        r2 = np.array(x[3:])
#        r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
#        r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)

        r1_len = np.linalg.norm(r1)
        r2_len = np.linalg.norm(r2)

        r1_hat = r1 / r1_len
        r2_hat = r2 / r2_len

        r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

        return (-4 + np.dot(r1_hat - r2_hat, r1 - r2 ) / (r12 * (1+alpha*r12)**2)
                - 1/ (r12*(1+alpha*r12)**3)
                - 1/ (4*(1+alpha*r12)**4)
                + 1/ r12 )


    def mcmc_p(self, x, alpha):
        '''
        this is the integrand function for mcmc

        '''
        ### helium wavefunction squared
        r1 = np.array(x[0:3])
        r2 = np.array(x[3:])
        r1_len = np.linalg.norm(r1)
        r2_len = np.linalg.norm(r2)
        r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

        return (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
                * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2


    def running_mean(self, x, N):
    # '''This is a helper function for computing the running mean'''
        cumsum = np.cumsum(np.insert(x, 0, 0))

        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('MontyPython object self-destructing')



m = MontyPython()

class MiniMiss():

    def __init__(self):
        print('MiniMiss optimisation machine initialised and ready!')

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
