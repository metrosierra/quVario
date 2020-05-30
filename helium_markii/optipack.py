#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario


### This script provides the montyCarlo object AND miniMiss object that handles any calls to monte carlo integration functions and minimisation functions of our design. It interfaces with the python packages installed for basic functionalities (ie mcint)
### optipack should serve helium_markii.py which is the higher level script for direct user interface


### as of 27 may 2020 montycarlo uses mcint, and minimiss uses scipy fmin functions as main work horses.


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

### Honestly OOP isnt really shining in advantage now, other than me not caring about the order of functions and using global variables liberally.
class montyCarlo():

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
        self.dims = 2
        sy.init_printing()


#        print(self.integrator_basic())
        
#        print(self.sampler(self.bounds))
        
        print(self.integrator_mcmc(self.integrand_mcmc_p, self.integrand_mcmc_q, 
                                   sample_iter = 100000, avg_iter = 10))

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
    def integrator_mcmc(self, pfunc, qfunc, sample_iter = 100000, avg_iter = 10):
        ''' fancy metropolis hastings integrator! where pfunc and qfunc give the
        function f you want to integrate over. 
        
        inputs:
            pfunc: effective probability density function
            qfunc: some function where pfunc*qfunc = f 
            
        outputs: 
            result: result of integral
            error: error of integral
        
        '''
        vals = np.zeros(10)
        val_errors = np.zeros(10)
            
        for i in range (avg_iter):
            
            mc_samples = self.metropolis_hastings(self.integrand_mcmc_p, sample_iter, 
                                                  initial_pt = np.array([0]), dims = 1)
            
            func_vals = self.integrand_mcmc_q(mc_samples)
            
            sums = np.sum(func_vals)
            
            vals[i] = (sums/sample_iter)
        
        vals_squared = np.sum(vals**2)
        
        vals_avg = np.sum(vals)/ avg_iter
        
        for i in range (avg_iter):
            
            val_errors[i] = (np.sqrt(vals_squared/ avg_iter - (vals_avg) ** 2))
            
        result = vals_avg
        error = np.sum(val_errors)/ np.sqrt(avg_iter)
        
        return result, error
    
    def metropolis_hastings(self, p, iter=100000, initial_pt = [0.,0.,0.,0.,0.,0.], dims = 6):
        ''' Metropolis algorithm for sampling from a function p
        
            inputs: 
                
            outputs: 
    
        '''
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
            if np.random.rand() < p(proposed_pt) / p(initial_pt):
                initial_pt = proposed_pt

            samples[i] = np.array(initial_pt)

        return samples
    
    
    def integrand_mcmc_q(self, x):
        ''' this is the integrand function for mcmc
        
        inputs: x(array), denoting iter number of sample points, given by
            metropolis hastings
            
        output:
            array of function values 
            
        ''' 

        return x**2 #sp.exp(-(x) ** 2)
    
    def integrand_mcmc_p(self, x):
        ''' this is the integrand function for mcmc
        
        inputs: x(array), denoting iter number of sample points, given by
            metropolis hastings
            
        output:
            array of function values 
        
        ''' 

        return 1 / np.sqrt(2 * np.pi) * sp.exp(-(x) ** 2 /2)

    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass


zsm = montyCarlo()
