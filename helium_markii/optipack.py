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
        self.n = 100000
        self.endpt = 1
        self.stpt = -1
        self.bounds = [(-1,1),(-1,1),(-1,1)]
        self.dims = 2
        sy.init_printing()

#        self.macro1()
        print(self.integrator_basic())
#        print(self.sampler(self.bounds))

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
        '''
        using mcint package to determine the integral, crudely

        '''
        # measure is the 'volume' over the region you are integrating over

        result,error = mcint.integrate(self.integrand,self.sampler(self.bounds),
                                        self.get_measure(self.bounds), self.n)

        return result

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

        '''

        return sp.exp(-(x[0]*x[1]*x[2]) ** 2)  #x**2

    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass


### start script
#e = eiGen()

m = montyCarlo()
