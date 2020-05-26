#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario


### This script provides the montyCarlo object that handles any calls to monte carlo integration functions of our design. It interfaces with the python packages installed for basic functionalities (ie mcint)
### montycarlo should serve helium_markii.py which is the higher level script for direct user interface

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
        self.n = 100000000
        self.endpt = 1
        self.stpt = -1
        self.dims = 2
        sy.init_printing()

#        self.macro1()
        print(self.integrator_basic())

    def trialfunction(self):
        pass

    def get_measure(self):
        f = lambda x0, x1: 1.0
        bounds = (self.stpt, self.endpt)
        measure = integrate.nquad(f, [bounds, bounds])
#        print(measure)
        return measure[0]

    def integrator_basic(self):
        ''' using mcint package to determine the integral, crudely
        '''
        # measure is the 'volume' over the region you are integrating over
        result,error = mcint.integrate(self.integrand,self.sampler(),
                                        self.get_measure(), self.n)
        return result

    def sampler(self):
        while True:
            x = random.uniform(self.stpt,self.endpt)
            y = random.uniform(self.stpt,self.endpt)
            yield (x,y)

    def integrand(self, x):

        return sp.exp(-(x[0]*x[1]) ** 2)  #x**2

    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass


### start script
#e = eiGen()

m = montyCarlo()
