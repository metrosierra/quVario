#!/usr/bin/env python3

# Made 2020, Mingsong Wu
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario

### READINGS IN readings FOLDER VERY USEFUL FOR HELIUM ATOM APPROXIMATIONS

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

    ### initialisation protocol for eiGen object
    def __init__(self):
        self.hbar = sc.hbar
        self.mass_e = sc.electron_mass
        self.q = sc.elementary_charge
        self.pi = sc.pi
        self.perm = sc.epsilon_0
        self.k = (self.q**2)/(4*self.pi*self.perm)
        sy.init_printing()

        self.macro1()



    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass

### start script
e = eiGen()
