#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario


### This script provides the hamLet object which handles calls for functions pertaining to hamiltonian generation (supposed to be general, but this case our focus is helium 5 term hamiltonian)

### hamiltonian should serve helium_markii.py which is the higher level script for direct user interface

### This script also provides the psiLet object which handles calls for functions related to creating a trial wavefunction object. It MUST be compatible with hamiltonian such that the objects can interact to give expectation term (quantum chem)

### psifunction should serve helium_markii.py which is the higher level script for direct user interface

import math
import sys
import os
import time
from datetime import datetime

import numpy as np
import scipy as sp
import scipy.constants as sc
import sympy as sy
from sympy import conjugate, simplify, lambdify, sqrt
from sympy import *
from IPython.display import display
from scipy import optimize, integrate
from sympy.parsing.sympy_parser import parse_expr


'''
MS's note: I will recycle my previous script for the two sets of electron coordinates and use sympy to symbolically generate the laplacian for each electron. I think we can safe assume our trial functions are analytical indefinitely so finding the laplacian is easy?

everything will be kept to atomic units
'''

class hamLet():
    def __init__(self):
        starttime = time.time()
        print('Hamiltonian Operator object initialised')
        self.coordsys = 'cartesian'

        with psiLet() as psi:
            self.bases = psi.bases
            self.alphas = psi.alphas
            self.trial = parse_expr(psi.manualfunc())

            #for convenience
            self.r01 = parse_expr(psi.r01)
            self.r0 = parse_expr(psi.r0)
            self.r1 = parse_expr(psi.r1)


        self.he_expect = self.he_expectation()
        print('Helium energy expectation (non normalised, pre integrand) generated, available as "self".he_expect')
        self.he_normalisation = self.he_norm()
        print('Helium expectation integral normalisation term generated (pre integrand), available as "self".he_normalisation')
        # display(hi)
        endtime = time.time()
        elapsedtime = endtime - starttime
        print('time',elapsedtime)


    def he_expectation(self):
        operated = self.he_operate(self.trial)
        return self.trial*operated


    def he_operate(self, expr):
        lap1 = -0.5*self.laplace(self.trial, 0)
        lap2 = -0.5*self.laplace(self.trial, 1)
        attract = -(2/self.r0 + 2/self.r1)
        repel = 1/self.r01
        return lap1 + lap2 + attract + repel

    def he_norm(self):
        return self.trial*self.trial

    def laplace(self, expr, index, coordsys = 'c'):
        if coordsys == 'cartesian' or coordsys == 'c':
            x = self.bases[index][0]
            y = self.bases[index][1]
            z = self.bases[index][2]
            grad = [sy.diff(expr, x), sy.diff(expr, y), sy.diff(expr, z)]
            lap = sy.diff(grad[0], x) + sy.diff(grad[1], y) + sy.diff(grad[2], z)


        elif coord_sys == 'spherical' or coordsys == 's':
            r = self.bases[index][0]
            t = self.bases[index][1]
            p = self.bases[index][2]
            grad = [sy.diff(expr, r), 1/r * sy.diff(expr, t), 1/(r*sin(t)) * sy.diff(expr, p)]
            lap = (1/r**2)*sy.diff(r**2 * grad[0], r) + (1/(r*sin(t)))*(sy.diff(grad[1] * sin(t), t) + sy.diff(grad[2], p))

        return lap

    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass



class psiLet():
    def __init__(self, electrons = 2, alphas = 1, coordsys = 'cartesian'):
        print('psiLet object initialised, reading input txt file (if any) for trial wavefunction')
        self.spawn_electrons(2, coord_sys = coordsys)
        print('{} {} electron coordinate set(s) spawned'.format(electrons, coordsys))
        self.spawn_alphas(1)
        print('{} alpha parameter(s) spawned'.format(alphas))

    def spawn_electrons(self, number, coord_sys = 'cart'):
        self.bases = []
        if coord_sys == 'spherical' or coord_sys == 's':
            for i in range(number):
                temp = []
                for q in ['r', 'theta', 'phi']:
                    temp.append(sy.symbols(q + str(i), real = True))
                self.bases.append(temp)

        elif coord_sys == 'cartesian' or coord_sys == 'cart':
            for i in range(number):
                temp = []
                for q in ['x', 'y', 'z']:
                    temp.append(sy.symbols(q + str(i)))
                self.bases.append(temp)

        elif coord_sys == 'cylindrical' or coord_sys == 'cylin':
            for i in range(number):
                temp = []
                for q in ['rho', 'phi', 'z']:
                    temp.append(sy.symbols(q + str(i)))
                self.bases.append(temp)


    def spawn_alphas(self, number):
        self.alphas = []
        for i in range(number):
            self.alphas.append(sy.symbols('alpha' + str(i)))

    def getfunc(self):
        pass

    def manualfunc(self):
        self.r0 = '(x0**2 + y0**2 + z0**2)**0.5'
        self.r1 = '(x1**2 + y1**2 + z1**2)**0.5'
        self.r01 = '((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)**0.5'

        expr = 'exp(-2*({}+{}))*'.format(self.r0, self.r1) + 'exp(({})/(2*(1+4*{}*alpha0)))'.format(self.r01, self.r01)
        return expr


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('psiLet object destructing')

ham = hamLet()
