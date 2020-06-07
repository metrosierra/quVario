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
from numba import jit, njit
from sympy.utilities.lambdify import lambdastr

'''
MS's note: I will recycle my previous script for the two sets of electron coordinates and use sympy to symbolically generate the laplacian for each electron. I think we can safe assume our trial functions are analytical indefinitely so finding the laplacian is easy?

everything will be kept to atomic units
'''

class HamLet():
    def __init__(self):
        starttime = time.time()
        print('Hamiltonian Operator object initialised')
        self.coordsys = 'cartesian'

        with PsiLet() as psi:
            self.bases = psi.bases
            self.alphas = psi.alphas
            self.trial = parse_expr(psi.manual1())
            self.elocal = parse_expr(psi.manual2())
            # self.e_local = parse_expr(psi.manual2())
            #for convenience
            self.r01 = parse_expr(psi.r01)
            self.r0 = parse_expr(psi.r0)
            self.r1 = parse_expr(psi.r1)
        self.variables = [a for b in self.bases for a in b]


    #custom 'numbafy' protocol. adapted from Prof Slavic (jankoslavic)
    def numbafy(self, expression, coordinates = None, parameters = None, name='trial_func'):

        if coordinates:
            code_coord = []
            for i, x in enumerate(coordinates):
                code_coord.append(f'{x} = coordinates[{i}]')
            code_coord = '\n    '.join(code_coord)

        if parameters:
            code_param = []
            for i, x in enumerate(parameters):
                code_param.append(f'{x} = parameters[{i}]')
            code_param = '\n    '.join(code_param)
#             if constants:
#                 code_constants = []
#                 for k, v in constants.items():
#                     code_constants.append(f'{k} = {v}')
#                 code_constants = '\n    '.join(code_constants)

        temp = lambdastr((), expression)
        temp = temp[len('lambda : '):]
        code_expression = f'{temp}'

        template = f"""
@njit(parallel = True)
def {name}(coordinates, parameters):

    {code_coord}
    {code_param}

    return {code_expression}"""
        print('Function template generated! Its name is ', name)
        return template


    def he_getfuncs(self):
        return self.he_elocal(), self.he_norm(), self.he_trial()

    def he_elocal(self):
        operated = self.he_operate(self.trial)
        return (self.variables, operated/self.trial)

    def he_norm(self):
        return (self.variables, self.trial*self.trial)

    def he_trial(self):
        return (self.variables, self.trial)

    def he_operate(self, expr):
        lap1 = -0.5*self.laplace(expr, 0)
        lap2 = -0.5*self.laplace(expr, 1)
        attract = -2*(1/self.r0 + 1/self.r1)*expr
        repel = (1/self.r01)*expr
        return lap1 + lap2 + attract + repel


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
        print('HamLet object self-destructing...')



class PsiLet():
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

    def manual1(self):
        self.r0 = '(x0**2 + y0**2 + z0**2)**0.5'
        self.r1 = '(x1**2 + y1**2 + z1**2)**0.5'
        self.r01 = '((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)**0.5'

        expr = f'''exp(-2*{self.r0}-2*{self.r1}+{self.r01}/(2*(1+alpha0*{self.r01})))'''
        # expr = f'exp(-alpha0*({self.r0} + {self.r1}))'
        return expr

    def manual2(self):
        self.r0 = '(x0**2 + y0**2 + z0**2)**0.5'
        self.r1 = '(x1**2 + y1**2 + z1**2)**0.5'
        self.r01 = '((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)**0.5'
        self.dot01 = 'x0*x1 + y0*y1 + z0*z1'

        expr = f"""-4 + ({self.r0} + {self.r1})*(1 - {self.dot01}/({self.r0}*{self.r01}))/({self.r01}*(1+alpha0*{self.r01})**2) - 1/({self.r01}*(1+alpha0*{self.r01})**3) - 1/(4*{self.r01}*(1+alpha0*{self.r01})**4) + 1/{self.r01}"""
        return expr

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('psiLet object self-destructing')

ham = HamLet()
#
# variables, lolz = ham.he_elocal()
#
# q = ham.numbafy(lolz, ham.variables, ham.alphas)
# exec(q)
# # p = ham.numbafy(ham.e_local, parameters = variables, name= 'trial2' )
# # exec(p)
# trial = np.array([1,1,1,2,1,2,])
# alpha = np.array([1])
# hi = trial_func(trial, alpha)
# # hi2 = trial2([1,1,1,2,1,2,2])
#
# print(hi)
