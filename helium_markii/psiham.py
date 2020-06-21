#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario


### This script provides the hamLet object which handles calls for functions pertaining to hamiltonian generation (supposed to be general, but this case our focus is helium 5 term hamiltonian)

### hamiltonian should serve helium_markii.py which is the higher level script for direct user interface

### This script also provides the psiLet object which handles calls for functions related to creating a trial wavefunction object. It MUST be compatible with hamiltonian such that the objects can interact to give expectation term (quantum chem)

### psifunction should serve helium_markii.py which is the higher level script for direct user interface

#%%%%%%%%%%%%%%%%%%

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
    def __init__(self, trial_expr, **kwargs):
        starttime = time.time()
        print('Hamiltonian Operator object initialised')
        self.coordsys = 'cartesian'

        with PsiLet(**kwargs) as psi:
            self.bases = psi.bases
            self.alphas = psi.alphas
            func_selected = psi.getfunc(trial_expr)
            self.trial = simplify(parse_expr(func_selected()))

            #for convenience
            self.r0 =  parse_expr(psi.r0)
            self.r1 =  parse_expr(psi.r1)
            self.r01 =  parse_expr(psi.r01)
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

        temp = lambdastr((), expression)
        temp = temp[len('lambda : '):]
        temp = temp.replace('math.exp', 'np.exp')
        code_expression = f'{temp}'

        template = f"""
@njit
def {name}(coordinates, parameters):
    {code_coord}
    {code_param}
    return {code_expression}"""
        print('Numba function template generated! Its name is ', name)
        return template

    def vegafy(self, expression, coordinates = None, name = 'vega_func'):

        if coordinates:
            code_coord = []
            for i, x in enumerate(coordinates):
                code_coord.append(f'{x} = coordinates[:,{i}]')
            code_coord = '\n    '.join(code_coord)

        temp = lambdastr((), expression)
        temp = temp[len('lambda : '):]
        temp = temp.replace('math.exp', 'np.exp')
        code_expression = f'{temp}'

        template = f"""
@vegas.batchintegrand
def {name}(coordinates):
    {code_coord}
    return {code_expression}"""
        print('Vega function template generated! Its name is ', name)
        return template

    def he_getfuncs(self):
        return self.he_elocal(), self.he_norm(), self.he_trial()

    def he_elocal(self):
        operated = self.he_operate(self.trial)
        return (self.variables, operated/self.trial)

    def he_norm(self):
        return (self.variables, self.trial*self.trial)

    def he_expect(self):
        operated = self.he_operate(self.trial)
        return (self.variables, operated*self.trial)

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
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('HamLet object self-destructing...')



class PsiLet():
    def __init__(self, electrons = 2, alphas = 1, coordsys = 'cartesian'):
        print('psiLet object initialised.')
        self.spawn_electrons(electrons, coord_sys = coordsys)
        print('{} {} electron coordinate set(s) spawned'.format(electrons, coordsys))
        self.spawn_alphas(alphas)
        print('{} alpha parameter(s) spawned'.format(alphas))

        self.r0 = '(x0**2 + y0**2 + z0**2)**0.5'
        self.r1 = '(x1**2 + y1**2 + z1**2)**0.5'
        self.r01 = '((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)**0.5'
        self.dot01 = 'x0*x1 + y0*y1 + z0*z1'


    def spawn_electrons(self, number, coord_sys = 'cartesian'):
        self.bases = []
        self.basis_names = {
        'cartesian': ['x', 'y', 'z'],
        'spherical': ['r', 'theta', 'phi'],
        'cylindrical': ['rho', 'phi', 'z']
        }

        for i in range(number):
            temp = []
            for q in self.basis_names[coord_sys]:
                temp.append(sy.symbols(q + str(i)))
            self.bases.append(temp)

    def spawn_alphas(self, number):
        self.alphas = []
        for i in range(number):
            self.alphas.append(sy.symbols('alpha' + str(i)))

    def getfunc(self, function):
        self.func_dict = {

        'onepara1': self.onepara1,
        'twopara1': self.twopara1,
        'threepara1': self.threepara1,
        'threepara2': self.threepara2

        }
        return self.func_dict[function]


    def onepara1(self):
        expr = f'''exp(-2*{self.r0}-2*{self.r1}+{self.r01}/(2*(1+alpha0*{self.r01})))'''
        return expr

    def twopara1(self):
        expr = f'''exp(-alpha0*({self.r0}+{self.r1})+{self.r01}/(2*(1+alpha1*{self.r01})))'''
        return expr

    def threepara1(self):
        expr = f'''(exp(-alpha0*{self.r0}-alpha1*{self.r1}) + exp(-alpha1*{self.r0}-alpha0*{self.r1}))*(1 + alpha2*{self.r01})'''
        return expr

    def threepara2(self):
        expr = f'''(exp(-alpha0*{self.r0}-alpha1*{self.r1}) + exp(-alpha1*{self.r0}-alpha0*{self.r1}))*exp({self.r01}/(2*(1 + alpha2*{self.r01})))'''
        return expr


    ### this function is flawed, gives numerically incorrect result. Use with caution
    def manual2(self):

        expr = f'''-4 + ({self.r0} + {self.r1})*(1 - {self.dot01}/({self.r0}*{self.r1}))/({self.r01}*(1+alpha0*{self.r01})**2) - 1/({self.r01}*(1+alpha0*{self.r01})**3) - 1/(4*(1+alpha0*{self.r01})**4) + 1/{self.r01}'''
        return expr

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('psiLet object self-destructing')
