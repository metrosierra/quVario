#!/usr/bin/env python3

# Made 2020, Mingsong Wu
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario

### READINGS IN readings FOLDER VERY USEFUL FOR HELIUM ATOM APPROXIMATIONS



import math
import sys
import os

### using sympy methods for symbolic integration. Potentially more precise and convenient (let's not numerically estimate yet)
import numpy as np
import scipy as sp
import scipy.constants as sc
import sympy as sy
from sympy import conjugate, simplify, lambdify
from sympy import *
from IPython.display import display


class eiGen():

    def __init__(self):
        hbar = sc.hbar
        mass_e = sc.electron_mass
        q = sc.elementary_charge
        pi = sc.pi
        perm = sc.epsilon_0
        self.k = (q**2)/(4*pi*perm)
        sy.init_printing()
        self.macro1()

    def spawn_electrons(self, number, coord_sys = 's'):
        self.bases = []
        if coord_sys == 'spherical' or 's':
            for i in range(number):
                temp = []
                for q in ['r', 'theta', 'phi']:
                    temp.append(sy.symbols(q + str(i)))
                self.bases.append(temp)

            print(self.bases)

        elif coord_sys == 'cartesian' or 'cart':
            for i in range(number):
                temp = []
                for q in ['x', 'y', 'z']:
                    temp.append(sy.symbols(q + str(i)))
                self.bases.append(temp)

        elif coord_sys == 'cylindrical' or 'cylin':
            for i in range(number):
                temp = []
                for q in ['rho', 'phi', 'z']:
                    temp.append(sy.symbols(q + str(i)))
                self.bases.append(temp)

    def laplace(self, expr, index, coord_sys = 'spherical'):

        if coord_sys == 'spherical' or 's':
            r = self.bases[index][0]
            t = self.bases[index][1]
            p = self.bases[index][2]
            grad = [sy.diff(expr, r), 1/r * sy.diff(expr, t), 1/(r*sin(t)) * sy.diff(expr, p)]
            lap = (1/r**2)*sy.diff(r**2 * grad[0], r) + (1/(r*sin(t)))*(sy.diff(grad[1] * sin(t), t) + sy.diff(grad[2], p))
        return lap

    def expectation(self, psi, ham):
        pass


    def hamiltonian(self):

        pass




    def psi_trial(self):
        alpha = sy.symbols('alpha', real = True)
        self.psi = exp(-alpha(self.r1.r + self.r2.r))
        pass




    def macro1(self):

        alpha = sy.symbols('alpha', real = True)
        self.spawn_electrons(2)

        r0 = self.bases[0][0]
        theta0 = self.bases[0][1]
        phi0 = self.bases[0][2]

        r1 = self.bases[1][0]
        theta1 = self.bases[1][1]
        phi1 = self.bases[1][2]

        self.psi = sy.exp(-alpha*(r0 + r1))
        jacobian = (r0**2)*sin(theta0)

        self.integrand = conjugate(self.psi) * self.laplace(self.psi, 0) * jacobian
        self.expectprime = sy.integrate(self.integrand, (theta0, 0, pi), (phi0, 0, 2*pi),(r0, 0, oo), (r1, 0, oo), conds="none")
        self.normal = sy.integrate(conjugate(self.psi)*self.psi*jacobian, (theta0, 0, pi), (phi0, 0, 2*pi), (r0, 0, oo), (r1, 0, oo), conds="none")
        self.expectfinal = self.expectprime / self.normal * 0.5
        print(self.expectfinal)


    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass

e = eiGen()
