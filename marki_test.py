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
from sympy import conjugate, simplify, lambdify, sqrt
from sympy import *
from IPython.display import display
from scipy import optimize, integrate

import mcint
import random


class eiGen():

    def __init__(self):
        self.hbar = sc.hbar
        self.mass_e = sc.electron_mass
        self.q = sc.elementary_charge
        self.pi = sc.pi
        self.perm = sc.epsilon_0
        self.k = (self.q**2)/(4*self.pi*self.perm)
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
# (self.hbar**2/(2*self.mass_e))
# self.k *
    #cylindrical coordinate helium atom hamiltonian
    def hamiltonian_he(self, trial):
        self.ke = -0.5 * (self.laplace(trial, 0) + self.laplace(trial, 1))
        self.nucleus = -(2/self.r0 + 2/self.r1)
        self.repel = -self.k * (-1/ sqrt(self.r0**2 + self.r1**2 - 2*self.r0*self.r1*cos(self.theta0)))

        print('\nKinetic Energy terms (self.ke), Nucleus Attraction terms (self.nucleus), Electron Repulsion terms (self.repel) generated!!!\n')
        return self.ke, self.nucleus, self.repel



    def spawn_psi(self):
        self.psi = -sy.exp(-self.alpha0*(self.r0 + self.r1))

    def spawn_alphas(self, number):
        for i in range(number):
            setattr(self, 'alpha{}'.format(i), sy.symbols('alpha{}'.format(i), real = True))

    def jacob_gen(self, index, dimensions = 3, type = 's'):
        if dimensions == 3 and type == 's':
            return self.bases[index][0]**2*sin(self.bases[index][1])


    def symintegrate(self, expr, number):

        for i in range(number):
            dummy = sy.integrate(expr*self.jacob_gen(index = i), (self.bases[i][0], 0, oo), (self.bases[i][1], 0, pi),(self.bases[i][2], 0, 2*pi), conds = "none")
            expr = dummy
        return dummy

    def scipy_integral(self, expr, variables, number = 2):
        grand_jacob = 1
        for i in range(number):
            grand_jacob *= self.jacob_gen(index = i)
        lamb = lambdify(variables, expr*grand_jacob, 'scipy')
        print(lamb)
        self.alpha0 = 2.0
        integrand = integrate.nquad(lamb, [[0, sp.inf],  # r0
                                            [0, sp.pi],   # theta0
                                            [0, 2*sp.pi], # phi0
                                            [0, sp.inf],   # r1
                                            [0, sp.pi],   # theta1
                                            [0, 2*sp.pi]], args = (self.alpha0,)) #phi1


        domainsize =

        #
        # random.seed(1)
        # result, error = mcint.integrate(integrand, sampler(), measure=domainsize, n = 100000)

        return integrand[0]




    def macro1(self):

        self.spawn_alphas(2)
        self.spawn_electrons(2)

        self.r0 = self.bases[0][0]
        self.theta0 = self.bases[0][1]
        self.phi0 = self.bases[0][2]

        self.r1 = self.bases[1][0]
        self.theta1 = self.bases[1][1]
        self.phi1 = self.bases[1][2]

        self.spawn_psi()
        ke, attract, repel = self.hamiltonian_he(self.psi)

        normal = self.symintegrate(conjugate(self.psi)*self.psi, 2)
        result1 = self.symintegrate(conjugate(self.psi)*ke, 2)
        result2 = self.symintegrate(conjugate(self.psi)*attract*self.psi, 2)
        display(result1/normal + result2/normal)
        hydrogenic = lambdify((self.alpha0), result1/normal + result2/normal, 'scipy')

        cross_term = simplify(conjugate(self.psi)*repel*self.psi/normal)

        int_var = (self.r0, self.theta0, self.phi0, self.r1, self.theta1, self.phi1, self.alpha0)


        initial_guess = 2.0
        hi = optimize.fmin(hydrogenic, initial_guess)

        hi2 = self.scipy_integral(cross_term, int_var)
        # hi2 = optimize.fmin(self.scipy_integral, initial_guess, args = (cross_term, int_var))
        print(hi2)

    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass

e = eiGen()
