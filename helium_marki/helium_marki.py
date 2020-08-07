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
class eiGen():

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

    ### important bit: generating sets of 3d spherical coordinate sympy symbols for each electron. In this case 2, but can be easily scaled arbitrarily. All generated symbols stored in master self.bases list. Somehow I can't assign each variable a self attribute directly????
    def spawn_electrons(self, number, coord_sys = 's'):
        self.bases = []
        if coord_sys == 'spherical' or 's':
            for i in range(number):
                temp = []
                for q in ['r', 'theta', 'phi']:
                    temp.append(sy.symbols(q + str(i)))
                self.bases.append(temp)

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

    ### this is a kind of brute force laplace, because I cannot use the sympy.laplace function AND specify which electron it should operate wrt
    def laplace(self, expr, index, coord_sys = 'spherical'):

        if coord_sys == 'spherical' or 's':
            r = self.bases[index][0]
            t = self.bases[index][1]
            p = self.bases[index][2]
            grad = [sy.diff(expr, r), 1/r * sy.diff(expr, t), 1/(r*sin(t)) * sy.diff(expr, p)]
            lap = (1/r**2)*sy.diff(r**2 * grad[0], r) + (1/(r*sin(t)))*(sy.diff(grad[1] * sin(t), t) + sy.diff(grad[2], p))
        return lap

    ### Just makes the psi function variable. #TODO! augment this to read off a txt file (easier to see or find or collect the various trial forms)

    def spawn_psi(self):
        self.psi = -sy.exp(-self.alpha0*(self.r0 + self.r1))

    ### Just generating parameters. Only need 1 now.
    def spawn_alphas(self, number):
        for i in range(number):
            setattr(self, 'alpha{}'.format(i), sy.symbols('alpha{}'.format(i), real = True))

    ### Automatically slaps the SPHERICAL VOLUME jacobian for self.symintegrate. Tightly paired with said function.
    def jacob_gen(self, index, dimensions = 3, type = 's'):
        if dimensions == 3 and type == 's':
            return self.bases[index][0]**2*sin(self.bases[index][1])


    ## Symbolic integration (volume only for now) with spherical jacobian added for each set of coordinates
    def symintegrate(self, expr, number, dimensions = 3, type = "spherical"):
        if type == 'spherical' or type == 's':
            if dimensions == 3:
                for i in range(number):
                    temp = sy.integrate(expr*self.jacob_gen(index = i), (self.bases[i][0], 0, oo), (self.bases[i][1], 0, pi),(self.bases[i][2], 0, 2*pi), conds = "none")
                    expr = temp
        return temp


    ### Final workhorse that implements scipy fmin function to minimise our expression. The processing is general but the output is specific to our current helium example (1 parameter)
    def iterate(self, func, guess, args):
        print('\nIterator initialised!! Please be patient!!\n')
        starttime = time.time()

        temp = optimize.fmin(func, guess, args = (args), full_output = 1)

        endtime = time.time()
        elapsedtime = endtime - starttime
        now = datetime.now()
        date_time = now.strftime('%d/%m/%Y %H:%M:%S')

        # just returns datetime of attempt, elapsed time, optimised parameter, optimised value, number of iterations
        return [date_time, elapsedtime, guess, temp[0], temp[1], temp[3]]


    def lambdah(self, expr, var, type = 'scipy'):

        return lambdify((var), expr, type)

################################################################
#These are situation specific functions, for our particular helium integration problem.

    ###specifically atomic units
    def hamiltonian_he(self, trial):
        self.ke = -0.5 * (self.laplace(trial, 0) + self.laplace(trial, 1))
        self.nucleus = -(2/self.r0 + 2/self.r1)
        self.repel = (1/ sqrt(self.r0**2 + self.r1**2 - 2*self.r0*self.r1*cos(self.theta0)))

        print('\nKinetic Energy terms (self.ke), Nucleus Attraction terms (self.nucleus), Electron Repulsion terms (self.repel) generated!!!\n')
        return self.ke, self.nucleus, self.repel

    ### This our custom situtaion, where the hydrogenic component is analytic (ie a nice expression with parameter as sole variable) and the cross repulsion term is a numerically evaluated integral. So programme must substitute trial parameter and THEN integrate to check if end value is minimum
    def final_expr1(self, a, cross, hydrogenic):
        integrand = integrate.nquad(cross, [[0, sp.inf],  # r0
                                            [0, sp.pi],   # theta0
                                            [0, sp.inf], # phi0
                                        ], args = (a,)) #phi1

        print(hydrogenic(a) + integrand[0], 'heart trees')
        return hydrogenic(a) + integrand[0]

    ### writes self.iterate results to a paired txr file. If file no exist it is spawned.
    def custom_log(self, data = [], comments = None):
        log = open('marki_log.txt', 'a+')
        separator = '\n{}\n'.format(''.join('#' for i in range(10)))
        info = separator + 'Datetime = {}\n'.format(data[0]) + 'Time taken (s) = {}\n'.format(data[1]) + 'Initial parameter guess (atomic untis) = {}\n'.format(data[2]) + 'Optimised parameter (atomic units) = {}\n'.format(data[3]) + 'Optimised function value (atomic units) = {}\n'.format(data[4]) + 'Number of iterations = {}\n'.format(data[5]) + 'Comments = ({})\n'.format(comments)
        print(info)
        log.write(info)
        log.close()

    ### actual sequence of processing events. Technically this can be transfered to a separate macro script. Hence the name macro.
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
        self.hydrogenic = self.lambdah(result1/normal + result2/normal, (self.alpha0))

        jacob_special = 2*4*sp.pi**2*self.r0**2*self.r1**2*sin(self.theta0)
        cross_vars = (self.r0, self.theta0, self.r1, self.alpha0)
        self.cross_term = self.lambdah(simplify(jacob_special*conjugate(self.psi)*repel*self.psi/normal), (cross_vars))

        result = self.iterate(self.final_expr1, guess = 1.6, args = (self.cross_term, self.hydrogenic))
        self.custom_log(result, 'test run 2 with cross term, psi = -self.alpha0*(self.r0 + self.r1)')



    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass

### start script
e = eiGen()
