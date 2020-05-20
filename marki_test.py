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
import sympy as sy
from sympy import exp, conjugate, symbols, simplify, integrate, lambdify

class eiGen():

    def __init__(self):
        hbar = sp.hbar
        mass_e = sp.electron_mass
        q = sp.elementary_charge
        pi = sp.pi
        perm = sp.epsilon_0
        k = (q**2)/(4*pi*perm)


    def delsquare(self, wavefunction):
        #instantiate del operator object, spits out symbolic result after the laplacian is applied
        delop = sp.vector.Del()
        laplace = delop.dot(delop(wavefunction)).doit()
        return laplace


    #TODO! sort out the twin spherical coordinate system laplacian for r1 r2
    #this is SPECIFICALLY the independent electron pair model, neglecting electron interaction, in a helium system
    def ham_he(self):
        ###in the future, I would want to implement a txt reader so i don't have to touch the "workhorse" code
        #instantiate coordinate system class, spherical
        self.r1 = sy.CoordSys3D('r1', transformation = 'spherical')
        self.del1 = self.delsquare()



        self.r2 = sy.CoordSys3D('r2', transformation = 'spherical')

        self.ham = -(0.5*mass_e*hbar**2)(self.del2(self.psi) + self.del2(self.psi)) - 2*k*(self.r1.r**-1 - self.r2.r*-1)


    def psi_trial(self):
        alpha = sy.symbols('alpha', real = True)
        self.psi = exp(-alpha(self.r1.r + self.r2.r))





    def __enter__(self):
        pass

    def __exit__(self, e_type, e_val, traceback):
        pass
