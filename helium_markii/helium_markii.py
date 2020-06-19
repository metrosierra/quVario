#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario

import math
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from optipack import MiniMiss, integrator_mcmc, metropolis_hastings
import psiham

class Noble():

    def __init__(self, qfunc_name = 'qfunc'):
        print('Noble initialising')
        self.qfunc_name = qfunc_name
        self.mini = MiniMiss()

        psilet_args = {'electrons': 2, 'alphas': 3, 'coordsys': 'cartesian'}
        self.ham = psiham.HamLet(trial_expr = 'threepara2', **psilet_args)
        self.gen_he_elocal()
        self.gen_pdf()

        self.mcmc_length = 10000
        self.walkers = 100
        data = self.minimise(function = self.mcmc_metro, guess = np.array([2.001, 2.0, 0.2]))
        self.custom_log(data, comments = 'threepara2 used, with minus between two electron terms')

    def gen_he_elocal(self):
        variables, expr = self.ham.he_elocal()
        temp = self.ham.numbafy(expr, coordinates = variables, parameters = self.ham.alphas, name = self.qfunc_name)
        exec(temp, globals())

    def gen_pdf(self):
        variables, expr = self.ham.he_norm()
        temp = self.ham.numbafy(expr, coordinates = variables, parameters = self.ham.alphas, name = 'pfunc')
        exec(temp, globals())

    def gen_pp(self):
        temp = self.ham.numbafy(self.ham.elocal, coordinates = self.ham.variables, parameters = self.ham.alphas, name = 'pp')
        exec(temp, globals())


    def mcmc_metro(self, args):
        temp = integrator_mcmc(pfunc, qfunc, self.mcmc_length, self.walkers, alpha = args, dims = 6)
        self.std = temp[1]
        self.reject_ratio = temp[2]
        return temp[0]

    def minimise(self, function, guess):
        temp = self.mini.minimise(func = function, guess = guess, ftol = 0.01)
        return temp

    def custom_log(self, data = [], comments = None):
        log = open('markii_log.txt', 'a+')
        separator = '\n{}\n'.format(''.join('#' for i in range(10)))
        info = separator + 'Datetime = {}\n'.format(data[0]) + 'Time taken (s) = {:.4f}\n'.format(data[1]) + 'Initial parameter guess (atomic untis) = {}\n'.format(data[2]) + 'Optimised parameter (atomic units) = {}\n'.format(data[3]) + 'Optimised function value (atomic units) = {:.4f} with std = {:.4f}\n'.format(data[4], self.std) + 'Walkers = {}, MCMC iterations = {} with rejection ratio = {:.4f}\n'.format(self.walkers, self.mcmc_length, self.reject_ratio) + 'Comments = ({})\n'.format(comments)
        print(info)
        log.write(info)
        log.close()


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('Noble object self-destructing')

n = Noble()
