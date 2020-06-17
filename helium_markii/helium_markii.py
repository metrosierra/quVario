#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario

import math
from optipack import MontyPython, MiniMiss
import psiham
import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt


from metronumba import integrator_mcmc, metropolis_hastings, mcmc_q, mcmc_p

class Noble():

    def __init__(self, qfunc_name = 'qfunc'):
        print('Noble initialising')
        self.qfunc_name = qfunc_name
        self.monty = MontyPython(dimensions = 6)
        self.mini = MiniMiss()

        psilet_args = {'electrons': 2, 'alphas': 3, 'coordsys': 'cartesian'}
        self.ham = psiham.HamLet(trial_expr = 'threepara', **psilet_args)
        self.gen_q()
        self.gen_p()
        # self.gen_pp()
        self.monty.plotgraph = True
        data = self.finalfinal(guess = np.array([2., 2., 0.2]))
        self.custom_log(data)

#    def get_functions(self):

    # def expr1(self):
    #     return self.monty.integrator_mcmc(self.pfunc, self.he_elocal)

    def gen_q(self):
        variables, expr = self.ham.he_elocal()
        temp = self.ham.numbafy(expr, coordinates = variables, parameters = self.ham.alphas, name = self.qfunc_name)
        exec(temp, globals())

    def gen_p(self):
        variables, expr = self.ham.he_norm()
        temp = self.ham.numbafy(expr, coordinates = variables, parameters = self.ham.alphas, name = 'pfunc')
        exec(temp, globals())

    def gen_pp(self):
        temp = self.ham.numbafy(self.ham.elocal, coordinates = self.ham.variables, parameters = self.ham.alphas, name = 'pp')
        exec(temp, globals())


    def final_comp(self, args):
        temp = integrator_mcmc(pfunc, qfunc, 10000, 100, alpha = args, dims = 6)
        self.std = temp[1]
        self.reject_ratio = temp[2]
        # temp = integrator_mcmc(mcmc_p, mcmc_q, 10000, 10, alpha = float(args), dims = 6)
        # time = np.linspace(0,len(temp[3]), len(temp[3]))
        # # print(temp[3])
        # plt.plot(time, temp[3])
        # plt.show()
        # plt.show(block=False)
        # plt.pause(0.8)
        # plt.close()

        return temp[0]

    def finalfinal(self, guess):
        temp = self.mini.minimise(func = self.final_comp, guess = guess, ftol = 0.01)
        return temp

    def custom_log(self, data = [], comments = None):
        log = open('markii_log.txt', 'a+')
        separator = '\n{}\n'.format(''.join('#' for i in range(10)))
        info = separator + 'Datetime = {}\n'.format(data[0]) + 'Time taken (s) = {:.4f}\n'.format(data[1]) + 'Initial parameter guess (atomic untis) = {}\n'.format(data[2]) + 'Optimised parameter (atomic units) = {}\n'.format(data[3]) + 'Optimised function value (atomic units) = {:.4f} with std = {:.4f}\n'.format(data[4], self.std) + 'Number of iterations = {:.4f}, with rejection ratio = {:.4f}\n'.format(data[5], self.reject_ratio) + 'Comments = ({})\n'.format(comments)
        print(info)
        log.write(info)
        log.close()


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('Noble object self-destructing')

n = Noble()
