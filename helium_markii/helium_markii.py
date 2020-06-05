#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario

import math
from optipack import MontyPython, MiniMiss
import psiham
import time
import numpy as np
import types
from numba import jit, njit


class Noble():

    def __init__(self, qfunc_name = 'qfunc'):
        print('Noble initialising')
        self.qfunc_name = qfunc_name
        self.monty = MontyPython(dimensions = 6)
        self.mini = MiniMiss()
        self.ham = psiham.HamLet()
        self.gen_q()
        self.gen_p()
        self.gen_pp()
        self.monty.plotgraph = True
        data = self.finalfinal()
        self.custom_log(data)

#    def get_functions(self):

    def expr1(self):
        return self.monty.integrator_mcmc(self.pfunc, self.he_elocal)

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
        hi = self.monty.integrator_mcmc(pfunc, qfunc, np.array([1,1,1,1,1,2]), 10000, 10, alpha = float(args))
        return hi[0]

    def finalfinal(self):
        temp = self.mini.minimise(func = self.final_comp, guess = 2.0, ftol = 0.05)
        return temp

    def custom_log(self, data = [], comments = None):
        log = open('markii_log.txt', 'a+')
        separator = '\n{}\n'.format(''.join('#' for i in range(10)))
        info = separator + 'Datetime = {}\n'.format(data[0]) + 'Time taken (s) = {}\n'.format(data[1]) + 'Initial parameter guess (atomic untis) = {}\n'.format(data[2]) + 'Optimised parameter (atomic units) = {}\n'.format(data[3]) + 'Optimised function value (atomic units) = {}\n'.format(data[4]) + 'Number of iterations = {}\n'.format(data[5]) + 'Comments = ({})\n'.format(comments)
        print(info)
        log.write(info)
        log.close()


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('Noble object self-destructing')

n = Noble()
