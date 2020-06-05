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
        self.final_comp()



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


    def final_comp(self):
        hi = self.monty.integrator_mcmc(pfunc, qfunc, np.zeros(1), 10000, 10, alpha= 1.)
        print(hi)





    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('Noble object self-destructing')

n = Noble()
