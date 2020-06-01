#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario


from optipack import MontyPython, MiniMiss
import psiham
import time
import numpy as np


class Noble():

    def __init__(self):
        print('Noble initialising')
        starttime = time.time()
        self.Monty = MontyPython()
        self.Mini = MiniMiss()
        self.psiham = psiham.HamLet()
        print(type(self.psiham))
        self.he_elocal, self.he_norm, self.he_trial = self.psiham.he_getfuncs()
        # print(self.he_trial(0,0,0,0,0,0))
        self.trial_norm = self.Monty.integrator_mcmc(self.he_trial, self.he_trial)
        # print(self.he_trial([0,0,0,0,0,0]))
        endtime = time.time()
        elapsedtime = endtime - starttime
        print(elapsedtime, 'done')
        self.pfunc = self.psiham.lambdah(self.psiham.trial/float(self.trial_norm[0]))
        #
        print(self.expr1())

#    def get_functions(self):

    def expr1(self):
        return self.Monty.integrator_mcmc(self.pfunc, self.he_elocal)


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('Noble object self-destructing')

n = Noble()
