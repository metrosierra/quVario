#!/usr/bin/env python3

# Made 2020, Mingsong Wu, Kenton Kwok
# mingsongwu [at] outlook [dot] sg
# github.com/starryblack/quVario

import math
import time
import numpy as np
from numba import njit, prange
import vegas

import matplotlib.pyplot as plt

from optipack import MiniMiss, LasVegas
from optipack import integrator_mcmc, metropolis_hastings
from optipack import Uint, Ueval
import psiham

import matplotlib.pyplot as plt

class Noble():

    def __init__(self, evals = 10000, iters = 40, verbose = True):
        print('Noble initialising')
        self.dims = 6
        self.mini = MiniMiss()
        self.vegas = LasVegas()
        psilet_args = {'electrons': 2, 'alphas': 3, 'coordsys': 'cartesian'}
        self.ham = psiham.HamLet(trial_expr = 'threepara2', **psilet_args)
        self.verbose = verbose

        self.numba_he_elocal()
        self.numba_pdf()
        self.numba_expec()
        self.vega_expec()
        self.vega_pdf()

        self.evals = evals
        self.iters = iters


        # data = self.minimise(function = self.monte_uniform, guess = np.array([2.001, 2.0, 0.2]))
        # data = self.minimise(function = self.mcmc_metro, guess = np.array([2.001, 2.0, 0.2]))

        # mcmc_args = [pfunc, qfunc , self.evals, self.iters, 6]
        # data = jit_grad_descent(integrator_mcmc, guess = np.array([1.9, 2.0, 0.2]), tolerance = 0.0005)
        # print(data)

        # data = self.grad_descent(self.monte_vegas, np.array([1.9, 2.0, 0.2]))
        # print(data)
        # for i in range(100):
        #     self.temp.append(self.monte_vegas(np.array([1.9, 2.0, 0.2])))


        # for i in range(100):
        #     self.temp.append(self.mcmc_metro(np.array([1.9, 2.0, 0.2])))

        # print(data[3])


        # self.custom_log(data, comments = 'Uniform integrator, 2 para')

    def numba_he_elocal(self):
        variables, expr = self.ham.he_elocal()
        temp = self.ham.numbafy(expr, coordinates = variables, parameters = self.ham.alphas, name = 'qfunc')
        exec(temp, globals())

    def numba_pdf(self):
        variables, expr = self.ham.he_norm()
        temp = self.ham.numbafy(expr, coordinates = variables, parameters = self.ham.alphas, name = 'pfunc')
        exec(temp, globals())

    def numba_expec(self):
        variables, expr = self.ham.he_expect()
        temp = self.ham.numbafy(expr, coordinates = self.ham.variables, parameters = self.ham.alphas, name = 'expec')
        exec(temp, globals())

    def vega_expec(self):
        variables, expr = self.ham.he_expect()
        temp = self.ham.vegafy(expr, coordinates = self.ham.variables, name = 'vega_expec')
        exec(temp, globals())

    def vega_pdf(self):
        variables, expr = self.ham.he_norm()
        temp = self.ham.vegafy(expr, coordinates = self.ham.variables, name = 'vega_pfunc')
        exec(temp, globals())

    def monte_metro(self, args):
        temp = integrator_mcmc(pfunc, qfunc , self.evals, self.iters, alpha = args, dims = self.dims, verbose = self.verbose)
        self.std = temp[1]
        self.reject_ratio = temp[2]
        return temp[0]

    def monte_uniform(self, args):
        temp = Ueval(pfunc, expec, self.evals, self.iters, alpha = args, dimensions = self.dims)
        self.std = temp[1]
        self.reject_ratio = 0.0
        return temp[0]

    def monte_vegas(self, alpha):
        global alpha0
        global alpha1
        global alpha2
        alpha0 = alpha[0]
        alpha1 = alpha[1]
        alpha2 = alpha[2]

        temp = self.vegas.vegas_int(vega_expec, vega_pfunc, self.evals, self.iters, dimensions = self.dims, volumespan = 8., verbose = self.verbose)
        return temp['energy']

    def nead_min(self, function, guess):
        temp = self.mini.neldermead(func = function, guess = guess, ftol = 0.01)
        # temp = grad_des(func = function, guess = guess, tolerance = 0.0001)
        return temp

    def grad_descent(self, function, guess):
        temp = self.mini.gradient(func = function, guess = guess, tolerance = 0.05)
        return temp

    def stability_protocol(self, fixedpoint, iterations):
        temp_master = [[],[],[]]
        functions = [self.monte_metro, self.monte_uniform, self.monte_vegas]
        for index, func in enumerate(functions):
            start = time.time()
            for i in range(iterations):
                temp_master[index].append(func(guess))
            print('Function #{} done!'.format(index))
        return temp_master


    def custom_log(self, data = [], comments = None):
        log = open('markii_log.txt', 'a+')
        separator = '\n{}\n'.format(''.join('#' for i in range(10)))
        info = separator + 'Datetime = {}\n'.format(data[0]) + 'Time taken (s) = {:.4f}\n'.format(data[1]) + 'Initial parameter guess (atomic untis) = {}\n'.format(data[2]) + 'Optimised parameter (atomic units) = {}\n'.format(data[3]) + 'Optimised function value (atomic units) = {:.4f} with std = {:.4f}\n'.format(data[4], self.std) + 'Iterations = {}, Evaluations/iter = {} with rejection ratio = {:.4f}\n'.format(self.iters, self.evals, self.reject_ratio) + 'Comments = ({})\n'.format(comments)
        print(info)
        log.write(info)
        log.close()


    def __enter__(self):
        return self

    def __exit__(self, e_type, e_val, traceback):
        print('Noble object self-destructing')

@njit
def jit_grad_des(func, guess, tolerance):

    cycle_count = 0
    position0 = guess
    ep = 0.00000001
    step = 0.2
    step_ratio = 0.5
    epsilons = np.identity(len(guess))*ep
    delta1 = 10.
    satisfied = False
    point_collection = []


    while satisfied == False:
        cycle_count+=1
        print('Cycle', cycle_count)
        value1 = func(pfunc, qfunc, sample_iter = 10000, walkers = 40, alpha = position0, dims = 6)[0]

        vector = np.zeros(len(guess))
        for i in range(len(guess)):
            vector[i] = (func(pfunc, qfunc, sample_iter = 10000, walkers = 40, alpha = position0 + epsilons[i], dims = 6)[0] - func(pfunc, qfunc, sample_iter = 10000, walkers = 40, alpha = position0, dims = 6)[0])/ep

        vectornorm = vector/(np.linalg.norm(vector))
        print(vectornorm)
        position0 += -vectornorm * step
        value2 = func(pfunc, qfunc, sample_iter = 10000, walkers = 40, alpha = position0, dims = 6)[0]
        delta1 = value2 - value1

        positionprime = position0 + vectornorm*(step*step_ratio)
        value3 = func(pfunc, qfunc, sample_iter = 10000, walkers = 40, alpha = positionprime, dims = 6)[0]
        delta2 = value3 - value1
        if delta2 < delta1:
            print('shrink!')
            position0 = positionprime
            step = step*step_ratio
            delta1 = delta2
            value2 = value3

        point_collection.append(value2)
        if convergence == 'strict':
            satisfied = abs(delta1) < tolerance
            finalvalue = value2

        elif convergence == 'fuzzy':
            if len(point_collection) >= 5:
                data_set = point_collection[-5:]
                print('std', statistics.pstdev(data_set))
                print('mean', statistics.mean(data_set))
                satisfied = abs(statistics.pstdev(data_set)/statistics.mean(data_set)) < tolerance
                finalvalue = statistics.mean(data_set)

    print('Convergence sucess!')
    return finalvalue, position0



n = Noble(evals = 10000, iters = 10, verbose = False)



evals = [10000, 50000, 100000, 500000, 1000000]
paras = [2,1]
guesses = [np.array([2.0,0.2]),  np.array([0.2])]
trials = ['twopara1', 'onepara1']

for x, para in enumerate(paras):
    data = []
    for i in evals:
        n.evals = i
        psilet_args = {'electrons': 2, 'alphas': para, 'coordsys': 'cartesian'}
        n.ham = psiham.HamLet(trial_expr = trials[x], **psilet_args)
        n.numba_pdf()
        n.numba_expec()
        e_min = n.nead_min(n.monte_uniform, guesses[x])[3][1]
        data.append(np.array([e_min, n.std]))
        print(data)
        print('{} and {}'.format(para,i), 'done')
    np.savetxt('__results__/processed/uniform_test/{}para_{}iterations_10avg.txt'.format(para, i), np.c_[np.array(data)])




# # n.stability_protocol(np.array([1.9, 2.0, 0.2]), 100)
# temp =[]
# for i in range(100):
#     start = time.time()
#     result = n.grad_descent(n.monte_metro, np.array([1.6, 2.0, 0.2]))
#     temp.append(result[0])
#     print(time.time() - start)





#%%%%%%%%%%%%%

# print(np.mean(temp))
# plt.hist(temp, bins = 15)

# iterations = 100
# np.savetxt('__results__/stability_study/mcmc_{}iterations.txt'.format(iterations), np.c_[np.array(temp0)])
# np.savetxt('__results__/stability_study/uniform_{}iterations.txt'.format(iterations), np.c_[np.array(temp1)])
# np.savetxt('__results__/stability_study/vegas.txt_{}iterations.txt'.format(iterations), np.c_[np.array(temp2)])
#
#
# plt.hist(temp0, bins = 10, histtype=u'step')
# plt.hist(temp1, bins = 10, histtype=u'step')
# plt.hist(temp2, bins = 10, histtype=u'step')
# plt.legend(['Metro', 'Uniform', 'Vegas'])
#
#
# plt.show()

#%%%%%%%%%%%
