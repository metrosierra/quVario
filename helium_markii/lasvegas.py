#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vegas example from the Basic Integrals section
of the Tutorial and Overview section of the
vegas documentation.
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import numpy as np
import time
from numba import jit, njit
from scipy.optimize import fmin
import matplotlib.pyplot as plt

#from optipack import MiniMiss, integrator_mcmc, metropolis_hastings, mcmc_q, mcmc_p
import psiham

psilet_args = {'electrons': 2, 'alphas': 1, 'coordsys': 'cartesian'}
ham = psiham.HamLet(trial_expr = 'twopara1', **psilet_args)
variables, expr = ham.he_expect()
temp1 = ham.vegafy(expr, coordinates = variables, name = 'expec')

variables, expr = ham.he_norm()
temp2 = ham.vegafy(expr, coordinates = variables, name = 'norm')
exec(temp1, globals())
exec(temp2, globals())

final_results = {}

SHOW_GRID = False
OPTIM = True
def main(alpha):

    global alpha0
#    global alpha1
#    global alpha2
    alpha0 = alpha[0]
#    alpha1 = alpha[1]
#    alpha2 = alpha[2]

    start_time = time.time()

    # assign integration volume to integrator
    bound = 8
    dims = 6
    # creates symmetric bounds specified by [-bound, bound] in dims dimensions
    symm_bounds = dims * [[-bound,bound]]

    # simultaneously initialises expectation and normalisation integrals
    expinteg = vegas.Integrator(symm_bounds)
    norminteg = vegas.Integrator(symm_bounds)

    # adapt to the integrands; discard results
    expinteg(expec, nitn=5, neval=1000)
    norminteg(norm,nitn=5, neval=1000)
    # do the final integrals
    expresult = expinteg(expec, nitn=10, neval=100000)
    normresult = norminteg(norm, nitn=10, neval=100000)


    if not OPTIM:
        E = expresult.mean/normresult.mean
    else:
        E = expresult[0].mean/ normresult[0].mean
#    print('Energy is {} when alpha is {}'.format(E, alpha), ' with sdev = ', [expresult.sdev, normresult.sdev])
#    print("--- Iteration time: %s seconds ---" % (time.time() - start_time))

    final_results['energy'] = E
    final_results['dev'] = [expresult.sdev, normresult.sdev]
    final_results['pvalue'] = [expresult.Q, normresult.Q]

    return E

### Optimisation ALgorithm
start_time = time.time()
OPTIM = False

#plt.figure(figsize=(16,10))

#print('Plotting function initialised!')
#energies = []
#alpha = np.linspace(0.1, 0.2, 20)
# for i in alpha:
#     alpha0 = i
#
#     energy_samples = []
#     for j in range(1):
#         energy_samples.append(evalenergy(i))
#     avg = np.average(energy_samples)
#     print('Averaged energy for %f is %f' %(i, avg))
#     energies.append(avg)
# plt.plot(alpha, energies, color='dodgerblue')
# print("--- Total time: %s seconds ---" % (time.time() - start_time))

# plt.savefig('Vegas')

OPTIM = False
start_time = time.time()
result = fmin(main, [0.15], ftol = 0.01, xtol = 0.001, full_output=True)
#e = evalenergy(0.5)
#print(e)
print("--- Total time: %s seconds ---" % (time.time() - start_time))

#%%%%%%%%%%%%%%%%%%%%
#########################################################################
#@njit
#def f(x, alpha):
#    ''' Expectation value function, used as numerator for variational integral
#    '''
#    r1 = np.array([x[0], x[1], x[2]])
#    r2 = np.array([x[3], x[4], x[5]])
#
#    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
#    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
#
#    r1_hat = r1 / r1_len
#    r2_hat = r2 / r2_len
#
#    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)
#
#    EL =  (-4 + np.dot(r1_hat - r2_hat, r1 - r2 ) / (r12 * (1 + alpha*r12)**2)
#            - 1/ (r12*(1 + alpha*r12)**3)
#            - 1/ (4*(1 + alpha*r12)**4)
#            + 1/ r12 )
#
#    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
#            * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
#
#    return psisq * EL
#
#@njit
#def psisq(x, alpha):
#    ''' Squared trial wavefunction, used as denominator for variational integral
#    '''
#    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
#    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
#
#    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)
#
#    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
#            * np.exp(r12 / (2 * (1+ alpha* r12)))) ** 2
#
#    return psisq

#
#def evalenergy(alpha):
#    @vegas.batchintegrand
#    def expec(x):
#        ''' Expectation value function, used as numerator for variational integral
#        '''
#
#        r1 = np.array(x[:,0:3])
#        r2 = np.array(x[:,3:])
#
#        r1_len = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
#        r2_len = np.sqrt(x[:,3]**2 + x[:,4]**2 + x[:,5]**2)
#
#        r1_hat = r1 / np.reshape(r1_len, (-1,1))
#        r2_hat = r2 / np.reshape(r2_len, (-1,1))
#
#        r12 = np.sqrt((x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
#
#        EL =  (-4 + np.sum((r1_hat-r2_hat)*(r1-r2), 1) / (r12 * (1 + alpha*r12)**2)
#                - 1/ (r12*(1 + alpha*r12)**3)
#                - 1/ (4*(1 + alpha*r12)**4)
#                + 1/ r12 )
#
#        psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
#                * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
#        return psisq * EL
#
#    @vegas.batchintegrand
#    def norm(x):
#        ''' Squared trial wavefunction, used as denominator for variational integral
#        '''
##        x = np.reshape(x, (1,-1))
#
#        r1_len = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
#        r2_len = np.sqrt(x[:,3]**2 + x[:,4]**2 + x[:,5]**2)
#
#        r12 = np.sqrt((x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
#
#        psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
#                * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
#        return psisq
#
#    def main():
#        start_time = time.time()
#
#        # assign integration volume to integrator
#        bound = 8
#        dims = 6
#        # creates symmetric bounds specified by [-bound, bound] in dims dimensions
#        symm_bounds = dims * [[-bound,bound]]
#
#        # simultaneously initialises expectation and normalisation integrals
#        expinteg = vegas.Integrator(symm_bounds)
#        norminteg = vegas.Integrator(symm_bounds)
#
#        # adapt to the integrands; discard results
#        expinteg(expec, nitn=5, neval=1000)
#        norminteg(norm,nitn=5, neval=1000)
#        # do the final integrals
#        expresult = expinteg(expec, nitn=10, neval=10000)
#        normresult = norminteg(norm, nitn=10, neval=10000)
#
#### Code for printing the grid
#### and diagnostics
##        print(expresult.summary())
##        print('expresult = %s    Q = %.2f' % (expresult, expresult.Q))
##        if SHOW_GRID:
##            expinteg.map.show_grid(20)
##
##        print(normresult.summary())
##        print('normresult = %s    Q = %.2f' % (normresult, normresult.Q))
##        if SHOW_GRID:
##            norminteg.map.show_grid(20)
#
#        ### obtain numerical result
#        ### Different expressions for plotting/ minimisation algorithm
#        if not OPTIM:
#            E = expresult.mean/normresult.mean
#        else:
#            E = expresult[0].mean/ normresult[0].mean
##        print('Energy is %f when alpha is %f' %(E, alpha))
##        print("--- Iteration time: %s seconds ---" % (time.time() - start_time))
#        final_results['energy'] = E
#        final_results['dev'] = [expresult.sdev, normresult.sdev]
#        final_results['pvalue'] = [expresult.Q, normresult.Q]
#        return E
#    E = main()
#    return E


# #%% This is for one-parameter plotting
# #### Run plotting function, plot alpha against energies
# start_time = time.time()
# OPTIM = False
#
# plt.figure(figsize=(16,10))
# #high resolution bit closer to the minimum
# #alpha0 = np.linspace(0.001, 0.1, 5)
# #energies0 = []
# #for i in alpha0:
# #    energies0.append(evalenergy(i))
#
# print('Plotting function initialised!')
# energies = []
# alpha = np.linspace(0.1, 0.2, 20)
# for i in alpha:
#     energy_samples = []
#     for j in range(1):
#         energy_samples.append(evalenergy(i))
#     avg = np.average(energy_samples)
#     print('Averaged energy for %f is %f' %(i, avg))
#     energies.append(avg)
# plt.plot(alpha, energies, color='dodgerblue')
# print("--- Total time: %s seconds ---" % (time.time() - start_time))
#
# plt.savefig('Vegas')
#
# #%%
# ##low resolution bit
# #alpha = np.linspace(0.25, 2, 10)
# #energies2 = []
# #for i in alpha:
# #    energies2.append(evalenergy(i))
# #plt.plot(alpha, energies2, color = 'dodgerblue')
# #

# #### Use optimisation to obtain the minimum
# #### and plot a point
# ##OPTIM = False
# ##result = fmin(evalenergy, 0.2, ftol = 0.01, xtol = 0.001, full_output=True)
# ##plt.plot(result[0], result[1], 'ro', ms=5)
# #
# #plt.xlabel('alpha')
# #plt.ylabel('energy')
# #plt.grid()
# #plt.show()
# #print("--- Total time: %s seconds ---" % (time.time() - start_time))
#

##############################################################################
### THE FOLLOWING CODE IS FOR EXTRACTING DATA AND PLOTTING GRAPHS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### To plot a histogram of evaluations
#### VEGAS PLOTS FOR ENERGIES AND HISTOGRAMS
#N = 50
#
#energies = []
#optparas = []
#start_time = time.time()
#print('Loop initialised!')
#
#for i in range(N):
#    print(i)
#    loop_time = time.time()
#    result = fmin(evalenergy, [0.15], ftol = 0.01, xtol = 0.001, full_output=1, disp=0)
#    energies.append(result[1])
#    optparas.append(float(result[0]))
#    print("--- Loop time: %s seconds ---" % (time.time() - loop_time))
#    
#print("--- Total time: %s seconds ---" % (time.time() - start_time))
#
###########
#### Save data into a csv
#n = '1e5'
#name = 'vegas_1param_iters=%s_N=%i' %(n,N)
#np.savetxt('%s.csv' %name, np.transpose([energies,optparas]), 
#           fmt = '%s', delimiter=',', header = (name + '_t=2176'))
#
#
##########
#### Plot histograms
#avg = np.mean(energies)
#std = np.std(energies)
#plt.figure(figsize=(16,10))
#plt.rcParams.update({'font.size': 22})
#
#plt.hist(energies,10)
#plt.xlabel('Energy (Atomic Units)')
#plt.ylabel('N')
#plt.figtext(0.2,0.75,'oneparam, N=%i, \n' 
#            'n=%s, bound=8 \n'
#            'avg E= %.5f \n'
#            'std = %.5f'%(N, n, avg,std))
#plt.title('Histogram of distributed optimised energy values for VEGAS')
##plt.savefig('vegas_energyhist_1param_%s' %n)
#plt.show()
#
#alphaavg = np.mean(optparas)
#alphastd = np.std(optparas)
#plt.figure(figsize=(16,10))
#plt.xlabel('Variational parameter alpha')
#plt.ylabel('N')
#plt.figtext(0.6,0.7,'oneparam, N=%i, \n n=%s, bound=8 \n'
#            'avg alpha = %.5f \n'
#            'std = %.5f'%(N, n, alphaavg,alphastd))
#plt.title('Histogram of distributed optimised parameters for VEGAS')
#plt.hist(optparas,10)
##plt.savefig('vegas_alphahist_1_param_%s' %n)
#plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### VEGAS analysis Code to analyse the Q values
#
#Qexp = np.zeros((50,5))
#Qnorm = np.zeros((50,5))
#
#N = 50 
#n = 5e5
#alpha = [0.15606, 0.15387, 0.15156, 0.15109, 0.15336]
#
#index = 0
#
#qexp = [] 
#qnorm = []
#
#for i in range(N):
#    print(i)
#    loop_time = time.time()
#    evalenergy(alpha[index])
#    qexp.append(final_results['pvalue'][0])
#    qnorm.append(final_results['pvalue'][1])
#    print("--- Loop time: %s seconds ---" % (time.time() - loop_time))
#
#print('The average Qexp value is: ' ,np.average(qexp))
#print('The average Qnorm value is: ' ,np.average(qnorm))
#
##Qexp[:,index] = qexp
##Qnorm[:,index] = qnorm
#
#
###########
#### Save data into a csv
##Qe = np.zeros((50,5))
#
##name = 'vegas_1param_Qnorm' 
##np.savetxt('%s.csv' %name, Qnorm,
##           fmt = '%s', delimiter=',', header = (name + str(iters)))
#
#### Obtain the average
#iters = [1e4, 5e4, 1e5, 5e5, 1e6]
#x = []
#y = []
#for i in range(5):
#    x.append(np.average(Qexp[:,i]))
#    y.append(np.average(Qnorm[:,i]))
#
#plt.scatter(iters, x)
#plt.scatter(iters, y)
#plt.show()
#
#
#### Obtain the standard deviation
#xstd = []
#ystd = []
#for i in range(5):
#    xstd.append(np.std(Qexp[:,i]))
#    ystd.append(np.std(Qnorm[:,i]))
#
#plt.scatter(iters, xstd)
#plt.scatter(iters, ystd)
#plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### Analysis of the impact of bound sizes
#
#bounds = [1.5, 1.6, 1.7,1.8, 1.9]
#B_energy_2 = np.zeros((50, 5))
#B_alpha_2 = np.zeros((50, 5))
#
#N = 50
#OPTIM = False
#
#for j in range(len(bounds)):
#    bound = bounds[j]
#    benergy = []
#    bopt = []
#    def evalenergy(alpha):
#        @vegas.batchintegrand
#        def expec(x):
#            ''' Expectation value function, used as numerator for variational integral
#            '''
#            r1 = np.array(x[:,0:3])
#            r2 = np.array(x[:,3:])
#    
#            r1_len = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
#            r2_len = np.sqrt(x[:,3]**2 + x[:,4]**2 + x[:,5]**2)
#    
#            r1_hat = r1 / np.reshape(r1_len, (-1,1))
#            r2_hat = r2 / np.reshape(r2_len, (-1,1))
#    
#            r12 = np.sqrt((x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
#    
#            EL =  (-4 + np.sum((r1_hat-r2_hat)*(r1-r2), 1) / (r12 * (1 + alpha*r12)**2)
#                    - 1/ (r12*(1 + alpha*r12)**3)
#                    - 1/ (4*(1 + alpha*r12)**4)
#                    + 1/ r12 )
#    
#            psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
#                    * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
#            return psisq * EL
#    
#        @vegas.batchintegrand
#        def norm(x):
#            ''' Squared trial wavefunction, used as denominator for variational integral
#            '''
#    
#            r1_len = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
#            r2_len = np.sqrt(x[:,3]**2 + x[:,4]**2 + x[:,5]**2)
#    
#            r12 = np.sqrt((x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
#    
#            psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
#                    * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
#            return psisq
#    
#        def main():
#            # assign integration volume to integrator
#            dims = 6
#            # creates symmetric bounds specified by [-bound, bound] in dims dimensions
#            symm_bounds = dims * [[-bound,bound]]
#    
#            # simultaneously initialises expectation and normalisation integrals
#            expinteg = vegas.Integrator(symm_bounds)
#            norminteg = vegas.Integrator(symm_bounds)
#    
#            # adapt to the integrands; discard results
#            expinteg(expec, nitn=5, neval=1000)
#            norminteg(norm,nitn=5, neval=1000)
#            # do the final integrals
#            expresult = expinteg(expec, nitn=10, neval=100000)
#            normresult = norminteg(norm, nitn=10, neval=100000)
#    
#                ### obtain numerical result
#                ### Different expressions for plotting/ minimisation algorithm
#            if not OPTIM:
#                    E = expresult.mean/normresult.mean
#            else:
#                E = expresult[0].mean/ normresult[0].mean
#
#            final_results['energy'] = E
#            final_results['dev'] = [expresult.sdev, normresult.sdev]
#            final_results['pvalue'] = [expresult.Q, normresult.Q]
#            print(E)
#            return E
#        E = main()
#        return E
#        
#    for i in range(N):
#        print(j, i)
#        start_time = time.time()
#        result = fmin(evalenergy, [0.15], ftol = 0.01, xtol = 0.001, full_output=1, disp=0)
#        benergy.append(result[1])
#        bopt.append(float(result[0]))
#        print("--- Iteration time: %s seconds ---" % (time.time() - start_time))
#    B_energy_2[:,j] = benergy
#    B_alpha_2[:,j] = bopt
#    print(np.mean(benergy))
#
#name = 'vegas_bound_energy_comp_2'
#np.savetxt('%s.csv' %name, B_energy_2, 
#           fmt = '%s', delimiter=',', header = (name + str(bounds)))
#
#name = 'vegas_bound_ealpha_comp_2'
#np.savetxt('%s.csv' %name, B_alpha_2, 
#           fmt = '%s', delimiter=',', header = (name + str(bounds)))
#
#
#bounds = [1.5, 1.6, 1.7,1.8, 1.9]
#B_energy_mean = np.mean(B_energy_2, 0)
#B_alpha_mean = np.mean(B_alpha_2, 0)
#B_energy_std = np.std(B_energy_2, 0)
#B_alpha_std = np.std(B_alpha_2, 0)
#
#B_processed_2 = np.transpose([bounds, B_energy_mean, B_alpha_mean, B_energy_std, B_alpha_std])
#name = 'vegas_bound_processed_2'
#np.savetxt('%s.csv' %name, B_processed_2, 
#           fmt = '%s', delimiter=',', header = (name +'bound_energy_alpha_estd_astd'))

#%%

iters = [5e4, 1e5, 5e5, 1e6]
#twopara_energy = np.zeros((10, 4))
twopara_energy_backup = backup.copy()
#%%
N = 10

indx = 3

start_time = time.time()
twop_energy = []
twop_opt =[]

def main(alpha):
    print(alpha)
    global alpha0
    global alpha1
#    global alpha2
    alpha0 = alpha[0]
    alpha1 = alpha[1]
#    alpha2 = alpha[2]

    start_time = time.time()

    # assign integration volume to integrator
    bound = 8
    dims = 6
    # creates symmetric bounds specified by [-bound, bound] in dims dimensions
    symm_bounds = dims * [[-bound,bound]]

    # simultaneously initialises expectation and normalisation integrals
    expinteg = vegas.Integrator(symm_bounds)
    norminteg = vegas.Integrator(symm_bounds)

    # adapt to the integrands; discard results
    expinteg(expec, nitn=5, neval=1000)
    norminteg(norm,nitn=5, neval=1000)
    # do the final integrals
    expresult = expinteg(expec, nitn=10, neval=iters[indx])
    normresult = norminteg(norm, nitn=10, neval=iters[indx])

    if not OPTIM:
        E = expresult.mean/normresult.mean
    else:
        E = expresult[0].mean/ normresult[0].mean
#    print('Energy is {} when alpha is {}'.format(E, alpha), ' with sdev = ', [expresult.sdev, normresult.sdev])


    final_results['energy'] = E
    final_results['dev'] = [expresult.sdev, normresult.sdev]
    final_results['pvalue'] = [expresult.Q, normresult.Q]
#        print(E)
    print("--- Iteration time: %s seconds ---" % (time.time() - start_time))
    return E
    
for i in range(N):
    print(indx, i)
#    start_time = time.time()
    result = fmin(main, [2, 0.15], ftol = 0.1, xtol = 0.01, full_output=1, disp=1)
    twop_energy.append(result[1])
    
dur = time.time() - start_time
print("--- Iteration time: %s seconds ---" % (dur))
twopara_energy[:,indx] = twop_energy

#%%
##########
### Save data into a csv
#n = '1e5'
name = 'vegas_2param_iters'
np.savetxt('%s.csv' %name, (twopara_energy_backup), 
           fmt = '%s', delimiter=',', header = (name))

