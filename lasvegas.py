#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vegas example from the Basic Integrals section
of the Tutorial and Overview section of the
vegas documentation.
"""
from __future__ import print_function   # makes this work for python2 and 3

import vegas
import math
import numpy as np
import sys

if sys.argv[1:]:
    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_GRID = True

alpha = 0.2

def f(x):
    r1 = np.array(x[0:3])
    r2 = np.array(x[3:])
    
    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)

    r1_hat = r1 / r1_len
    r2_hat = r2 / r2_len

    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)

    EL =  (-4 + np.dot(r1_hat - r2_hat, r1 - r2 ) / (r12 * (1+alpha*r12)**2)
            - 1/ (r12*(1 + alpha*r12)**3)
            - 1/ (4*(1 + alpha*r12)**4)
            + 1/ r12 )
    
    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
            * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
    
    return psisq * EL

def psisq(x):

    r1_len = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    r2_len = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
    
    r12 = np.sqrt((x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2)
    
    psisq = (np.exp(-2 * r1_len)* np.exp(-2 * r2_len)
        * np.exp(r12 / (2 * (1+ alpha*r12)))) ** 2
    
    return psisq

def same_range(bounds, dims):
    return np.full((dims,2), bounds)

def main():
    # seed the random number generator so results reproducible
    np.random.seed((1, 2, 3, 4, 5, 6))

    # assign integration volume to integrator
    bounds = np.array([-10,10])
    integ = vegas.Integrator(same_range(bounds,6))
    
    norminteg = vegas.Integrator(same_range(bounds,6))
    
    # adapt to the integrand; discard results
    integ(f, nitn=5, neval=10000,)
    norminteg(psisq,nitn=5, neval=10000)

    # do the final integral
    result = integ(f, nitn=10, neval=100000)
    resultnorm = integ(psisq, nitn=10, neval=100000)
    
    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))
    if SHOW_GRID:
        integ.map.show_grid(20)
    
    print(resultnorm.summary())
    print('result = %s    Q = %.2f' % (resultnorm, result.Q))
    if SHOW_GRID:
        integ.map.show_grid(20)
    print(result/resultnorm)
    
if __name__ == '__main__':
    main()