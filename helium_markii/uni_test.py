#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:19:26 2020

@author: kenton
"""

import optipack, psiham

psilet_args = {'electrons': 2, 'alphas': 2, 'coordsys': 'cartesian'}
ham = psiham.HamLet(trial_expr = 'twopara1', **psilet_args)
variables, expr = ham.he_expect()
temp1 = ham.numbafy(expr, coordinates = variables, name = 'expec')

variables, expr = ham.he_norm()
temp2 = ham.numbafy(expr, coordinates = variables, name = 'norm')
