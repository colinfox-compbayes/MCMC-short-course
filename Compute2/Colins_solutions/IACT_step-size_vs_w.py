#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 19:37:33 2025

@author: colin
"""

import numpy as np
import matplotlib.pyplot as plt
from mcgauss import mcgauss  # assuming mcgauss.py is in the same directory
from puwr import tauint      # requires package py-uwerr

def diagnostics(X):
    """
    Compute acceptance rate and expected jump size from sample trace X.
    """
    steps = np.diff(X)
    acceptance_rate = np.count_nonzero(steps) / len(steps)
    # expected_jump_size = np.mean(np.abs(steps))
    expected_jump_size = np.sum(np.abs(steps))  / len(steps)
    return acceptance_rate, expected_jump_size

num_w = 20
n_samp = int(1e5)
w_vec = np.logspace(np.log10(0.1),np.log10(30),num_w)
ex_ss_vec = np.zeros(num_w)
a_rate_vec = np.zeros(num_w)
iacts = np.zeros(num_w)

for i, w in enumerate(w_vec):
    X = mcgauss(0, 1, 1, n_samp, w)
    a,s = diagnostics(X)
    ex_ss_vec[i] = s
    a_rate_vec[i] = a
    mean, delta, tint, d_tint = tauint([[X]], 0)
    iacts[i] = tint
    
plt.semilogx(w_vec,ex_ss_vec,label="mean step size")
plt.semilogx(w_vec,a_rate_vec,label="aceptance rate")
plt.semilogx(w_vec,1/iacts,label="1/IACT")
plt.semilogx(w_vec,w_vec*a_rate_vec,label="acc*step")
plt.grid(True)
plt.title("acceptance rate, mean step size, IACT as function of proposal window")
plt.legend()

plt.xlabel("proposal std-dev/target std-dev")
plt.show()
