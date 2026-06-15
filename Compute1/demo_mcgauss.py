#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:35:42 2025

@author: colin
"""

import numpy as np
import matplotlib.pyplot as plt
from mcgauss import mcgauss  # assuming mcgauss.py is in the same directory

# --- Proposal window comparison ---
wins = [0.3, 3, 30] # list of window sizes 
chains = []         # list to hold each chain
for i, w in enumerate(wins, start=1):
    X = mcgauss(0, 1, 1, int(1e3), w)
    chains.append(X)  # store the chain
    plt.figure(1)
    plt.subplot(len(wins), 1, i)
    plt.plot(X)
    plt.title(f"Proposal window w = {w}")
plt.tight_layout()
plt.show()

input("Paused — press Enter to continue...")

# --- Equilibrium distribution ---
w = 3
X = mcgauss(0, 1, 1, int(1e5), w)
plt.figure(3)
plt.hist(X, bins=30, density=True, edgecolor='black')
plt.title(f"Normalised histogram of samples (w = {w})")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

input("Paused — press Enter to continue...")

# --- Some stats ---
print(f"Variance: {np.var(X):.4f}")
print(f"Mean:     {np.mean(X):.4f}")

def diagnostics(X):
    """
    Compute acceptance rate and expected jump size from sample trace X.
    """
    steps = np.diff(X)
    acceptance_rate = np.count_nonzero(steps) / len(steps)
    expected_jump_size = np.mean(np.abs(steps))
    return acceptance_rate, expected_jump_size

for i, w in enumerate(wins):
    a,s = diagnostics(chains[i])
    print(f"w={wins[i]}: acceptance_rate = {a:.4f}  expected_jump_size = {s:.4f}")

