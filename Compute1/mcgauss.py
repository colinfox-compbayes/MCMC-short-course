#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:48:32 2025

@author: colin
"""

import numpy as np

def mcgauss(mu, sig, x0, N, w):
    """
    Return N samples from Normal(mu, sig^2) using 
    Random Walk Metropolis (RWM) with proposal window w, starting at x0.
    
    Parameters:
    - mu: mean of target distribution
    - sig: standard deviation of target distribution
    - x0: initial value
    - N: number of samples
    - w: proposal window (standard deviation of Normal proposal distribution)
    
    Returns:
    - X: array of samples
    """
    X = np.zeros(N)
    X[-1] = x0

    for k in range(N):
        # Normal proposal: standard deviation w
        xp = X[k-1] + w * np.random.randn()
        
        # Uncomment for Uniform proposal: width w
        # xp = X[k-1] + w * (2.*np.random.uniform() -1.)

        # Metropolis acceptance probability
        alpha = min(1, np.exp((-(xp - mu)**2 + (X[k-1] - mu)**2) / (2 * sig**2)))
        
        X[k] = xp if np.random.rand() < alpha else X[k-1] # accept-reject step

    return X

# Example usage (runs when this file is executed directly, 
#                but is not run when imported as: import mcgauss)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Parameters
    mu = 0.
    sig = 1.
    x0 = 1.
    N = int(1e4)
    w = 3.

    # Run RWM sampler
    X = mcgauss(mu, sig, x0, N, w)

    # Plot histogram
    plt.hist(X, bins=30, density=True, edgecolor='black')
    plt.title("Normalized Histogram of RWM Samples")
    plt.xlabel("Variable")
    plt.ylabel("Estimated Density Function")
    plt.show()
    
    # Plot trace
    plt.plot(X)
    plt.title("MCMC Trace")
    plt.xlabel("iteration")
    plt.ylabel("Variable")
    plt.show()

    # Print statistics
    print(f"Mean: {np.mean(X):.4f}")
    print(f"Variance: {np.var(X):.4f}")
    
    # Plot autocorrelation
    import pandas as pd
    fig, ax = plt.subplots()

    pd.plotting.autocorrelation_plot(X,ax) 
    ax.set_xlim(0,50)
    # ax.set_ylim(-0.1,1)
    # plt.plot(X)
    # plt.title("MCMC Trace")
    # plt.xlabel("iteration")
    # plt.ylabel("Variable")
    plt.show()
