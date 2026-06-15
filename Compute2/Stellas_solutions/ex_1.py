#!/usr/bin/env python3
"""
Exercise 1 for compute 2. 
"""

from __future__ import annotations
from mcgauss import mcgauss, mcgauss_uniform, mcgauss_adaptive
from puwr import tauint

import numpy as np
import matplotlib.pyplot as plt


def main(mc: callable) -> None:
    """Method to run a chosen Metropolis sampler for a 1D standard Gaussian target N(0, 1)
    for various window sizes, compute the corresponding IACTs, and plot window size vs IACT.
    
    Args:
        mc (callable): Metropolis sampler function to use (e.g. mcgauss or mcgauss_uniform).
    """
    mu = 0.0
    sig = 1.0
    x0 = 0.0
    ws = 0.01 * 2.0 ** np.arange(12)
    n = 1_100_000
    burn_in = 100_000

    rng = np.random.default_rng(0)
    
    iacts = []
    iact_sds = []

    for w in ws:
        chain = mc(mu, sig, x0, n, w, rng=rng)
        samples = chain[burn_in:]
        mean, mean_sd, tint, tint_sd = tauint([[samples]], 0)
        iacts.append(float(2*tint))
        iact_sds.append(float(2*tint_sd))
    iacts = np.array(iacts)
    
    method = "(normal proposal)" if mc is mcgauss else "(uniform proposal)"

    # plot window size vs IACT
    plt.errorbar(ws, iacts, yerr=iact_sds, fmt="o", capsize=5)
    plt.title(f"Window size vs IACT {method}\n(x0={x0:g}, n={n}, burn_in={burn_in})")
    plt.xlabel("window size")
    plt.ylabel(r"IACT $\pm$ IACT SD")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    # print window sizes and IACTs
    for w, iact, iact_sd in zip(ws, iacts, iact_sds):
        print(rf"w={w:g}: IACT={iact:.4f} $\pm$ {iact_sd:.4f}")
        

def part_c() -> None:
    """Method to run an adaptive Metropolis sampler for a 1D standard Gaussian target N(0, 1),
    plot the proposal standard deviations, and print the proposal standard deviation near the end and IACT.
    """
    mu = 0.0
    sig = 1.0
    x0 = 0.0
    w0 = 1.0
    n = 1_100_000
    burn_in = 100_000

    rng = np.random.default_rng(0)
    
    chain, w = mcgauss_adaptive(mu, sig, x0, n, w0, rng=rng)
    samples = chain[burn_in:]
    mean, mean_sd, tint, tint_sd = tauint([[samples]], 0)
    iact = float(2*tint)
    iact_sd = float(2*tint_sd)

    # plot proposal standard deviations
    plt.plot(w)
    plt.title(f"Adaptive Metropolis proposal standard deviations\n(x0={x0:g}, w0={w0:g}, n={n}, burn_in={burn_in})")
    plt.xlabel("iteration")
    plt.ylabel("proposal standard deviation")
    plt.yscale("log")
    plt.show()
    
    # print proposal standard deviation near end and IACT
    w_tail = w[int(0.9 * len(w)):]
    w_tail_mean = float(np.mean(w_tail))
    w_tail_sd = float(np.std(w_tail, ddof=1))
    print(rf"Proposal standard deviation near end={w_tail_mean:.4f} $\pm$ {w_tail_sd:.4f}")
    print(rf"IACT={iact:.4f} $\pm$ {iact_sd:.4f}")


if __name__ == "__main__":
    # main(mc=mcgauss)
    # main(mc=mcgauss_uniform)
    part_c()



