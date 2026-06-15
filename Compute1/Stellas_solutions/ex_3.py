#!/usr/bin/env python3
"""
Exercise 3 for compute 1.
"""

from __future__ import annotations
from mcexp import mcexp, mcexp_unstable

import numpy as np
import matplotlib.pyplot as plt

    
def main(x0: float, mc: callable) -> None:
    """Method to run a chosen Random-walk Metropolis sampler for a Exponential target Exp(lambda)
    starting from initial state x0 and plot the trace and normalized histogram of samples.
    
    Args:
        x0 (float): Initial state for the sampler.
        mc (callable): Random-walk Metropolis sampler function to use (e.g. mcexp or mcexp_unstable).
    """
    lambda_ = 1.0
    w = 1.0
    n = 50_000
    burn_in = 10_000
    
    rng = np.random.default_rng(0)

    chain = mc(lambda_, x0, n, w, rng=rng)
    samples = chain[burn_in:]
    
    _, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    unstable = "(unstable)" if mc is mcexp_unstable else ""
    
    # plot trace of samples
    ax_trace = axes[0]
    ax_trace.plot(samples, linewidth=1)
    ax_trace.set_title(f"Trace of samples {unstable}\n(x0={x0:g}, w={w:g}, n={n}, burn_in={burn_in})")
    ax_trace.set_xlabel("iteration")
    ax_trace.set_ylabel("x")

    # plot normalized histogram of samples
    ax_hist = axes[1]
    ax_hist.hist(samples, bins=60, density=True, alpha=0.6)
    ax_hist.set_title(f"Normalized histogram of samples {unstable}\n(x0={x0:g}, w={w:g}, n={n}, burn_in={burn_in})")
    ax_hist.set_xlabel("x")
    ax_hist.set_ylabel("density")
    plt.show()


if __name__ == "__main__":
    # main(x0=1.0, mc=mcexp)
    # main(x0=1000.0, mc=mcexp)
    main(x0=1000.0, mc=mcexp_unstable)