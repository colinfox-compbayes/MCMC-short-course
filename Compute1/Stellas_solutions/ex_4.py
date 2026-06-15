#!/usr/bin/env python3
"""
Exercise 4 for compute 1.
"""

from __future__ import annotations
from mcgeom import mcgeom

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    """Method to run a Random-walk Metropolis sampler for a Geometric target Geom(p)
    with and plot the trace and normalized histogram of samples.
    """
    p = 0.25
    x0 = 0
    n = 1_000_000
    burn_in = 0
    
    rng = np.random.default_rng(0)

    chain =  mcgeom(p, x0, n, rng=rng)
    samples = chain[burn_in:]
    
    _, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    # plot trace of samples
    ax_trace = axes[0]
    ax_trace.plot(samples, linewidth=1)
    ax_trace.set_title(f"Trace of samples\n(p={p:g}, x0={x0:g}, n={n}, burn_in={burn_in})")
    ax_trace.set_xlabel("iteration")
    ax_trace.set_ylabel("x")

    # plot normalized histogram of samples
    ax_hist = axes[1]
    ax_hist.hist(samples, bins=60, density=True, alpha=0.6)
    ax_hist.set_title(f"Normalized histogram of samples \n(p={p:g}, x0={x0:g}, n={n}, burn_in={burn_in})")
    ax_hist.set_xlabel("x")
    ax_hist.set_ylabel("density")
    plt.show()

    # print sample mean vs true mean
    sample_mean = float(np.mean(samples))
    true_mean = (1.0 - p) / p
    print(f"sample mean={sample_mean:.4f}, true mean={(true_mean):.4f}")


if __name__ == "__main__":
    main()