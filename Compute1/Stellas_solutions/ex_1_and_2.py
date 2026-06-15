#!/usr/bin/env python3
"""
Exercise 1 and 2 for compute 1.
"""

from __future__ import annotations
from mcgauss import mcgauss

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    """Method to run a Random-walk Metropolis sampler for a 1D standard Gaussian target N(0, 1) 
    with various proposal widths w and plot the trace and normalized histogram of samples.
    """
    mu = 0.0
    sig = 1.0
    x0 = 0.0
    ws = (0.3, 3.0, 30.0)
    n = 10_000
    burn_in = 0

    rng = np.random.default_rng(0)
    _, axes = plt.subplots(len(ws), 2, figsize=(10, 8), constrained_layout=True)

    for i, w in enumerate(ws):
        chain = mcgauss(mu, sig, x0, n, w, rng=rng)
        samples = chain[burn_in:]

        # plot trace of samples
        ax_trace = axes[i, 0]
        ax_trace.plot(samples, linewidth=1)
        ax_trace.set_title(f"Trace of samples\n(x0={x0:g}, w={w:g}, n={n}, burn_in={burn_in})")
        ax_trace.set_xlabel("iteration")
        ax_trace.set_ylabel("x")

        # plot normalized histogram of samples
        ax_hist = axes[i, 1]
        ax_hist.hist(samples, bins=60, density=True, alpha=0.6)
        ax_hist.set_title(f"Normalized histogram of samples\n(x0={x0:g}, w={w:g}, n={n}, burn_in={burn_in})")
        ax_hist.set_xlabel("x")
        ax_hist.set_ylabel("density")

    plt.show()


if __name__ == "__main__":
    main()