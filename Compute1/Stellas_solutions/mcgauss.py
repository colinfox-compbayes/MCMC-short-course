#!/usr/bin/env python3
"""
Implements a Random-walk Metropolis sampler for a 1D Gaussian target N(mu, sig^2).
"""

from __future__ import annotations

import numpy as np


def mcgauss(mu: float, sig: float, x0: float, n: int, w: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random-walk Metropolis sampler for a 1D Gaussian target N(mu, sig^2).

    Args:
        mu (float): Mean of the Gaussian target.
        sig (float): Standard deviation of the Gaussian target.
        x0 (float): Initial state.
        n (int): Number of samples to draw.
        w (float): Proposal standard deviation.
        rng (np.random.Generator | None): Random number generator.

    Returns:
        np.ndarray: Array of samples including the initial state at index 0.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if sig <= 0:
        raise ValueError("sig must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if w <= 0:
        raise ValueError("w must be > 0")

    rng = np.random.default_rng() if rng is None else rng

    x = np.empty(n + 1, dtype=float)
    x[0] = x0

    inv_2_sig_square = 1.0 / (2.0 * sig * sig)

    for k in range(1, n + 1):
        x_current = x[k - 1]
        x_proposal = x_current + w * rng.normal()
        
        # log(target(x_proposal)) - log(target(x_current))
        log_ratio = (- (x_proposal - mu) ** 2 + (x_current - mu) ** 2) * inv_2_sig_square
        # accept with probability min(1, exp(log_ratio))
        if np.log(rng.uniform()) < np.minimum(0.0, log_ratio):
            x[k] = x_proposal
        else:
            x[k] = x_current

    return x