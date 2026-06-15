#!/usr/bin/env python3
"""
Implements a Metropolis sampler for a Geometric target Geom(p).
"""

from __future__ import annotations

import numpy as np


def mcgeom(p: float, x0: int, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Metropolis sampler for a Geometric target Geom(p) with Bernoulli(0.5) 
    proposal to move up or down by 1. Note that we take the convention that 
    Geom(p) has pmf p * (1-p)^k for k >= 0.

    Args:
        p (float): Success probability of the Geometric target.
        x0 (int): Initial state.
        n (int): Number of samples to draw.
        rng (np.random.Generator | None): Random number generator.

    Returns:
        np.ndarray: Array of samples including the initial state at index 0.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if p <= 0 or p > 1:
        raise ValueError("p must be in (0, 1]")
    if x0 < 0:
        raise ValueError("x0 must be >= 0")
    if n < 0:
        raise ValueError("n must be >= 0")

    rng = np.random.default_rng() if rng is None else rng

    x = np.empty(n + 1, dtype=int)
    x[0] = x0
    
    log_1_minus_p = np.log1p(-p)

    for k in range(1, n + 1):
        x_current = x[k - 1]
        x_proposal = x_current + (2 * rng.integers(2) - 1)
        
        # target is 0 for x < 0, so always reject
        if x_proposal < 0:
            x[k] = x_current
            continue

        # log(target(x_proposal)) - log(target(x_current))
        log_ratio = (x_proposal - x_current) * log_1_minus_p
        # accept with probability min(1, exp(log_ratio))
        if np.log(rng.uniform()) < np.minimum(0.0, log_ratio):
            x[k] = x_proposal
        else:
            x[k] = x_current

    return x