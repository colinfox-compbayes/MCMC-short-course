#!/usr/bin/env python3
"""
Implements a Random-walk Metropolis sampler for a Exponential target Exp(lambda).
"""

from __future__ import annotations

import numpy as np


def mcexp(lambda_: float, x0: float, n: int, w: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random-walk Metropolis sampler for a Exponential target Exp(lambda).

    Args:
        lambda_ (float): Rate parameter of the Exponential target.
        x0 (float): Initial state.
        n (int): Number of samples to draw.
        w (float): Proposal standard deviation.
        rng (np.random.Generator | None): Random number generator.

    Returns:
        np.ndarray: Array of samples including the initial state at index 0.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if lambda_ <= 0:
        raise ValueError("lambda_ must be > 0")
    if x0 < 0:
        raise ValueError("x0 must be >= 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if w <= 0:
        raise ValueError("w must be > 0")

    rng = np.random.default_rng() if rng is None else rng

    x = np.empty(n + 1, dtype=float)
    x[0] = x0

    for k in range(1, n + 1):
        x_current = x[k - 1]
        x_proposal = x_current + w * rng.normal()
        
        # target is 0 for x < 0, so always reject
        if x_proposal < 0.0:
            x[k] = x_current
            continue

        # log(target(x_proposal)) - log(target(x_current))
        log_ratio = - lambda_ * (x_proposal - x_current)
        # accept with probability min(1, exp(log_ratio))
        if np.log(rng.uniform()) < np.minimum(0.0, log_ratio):
            x[k] = x_proposal
        else:
            x[k] = x_current

    return x


def mcexp_unstable(lambda_: float, x0: float, n: int, w: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random-walk Metropolis sampler for a Exponential target Exp(lambda).
    This implementation uses the naive numerically unstable calculation for the acceptance ratio.

    Args:
        lambda_ (float): Rate parameter of the Exponential target.
        x0 (float): Initial state.
        n (int): Number of samples to draw.
        w (float): Proposal standard deviation.
        rng (np.random.Generator | None): Random number generator.

    Returns:
        np.ndarray: Array of samples including the initial state at index 0.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if lambda_ <= 0:
        raise ValueError("lambda_ must be > 0")
    if x0 < 0:
        raise ValueError("x0 must be >= 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if w <= 0:
        raise ValueError("w must be > 0")

    rng = np.random.default_rng() if rng is None else rng

    x = np.empty(n + 1, dtype=float)
    x[0] = x0
            
    for k in range(1, n + 1):
        x_current = x[k - 1]
        x_proposal = x_current + w * rng.normal()
        
        # target is 0 for x < 0, so always reject
        if x_proposal < 0.0:
            x[k] = x_current
            continue

        # target(x_proposal) / target(x_current)
        ratio =  np.exp(-lambda_ * x_proposal) / np.exp(-lambda_ * x_current)
        # accept with probability min(1, exp(ratio))
        if rng.uniform() < np.minimum(1.0, ratio):
            x[k] = x_proposal
        else:
            x[k] = x_current

    return x