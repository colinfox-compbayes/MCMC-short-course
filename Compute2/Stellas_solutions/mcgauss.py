#!/usr/bin/env python3
"""
Implements various Metropolis samplers for a 1D Gaussian target N(mu, sig^2).
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


def mcgauss_uniform(mu: float, sig: float, x0: float, n: int, w: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Metropolis sampler for a 1D Gaussian target N(mu, sig^2) with uniform proposal.

    Args:
        mu (float): Mean of the Gaussian target.
        sig (float): Standard deviation of the Gaussian target.
        x0 (float): Initial state.
        n (int): Number of samples to draw.
        w (float): Proposal standard deviation, e.g. proposal is w * U(-sqrt(12)/2, sqrt(12)/2).
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
    sqrt_12_div_2 = np.sqrt(12) / 2.0
            
    for k in range(1, n + 1):
        x_current = x[k - 1]
        x_proposal = x_current + w * sqrt_12_div_2 * rng.uniform(-1.0, 1.0)
        
        # log(target(x_proposal)) - log(target(x_current))
        log_ratio = (- (x_proposal - mu) ** 2 + (x_current - mu) ** 2) * inv_2_sig_square
        # accept with probability min(1, exp(log_ratio))
        if np.log(rng.uniform()) < np.minimum(0.0, log_ratio):
            x[k] = x_proposal
        else:
            x[k] = x_current

    return x


def mcgauss_adaptive(mu: float, sig: float, x0: float, n: int, w0: float, *,
                    adapt_start: int = 1_000, adapt_interval: int = 10, eps: float = 1e-8, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive Metropolis sampler for a 1D Gaussian target N(mu, sig^2) with normal proposal.
    That is, the proposal is w_k * N(0,1), where w_k is adapted from the running variance of the chain:
    w_k^2 = 2.4^2 * Var_hat + eps.

    Args:
        mu (float): Mean of the Gaussian target.
        sig (float): Standard deviation of the Gaussian target.
        x0 (float): Initial state.
        n (int): Number of samples to draw.
        w0 (float): Initial proposal standard deviation.
        adapt_start (int): Number of initial iterations to run before starting adaptation.
        adapt_interval (int): Number of iterations between adaptations.
        eps (float): Small constant to add to the adapted proposal variance for numerical stability.
        rng (np.random.Generator | None): Random number generator.

    Returns:
        np.ndarray: Array of samples including the initial state at index 0.
        np.ndarray: Array of proposal standard deviations including the initial standard deviation at index 0.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if sig <= 0:
        raise ValueError("sig must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if w0 <= 0:
        raise ValueError("w0 must be > 0")

    rng = np.random.default_rng() if rng is None else rng

    x = np.empty(n + 1, dtype=float)
    x[0] = x0
    
    w = np.empty(n + 1, dtype=float)
    w[0] = w0
    w_current = float(w0)

    inv_2_sig_square = 1.0 / (2.0 * sig * sig)
    scale_square = 2.4 ** 2

    for k in range(1, n + 1):
        x_current = x[k - 1]
        x_proposal = x_current + w_current * rng.normal()
        
        # log(target(x_proposal)) - log(target(x_current))
        log_ratio = (- (x_proposal - mu) ** 2 + (x_current - mu) ** 2) * inv_2_sig_square
        # accept with probability min(1, exp(log_ratio))
        if np.log(rng.uniform()) < np.minimum(0.0, log_ratio):
            x[k] = x_proposal
        else:
            x[k] = x_current

        # adapt proposal scale every adapt_interval steps after adapt_start
        if k >= adapt_start and (k - adapt_start) % adapt_interval == 0:
            var_hat = np.var(x[:k+1], ddof=1) # could do this more effciently with a running variance computation...
            w_current = float(np.sqrt(scale_square * max(var_hat, 0.0) + eps))
        w[k] = w_current

    return x, w