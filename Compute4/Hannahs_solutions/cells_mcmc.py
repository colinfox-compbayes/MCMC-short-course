"""
cells_mcmc.py
===================

Metropolis-Hastings / RJMCMC for the Compute 4 “cells” inverse problem.

This script is written to match the *model assumptions in your notes* exactly:

Observed (data) model
---------------------
Let y be the observed image (npix x npix), with N = npix^2 pixels.
We assume an additive i.i.d. Gaussian noise model:

    y = A(x) + eps,    eps ~ N(0, sigma^2 I_N)

where A(x) is the *exact* noiseless renderer implemented in cells_render.py:

    A(x) = render_noiseless_from_latent(latent_x, RenderParams(...))

Prior model (marked spatial Poisson point process with order)
-------------------------------------------------------------
Let B = {1,2,...,npix} x {1,2,...,npix} be the discrete pixel-centre grid.
The latent state is an ordered marked point pattern:

    x = [(x_1, y_1, m_1), ..., (x_n, y_n, m_n)]

stored as a Python list, in BACK-to-FRONT order (later entries occlude earlier).

Priors:
    n ~ Poisson(lambda_n)                          (lambda_n fixed, default 8.0)
    (x_i, y_i) | n  iid Uniform(B)                 (discrete uniform over npix^2 sites)
    m_i | n      iid Uniform({0,1})                (P(m_i=0)=P(m_i=1)=1/2)
    order | n    Uniform over permutations         (pi(order|n) = 1/n!)

Hence:
    pi(x) = [lambda^n e^{-lambda} / n!] * (1/|B|)^n * (1/2)^n * (1/n!)
          = lambda^n e^{-lambda} / (n!)^2 / (|B|^n 2^n)

MCMC moves
----------
The chain uses four proposal kernels (matching the MATLAB-style template):
  1) Birth/death (RJMCMC)  : add/remove a cell (with order insertion/deletion)
  2) Flip label            : swap a cell's mark m in {0,1}
  3) Move point            : translate a cell centre within a window (with Hastings correction near boundaries)
  4) Swap two points       : swap two entries in the ordered list (changes occlusion ordering)

Command-line usage (typical)
----------------------------
    python cells_mcmc.py --y_npy outputs/slide.npy --nsamp 20000 --sigma 0.1 --seed 0 --out outputs/mcmc_run

Notes
-----
- This file depends on cells_render.py for the forward model A(x) and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from scipy.io import loadmat
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio


from cells_render import (
    RenderParams,
    render_noiseless_from_latent,
    validate_latent,
)

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Cell = Tuple[int, int, int]  # (x, y, m), 1-based coords, m in {0,1}


# -----------------------------
# Parameters (lambda fixed at 8)
# -----------------------------
@dataclass(frozen=True)
class MCMCParams:
    """
    Container for model/proposal parameters.

    Attributes
    ----------
    npix : int
        Image width/height in pixels (image is npix x npix).
    lambda_n : float
        Poisson mean for the prior on the number of cells n.
    sigma : float
        Noise standard deviation in the Gaussian likelihood N(A(x), sigma^2 I).

    move_window : int
        Half-width of the square translation window for simh3_move_point.
        The proposed (x',y') is uniform over the clipped window around (x,y).
    move_ratio : Tuple[float, float, float, float]
        Relative weights for choosing kernels 1..4 in choose_kernel().
        Interpreted as probabilities after normalization.

    render : RenderParams
        Rendering parameters used by cells_render.render_noiseless_from_latent.
    """

    npix: int = 100
    lambda_n: float = 8.0     # fixed by notes
    sigma: float = 0.1        # likelihood noise stddev

    move_window: int = 5
    move_ratio: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    render: RenderParams = RenderParams()

# -----------------------------
# Prior and likelihood (MATCH NOTES)
# -----------------------------

# =============================================================================
# logprior
# =============================================================================
def logprior(X: Sequence[Cell], par: MCMCParams) -> float:
    """
    Compute the exact log-prior log pi(x) implied by the notes.

    The latent state x is the ordered marked point pattern X, with n = len(X).
    The prior factorizes as:

        pi(x) = pi(n) * Π_i pi(r_i | n) * Π_i pi(m_i | n) * pi(order | n)

    where
      pi(n)         = lambda^n e^{-lambda} / n!
      pi(r_i | n)   = 1/|B| = 1/(npix^2)          (discrete uniform pixel centres)
      pi(m_i | n)   = 1/2                         (marks uniform on {0,1})
      pi(order | n) = 1/n!                        (uniform over permutations)

    Hence:
        log pi(x)
          = n log lambda - lambda - log(n!)
            - n log(npix^2) - n log 2 - log(n!)
          = n log lambda - lambda - 2 log(n!)
            - 2n log npix - n log 2

    Parameters
    ----------
    X : Sequence[Cell]
        Ordered list/sequence of cells, each (x, y, m) with:
          - x, y in {1,2,...,npix} (1-based integer pixel-centre coordinates)
          - m in {0,1} (categorical mark)
        Order is BACK-to-FRONT (later cells occlude earlier).
    par : MCMCParams
        Parameter bundle containing npix and lambda_n.

    Returns
    -------
    float
        The log prior density/mass log pi(x). Returns -np.inf if:
          - X fails validate_latent (out of support),
          - lambda_n <= 0.
    """
    n = len(X)

    # validate coords/marks in correct support
    try:
        validate_latent(X, par.npix)
    except ValueError:
        return -np.inf

    lam = float(par.lambda_n)
    if lam <= 0.0:
        return -np.inf

    log_n_fact = math.lgamma(n + 1.0)  # log(n!)
    lp = n * math.log(lam) - lam
    lp += -2.0 * log_n_fact
    lp += -2.0 * n * math.log(par.npix)  # log(npix^2) = 2 log(npix)
    lp += -n * math.log(2.0)             # marks uniform on {0,1}

    return float(lp)


# =============================================================================
# loglike
# =============================================================================
def loglike(X: Sequence[Cell], y: np.ndarray, par: MCMCParams) -> float:
    """
    Compute the exact Gaussian log-likelihood log p(y | x).

    Likelihood model (pixel-wise i.i.d. Gaussian):
        y | x ~ N(A(x), sigma^2 I_N),    N = npix^2

    So:
        p(y|x) = (2 pi sigma^2)^(-N/2) * exp( -||y - A(x)||^2 / (2 sigma^2) )

    This function returns the FULL log-likelihood including the normalizing
    constant (i.e., not just proportional-to terms).

    Parameters
    ----------
    X : Sequence[Cell]
        Latent ordered marked point pattern (see logprior docstring).
    y : np.ndarray
        Observed image array of shape (npix, npix). Values are typically in [0,1]
        for the provided data, but no clipping is enforced here.
    par : MCMCParams
        Must include sigma and render parameters (npix is implied by y size and
        RenderParams).

    Returns
    -------
    float
        The log-likelihood log p(y | x). Returns -np.inf if sigma <= 0.
    """
    sigma = float(par.sigma)
    if sigma <= 0.0:
        return -np.inf

    Ax = render_noiseless_from_latent(X, par.render)
    resid = (y - Ax).ravel()
    sse = float(np.dot(resid, resid))

    N = int(y.size)
    ll = -(N / 2.0) * math.log(2.0 * math.pi * sigma * sigma) - 0.5 * sse / (sigma * sigma)
    return float(ll)


# -----------------------------
# Proposals (Simh1..Simh4)
# Return (Xp, lh) where lh = log(q_reverse / q_forward)
# -----------------------------

# =============================================================================
# simh1_birth_death
# =============================================================================
def simh1_birth_death(X: List[Cell], rng: np.random.Generator, par: MCMCParams) -> Tuple[List[Cell], float]:
    """
    Birth/death reversible-jump move for changing the number of cells.

    Proposal design
    ---------------
    - If n == 0: must propose a birth.
    - Else: propose birth with prob 1/2 and death with prob 1/2.

    Birth proposal details
    ----------------------
    - New (x,y) is uniform on B = {1..npix}^2  => prob 1/(npix^2)
    - New mark m is uniform on {0,1}           => prob 1/2
    - Insertion position k is uniform among n+1 order slots => prob 1/(n+1)

    Death proposal details
    ----------------------
    - Choose an index k uniformly among n existing points => prob 1/n
      and remove it.

    Hastings correction
    -------------------
    Returns lh = log(q_reverse / q_forward), including the asymmetry at the
    n==0 boundary (and the reverse boundary when n-1==0).

    Parameters
    ----------
    X : List[Cell]
        Current latent state as an ordered list.
    rng : np.random.Generator
        NumPy random number generator.
    par : MCMCParams
        Contains npix (support size) used for uniform birth location.

    Returns
    -------
    Tuple[List[Cell], float]
        (Xp, lh) where:
          - Xp is the proposed new state (copied list),
          - lh is log(q_reverse / q_forward) for the MH accept ratio.
    """
    n = len(X)

    # birth probability
    if n == 0:
        p_birth = 1.0
    else:
        p_birth = 0.5
    p_death = 1.0 - p_birth

    do_birth = (rng.random() < p_birth)

    if do_birth:
        # propose new point
        x = int(rng.integers(1, par.npix + 1))
        y = int(rng.integers(1, par.npix + 1))
        m = int(rng.integers(0, 2))  # 0 or 1 with prob 1/2

        # insert into ordered list uniformly among n+1 slots
        k = int(rng.integers(0, n + 1))
        Xp = list(X)
        Xp.insert(k, (x, y, m))

        # q_forward
        log_q_f = math.log(p_birth)
        log_q_f += -2.0 * math.log(par.npix)  # 1/(npix^2)
        log_q_f += -math.log(2.0)             # mark 1/2
        log_q_f += -math.log(n + 1)           # insertion slot

        # reverse (from n+1): death prob is 1/2 (since n+1 > 0)
        n1 = n + 1
        p_birth_rev = 0.5
        p_death_rev = 0.5
        log_q_r = math.log(p_death_rev) + (-math.log(n1))  # choose removal index

        lh = log_q_r - log_q_f
        return Xp, float(lh)

    else:
        # death: remove one uniformly
        k = int(rng.integers(0, n))
        removed = X[k]
        Xp = list(X)
        Xp.pop(k)

        # q_forward
        log_q_f = math.log(p_death) + (-math.log(n))

        # reverse from n-1: birth prob depends on boundary
        n0 = n - 1
        if n0 == 0:
            p_birth_rev = 1.0
        else:
            p_birth_rev = 0.5

        x, y, m = removed
        # to get back exactly, must:
        #  - choose birth
        #  - choose same (x,y) and same mark
        #  - choose the same insertion slot k among n slots
        log_q_r = math.log(p_birth_rev)
        log_q_r += -2.0 * math.log(par.npix)
        log_q_r += -math.log(2.0)
        log_q_r += -math.log(n)  # because list length is n-1, insertion slots count is n

        lh = log_q_r - log_q_f
        return Xp, float(lh)


# =============================================================================
# simh2_flip_label
# =============================================================================
def simh2_flip_label(X: List[Cell], rng: np.random.Generator) -> Tuple[List[Cell], float]:
    """
    Flip-label move: choose one cell and flip its mark m -> 1-m.

    Parameters
    ----------
    X : List[Cell]
        Current latent state (ordered list).
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    Tuple[List[Cell], float]
        (Xp, lh) where:
          - Xp is the proposed state with one mark flipped (or unchanged if n==0),
          - lh = 0.0 because the proposal is symmetric.
    """
    n = len(X)
    if n == 0:
        return list(X), 0.0

    i = int(rng.integers(0, n))
    x, y, m = X[i]
    Xp = list(X)
    Xp[i] = (int(x), int(y), 1 - int(m))
    return Xp, 0.0  # symmetric

# =============================================================================
# simh3_move_point
# =============================================================================
def simh3_move_point(X: List[Cell], rng: np.random.Generator, par: MCMCParams) -> Tuple[List[Cell], float]:
    """
    Translate-move: randomly translate one cell centre within a window.

    The proposal chooses a cell i uniformly and proposes (x',y') uniformly
    over a clipped square window of half-width w around (x,y), where w is
    par.move_window.

    Because the window is clipped at image boundaries, the proposal is not
    strictly symmetric near boundaries; a Hastings correction is included:

        lh = log(q_reverse / q_forward)
           = log(|window_forward|) - log(|window_reverse|)

    Parameters
    ----------
    X : List[Cell]
        Current latent state (ordered list).
    rng : np.random.Generator
        NumPy random number generator.
    par : MCMCParams
        Contains npix (boundary) and move_window (window half-width).

    Returns
    -------
    Tuple[List[Cell], float]
        (Xp, lh) where:
          - Xp is the proposed state with one location moved (or unchanged if n==0),
          - lh is the boundary Hastings correction (0 in the interior).
    """
    n = len(X)
    if n == 0:
        return list(X), 0.0

    i = int(rng.integers(0, n))
    x, y, m = X[i]
    w = int(par.move_window)

    x_low = max(1, int(x) - w)
    x_high = min(par.npix, int(x) + w)
    y_low = max(1, int(y) - w)
    y_high = min(par.npix, int(y) + w)

    sx = x_high - x_low + 1
    sy = y_high - y_low + 1

    xp = int(rng.integers(x_low, x_high + 1))
    yp = int(rng.integers(y_low, y_high + 1))

    Xp = list(X)
    Xp[i] = (xp, yp, int(m))

    # Hastings correction for boundary window size changes
    x_low_r = max(1, xp - w)
    x_high_r = min(par.npix, xp + w)
    y_low_r = max(1, yp - w)
    y_high_r = min(par.npix, yp + w)
    sx_r = x_high_r - x_low_r + 1
    sy_r = y_high_r - y_low_r + 1

    lh = math.log(sx * sy) - math.log(sx_r * sy_r)
    return Xp, float(lh)


# =============================================================================
# simh4_swap_two
# =============================================================================
def simh4_swap_two(X: List[Cell], rng: np.random.Generator) -> Tuple[List[Cell], float]:
    """
    Permute/ordering move: swap two cells in the ordered list.

    This changes the occlusion ordering (front-to-back) while keeping the
    set of locations and marks unchanged.

    Parameters
    ----------
    X : List[Cell]
        Current latent state (ordered list).
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    Tuple[List[Cell], float]
        (Xp, lh) where:
          - Xp is the proposed state with two entries swapped (or unchanged if n<2),
          - lh = 0.0 because the proposal is symmetric.
    """
    n = len(X)
    if n < 2:
        return list(X), 0.0

    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n - 1))
    if j >= i:
        j += 1

    Xp = list(X)
    Xp[i], Xp[j] = Xp[j], Xp[i]
    return Xp, 0.0  # symmetric


# =============================================================================
# choose_kernel
# =============================================================================
def choose_kernel(rng: np.random.Generator, ratios: Tuple[float, float, float, float]) -> int:
    """
    Choose which proposal kernel to apply (1..4) based on relative weights.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random number generator.
    ratios : Tuple[float, float, float, float]
        Nonnegative relative weights for kernels (1,2,3,4). These are normalized
        internally to sum to 1.

    Returns
    -------
    int
        Kernel index in {1,2,3,4}.
    """
    r = np.array(ratios, dtype=float)
    r = r / float(np.sum(r))
    u = rng.random()
    return int(np.searchsorted(np.cumsum(r), u) + 1)  # 1..4


# =============================================================================
# run_mcmc
# =============================================================================
def run_mcmc(
    y: np.ndarray,
    nsamp: int,
    par: MCMCParams,
    seed: Optional[int] = None,
    plot_every: int = 0,
    plot_pause: float = 0.01,
) -> dict:
    """
    Run the MH/RJMCMC chain.

    The chain targets the posterior (up to normalization):

        pi(x | y) ∝ p(y | x) * pi(x)

    using four proposal kernels chosen by choose_kernel().

    Parameters
    ----------
    y : np.ndarray
        Observed image array (npix x npix).
    nsamp : int
        Number of MCMC iterations/samples to draw.
    par : MCMCParams
        Model and proposal parameters (npix, lambda_n, sigma, move_window, etc.).
    seed : Optional[int], default=None
        Random seed for reproducibility. If None, NumPy uses entropy from OS.

    Returns
    -------
    dict
        Dictionary containing chain traces and stored latent states:

        - "trace_n"   : np.ndarray shape (nsamp,), total number of cells n each iter
        - "trace_ll"  : np.ndarray shape (nsamp,), log-likelihood value each iter
        - "trace_lp"  : np.ndarray shape (nsamp,), log-prior value each iter
        - "trace_acc" : np.ndarray shape (nsamp,), 1 if accepted else 0
        - "trace_k"   : np.ndarray shape (nsamp,), kernel index used (1..4)
        - "chain_X"   : List[np.ndarray], each entry is (n,3) int array of (x,y,m)
    """
    rng = np.random.default_rng(seed)

    X: List[Cell] = []
    llold = loglike(X, y, par)
    lpold = logprior(X, par)

    trace_n = np.empty(nsamp, dtype=int)
    trace_ll = np.empty(nsamp, dtype=float)
    trace_lp = np.empty(nsamp, dtype=float)
    trace_acc = np.zeros(nsamp, dtype=int)
    trace_k = np.empty(nsamp, dtype=int)

    chain_X: List[np.ndarray] = []

    # Optional live plot setup
    fig = im_y = im0 = im1 = None
    if plot_every and plot_every > 0:
        fig, im_y, im0, im1 = _init_live_plot(y)

    for it in range(nsamp):
        kernum = choose_kernel(rng, par.move_ratio)
        trace_k[it] = kernum

        if kernum == 1:
            Xp, lh = simh1_birth_death(X, rng, par)
        elif kernum == 2:
            Xp, lh = simh2_flip_label(X, rng)
        elif kernum == 3:
            Xp, lh = simh3_move_point(X, rng, par)
        else:
            Xp, lh = simh4_swap_two(X, rng)

        llnew = loglike(Xp, y, par)
        lpnew = logprior(Xp, par)

        lalpha = llnew + lpnew - llold - lpold + lh

        if math.log(rng.random()) < lalpha:
            X = Xp
            llold = llnew
            lpold = lpnew
            trace_acc[it] = 1

        trace_n[it] = len(X)
        trace_ll[it] = llold
        trace_lp[it] = lpold

        chain_X.append(np.array(X, dtype=int))

        # Live plotting every so often
        if fig is not None and (it % plot_every == 0 or it == nsamp - 1):
            Ax_cur = render_noiseless_from_latent(X, par.render)
            resid_img = y - Ax_cur

            im0.set_data(Ax_cur)
            im1.set_data(resid_img)

            # auto-scale residual symmetrically around 0
            v = float(np.max(np.abs(resid_img)))
            v = max(v, 1e-8)
            im1.set_clim(-v, v)
            # mag = np.abs(resid_img)
            # im1.set_data(mag)
            # im1.set_cmap("gray")
            # im1.set_clim(0.0, np.percentile(mag, 99))  # robust upper limit

            fig.suptitle(
                f"iter {it+1}/{nsamp} | n={len(X)}",
                fontsize=11
            )
            fig.canvas.draw_idle()
            plt.pause(plot_pause)
    
    if fig is not None:
        plt.ioff()

    return {
        "trace_n": trace_n,
        "trace_ll": trace_ll,
        "trace_lp": trace_lp,
        "trace_acc": trace_acc,
        "trace_k": trace_k,
        "chain_X": chain_X,
    }


# =============================================================================
# _counts_good_bad
# =============================================================================
def _counts_good_bad(chain_X_obj):
    """
    Compute (n_good, n_bad, n_total) from stored latent states.

    Parameters
    ----------
    chain_X_obj : Any
        Typically a list (length T) where each entry is an array of shape (n,3)
        representing (x,y,m). This is exactly the structure produced by run_mcmc:
        outputs["chain_X"] after optional burn-in slicing.

        Convention (as in this file):
            m = 1 -> good
            m = 0 -> bad

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (ngood, nbad, ntotal) each an integer array of length T.
    """
    T = len(chain_X_obj)
    ngood = np.zeros(T, dtype=int)
    nbad = np.zeros(T, dtype=int)
    ntotal = np.zeros(T, dtype=int)

    for t in range(T):
        Xt = np.asarray(chain_X_obj[t])
        if Xt.size == 0:
            continue
        Xt = Xt.reshape(-1, 3)
        m = Xt[:, 2].astype(int)
        ngood[t] = int(np.sum(m == 1))
        nbad[t] = int(np.sum(m == 0))
        ntotal[t] = int(Xt.shape[0])

    return ngood, nbad, ntotal


# =============================================================================
# _save_trace_total_n
# =============================================================================
def _save_trace_total_n(trace_n, png_path: Path):
    """
    Save a trace plot of total cell count n versus iteration.

    Parameters
    ----------
    trace_n : array-like
        Sequence/array of total counts n over iterations (typically after burn-in).
    png_path : pathlib.Path
        Output path to save the PNG file.

    Returns
    -------
    None
        Writes an image to disk.
    """
    plt.figure(figsize=(8, 3))
    plt.plot(trace_n)
    plt.xlabel("iteration")
    plt.ylabel("total cells n")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


# =============================================================================
# _save_post_total_n
# =============================================================================
def _save_post_total_n(trace_n, png_path: Path):
    """
    Save an empirical posterior PMF plot for total cell count n.

    Parameters
    ----------
    trace_n : array-like
        Sequence/array of total counts n over iterations (typically after burn-in).
    png_path : pathlib.Path
        Output path to save the PNG file.

    Returns
    -------
    None
        Writes an image to disk.
    """
    vals, counts = np.unique(trace_n, return_counts=True)
    probs = counts / counts.sum()

    plt.figure(figsize=(6, 4))
    plt.bar(vals, probs)
    plt.xlabel("total cells n")
    plt.ylabel("posterior probability")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


# =============================================================================
# _save_post_good_bad_1d
# =============================================================================
def _save_post_good_bad_1d(ngood, nbad, png_path: Path):
    """
    Save a two-panel plot of the 1D marginal posteriors for (n_good, n_bad).

    The marginal posterior masses are estimated by relative frequencies of
    (n_good) and (n_bad) samples in the chain.

    Parameters
    ----------
    ngood : array-like
        Sequence/array of n_good values (length T).
    nbad : array-like
        Sequence/array of n_bad values (length T).
    png_path : pathlib.Path
        Output path to save the PNG file.

    Returns
    -------
    None
        Writes an image to disk.
    """
    # Discrete posterior mass via relative frequencies
    g_vals, g_counts = np.unique(ngood, return_counts=True)
    b_vals, b_counts = np.unique(nbad, return_counts=True)

    g_probs = g_counts / g_counts.sum()
    b_probs = b_counts / b_counts.sum()

    ymax = float(max(g_probs.max(), b_probs.max()))

    plt.figure(figsize=(10, 4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(g_vals, g_probs)
    ax1.set_xlabel("n_good")
    ax1.set_ylabel("posterior probability")
    ax1.set_ylim(0.0, ymax * 1.05)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(b_vals, b_probs)
    ax2.set_xlabel("n_bad")
    ax2.set_ylabel("posterior probability")
    ax2.set_ylim(0.0, ymax * 1.05)

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

def _init_live_plot(y: np.ndarray):
    """
    Initialize a 1x3 live plot:
      left  = observed data y (clipped to [0,1])
      mid   = A(X) current rendering
      right = residual y - A(X) (diverging)
    Returns figure + image handles for fast updates.
    """
    plt.ion()  # interactive mode on

    y_disp = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)

    fig, (ax_y, ax0, ax1) = plt.subplots(1, 3, figsize=(14, 5))

    im_y = ax_y.imshow(y_disp, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_y.set_title("Observed y (clipped)")
    ax_y.axis("off")

    im0 = ax0.imshow(np.zeros_like(y_disp), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax0.set_title("A(X) current")
    ax0.axis("off")

    im1 = ax1.imshow(np.zeros_like(y_disp), cmap="seismic", vmin=-0.1, vmax=0.1, interpolation="nearest")
    ax1.set_title("Residual: y - A(X)")
    ax1.axis("off")

    plt.tight_layout()
    plt.show(block=False)
    return fig, im_y, im0, im1

# =============================================================================
# main
# =============================================================================
def main() -> None:
    """
    CLI entry point.

    Loads observed image y (from Blockkurs .tif/.mat or from a provided .npy),
    runs the MH/RJMCMC sampler, saves traces and plots, and writes a final
    rendered noiseless image from the last sampled latent state.

    Command-line arguments (from argparse)
    --------------------------------------
    --use_blockkurs : bool
        If True, load the Blockkurs-provided data files under outputs/:
            - outputs/blockkurs_slide.tif (if --use_tiff True)
            - outputs/blockkurs_slide.mat (if --use_tiff False)
        If False, load --y_npy instead.
    --use_tiff : bool
        If using Blockkurs data, choose TIFF (True) vs MAT (False).
        Ignored if --use_blockkurs is False.
    --y_npy : str
        Path to observed pixel array .npy (used only if --use_blockkurs False).
    --nsamp : int
        Number of MCMC iterations.
    --seed : Optional[int]
        Random seed for reproducibility.
    --out : str
        Output stem (directory/name prefix) used for saving .npy traces and .png plots.

    --sigma : float
        Likelihood noise std deviation.
    --lambda_n : float
        Poisson mean for prior on n (notes use 8).
    --move_window : int
        Half-width of translation window (proposal kernel 3).
    --burn : int
        Number of initial samples to discard for posterior summaries/plots.

    Returns
    -------
    None
        Writes outputs to disk and prints a short run summary to stdout.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_blockkurs", type=bool, default=True, help="Use blockkurs mat or tiff instead of (your personal) slide.npy")
    ap.add_argument("--use_tiff", type=bool, default=False, help="Use blockkurs tiff file instead of mat (ignored if --use_blockkurs is False)")
    ap.add_argument("--y_npy", type=str, default="outputs/slide.npy", help="Path to observed pixel array .npy (from cells_render.py)")
    ap.add_argument("--nsamp", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=str, default="outputs/mcmc_run")
    ap.add_argument("--plot_every", type=int, default=200,
                help="If >0, show live plots every this many iterations (0 disables)")
    ap.add_argument("--plot_pause", type=float, default=0.01,
                    help="Pause time (seconds) for matplotlib UI refresh")

    ap.add_argument("--sigma", type=float, default=0.1, help="sigma in N(A(x), sigma^2 I)")
    ap.add_argument("--lambda_n", type=float, default=8.0, help="lambda in Poisson(lambda); notes use 8")
    ap.add_argument("--move_window", type=int, default=5)
    ap.add_argument("--burn", type=int, default=0, help="Number of initial samples to discard as burn-in (default 0)")
    args = ap.parse_args()

    if args.use_blockkurs:
        if args.use_tiff:
            y_path = Path("outputs/blockkurs_slide.tif")
            arr = iio.imread(y_path)
            y = arr.astype(np.float64) / 255.0
        else:
            y_path = Path("outputs/blockkurs_slide.mat")
            arr = loadmat(y_path)["slide"]
            y = np.asarray(arr)
    else:
        y_path = Path(args.y_npy)
        y = np.load(y_path)

    if y.ndim != 2 or y.shape[0] != y.shape[1]:
        raise ValueError(f"y must be a square 2D array, got shape {y.shape}")

    npix = int(y.shape[0])

    par = MCMCParams(
        npix=npix,
        lambda_n=float(args.lambda_n),
        sigma=float(args.sigma),
        move_window=int(args.move_window),
        render=RenderParams(npix=npix),
    )

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    print("Running MH MCMC.")
    print(f"  chain length: {args.nsamp}")
    print(f"  burn-in: {args.burn}")
    print(f"  lambda_n: {args.lambda_n}")
    print(f"  sigma: {args.sigma}")
    print(f"  move_window: {args.move_window}")

    outputs = run_mcmc(
        y=y,
        nsamp=int(args.nsamp),
        par=par,
        seed=args.seed,
        plot_every=int(args.plot_every),
        plot_pause=float(args.plot_pause),
    )

    np.save(out_stem.with_suffix(".trace_n.npy"), outputs["trace_n"])
    np.save(out_stem.with_suffix(".trace_ll.npy"), outputs["trace_ll"])
    np.save(out_stem.with_suffix(".trace_lp.npy"), outputs["trace_lp"])
    np.save(out_stem.with_suffix(".trace_acc.npy"), outputs["trace_acc"])
    np.save(out_stem.with_suffix(".trace_k.npy"), outputs["trace_k"])
    np.save(out_stem.with_suffix(".chain_X.npy"), np.array(outputs["chain_X"], dtype=object), allow_pickle=True)

    acc_rate = float(np.mean(outputs["trace_acc"]))
    print("\nMH MCMC run complete.")
    print(f"  y loaded from: {y_path}")
    print(f"  acceptance rate: {acc_rate:.3f}")
    print(f"  final N: {int(outputs['chain_X'][-1].shape[0])}")
    print(f"  outputs stem: {out_stem}")

    # Print final latent state (robust to n=0)
    final_X_arr = np.asarray(outputs["chain_X"][-1], dtype=int)
    if final_X_arr.size == 0:
        final_latent: List[Cell] = []
    else:
        final_latent = [tuple(map(int, row)) for row in final_X_arr.reshape(-1, 3)]

    print(f"  final latent X (n={len(final_latent)}):")
    for i, (x, y, m) in enumerate(final_latent, start=1):
        print(f"    cell {i}: x={x}, y={y}, m={m}")

    # Load + print the latent state used to generate y (if available)
    if not args.use_blockkurs:
        y_latent_path = y_path.with_name(f"{y_path.stem}_latent.npy")  # e.g. slide.npy -> slide_latent.npy
        if y_latent_path.exists():
            y_latent_arr = np.asarray(np.load(y_latent_path), dtype=int)

            if y_latent_arr.size == 0:
                y_latent: List[Cell] = []
            else:
                y_latent_arr = y_latent_arr.reshape(-1, 3)
                y_latent = [tuple(map(int, row)) for row in y_latent_arr]

            print(f"  y latent loaded from: {y_latent_path} (n={len(y_latent)})")
            for i, (x_i, y_i, m_i) in enumerate(y_latent, start=1):
                print(f"    cell {i}: x={x_i}, y={y_i}, m={m_i}")
        else:
            print(f"  note: latent file not found at {y_latent_path} (skipping)")

    # Render final state and save an image
    x_final = render_noiseless_from_latent(final_latent, par.render)

    png_path = out_stem.with_suffix(".final_render.png")
    plt.figure(figsize=(5, 5))
    plt.imshow(x_final, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"  saved final render PNG to: {png_path}")

    # Burn-in
    burn = int(args.burn)
    trace_n_bt = outputs["trace_n"][burn:]
    chain_X_bt = outputs["chain_X"][burn:]

    # Counts for good/bad from marks m
    ngood, nbad, ntotal = _counts_good_bad(chain_X_bt)

    # Save plots
    trace_png = out_stem.with_suffix(".trace_total_n.png")
    post_n_png = out_stem.with_suffix(".post_total_n.png")
    post_gb_png = out_stem.with_suffix(".post_good_bad.png")

    _save_trace_total_n(trace_n_bt, trace_png)
    _save_post_total_n(trace_n_bt, post_n_png)
    _save_post_good_bad_1d(ngood, nbad, post_gb_png)

    print("\nSaved plots:")
    print(f"  {trace_png}")
    print(f"  {post_n_png}")
    print(f"  {post_gb_png}")

    # Keep the live plot window open after the run finishes
    if args.plot_every and args.plot_every > 0:
        # Save the final live plot figure (only if live plotting was enabled)
        final_live_png = out_stem.with_suffix(".live_final.png")
        plt.savefig(final_live_png, dpi=200, bbox_inches="tight")
        print(f"  {final_live_png}")
        plt.show(block=True)
            
# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()