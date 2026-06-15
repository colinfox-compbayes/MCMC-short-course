#!/usr/bin/env python3
"""
Exercise 1 for compute 3. 
"""

from __future__ import annotations
from scipy.integrate import solve_ivp

import numpy as np
import matplotlib.pyplot as plt


def heatfem(x: np.ndarray, m: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute stiffness matrix K and mass matrix M via FEM discretization for
    differential operators  -D u''  and  m u  over [x[0], x[-1]].

    Args:
        x (np.ndarray): (N+1,) x values, sorted ascending.
        m (np.ndarray): (N,) mass values in each element (x_i, x_{i+1}).
        D (np.ndarray): (N,) D values in each element (x_i, x_{i+1}).

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of stiffness matrix K and mass matrix M, each of shape (N+1, N+1).
        
    Raises:
        ValueError: If inputs are invalid.
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    D = np.asarray(D, dtype=float)

    num_points = x.size
    num_elements = num_points - 1
    if m.size != num_elements or D.size != num_elements:
        raise ValueError(f"Expected m and D of length {num_elements}, got {m.size} and {D.size}.")

    K = np.zeros((num_points, num_points), dtype=float)
    M = np.zeros((num_points, num_points), dtype=float)

    K_local = np.array([[1.0, -1.0],
                    [-1.0, 1.0]])
    M_local = np.array([[2.0, 1.0],
                    [1.0, 2.0]]) / 6.0
    
    for element in range(num_elements):
        element_length = x[element + 1] - x[element]
        if element_length <= 0:
            raise ValueError("x must be strictly increasing.")
        element_m = m[element]
        element_D = D[element]

        idx = np.array([element, element + 1])
        K[np.ix_(idx, idx)] += (element_D / element_length) * K_local
        M[np.ix_(idx, idx)] += (element_m * element_length) * M_local

    return K, M


def make_heat_T(
    *,
    L: float = 10.0,
    x_dim: int = 10,
    initial_heat_start_proportion: float = 0.75,
    initial_heat_end_proportion: float = 0.80,
) -> callable:
    """
    Returns a function heat(D, T) that computes the solution u(., T) at final
    time T for diffusivity D, with initial condition of heat = 1 between 
    initial_heat_start_proportion*L and initial_heat_end_proportion*L, and 0 elsewhere, 
    and with noise SD s. Uses FEM discretization in space and solve_ivp in time.

    Args:
        L (float, optional): Length of the rod.
        x_dim (int, optional): Number of spatial points (including boundaries).
        initial_heat_start_proportion (float, optional): Proportion of L where initial heat starts.
        initial_heat_end_proportion (float, optional): Proportion of L where initial heat ends.

    Returns:
        heat (callable): Function that takes diffusivity D and final time T, and returns u(., T) at spatial points excluding boundaries.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if L <= 0:
        raise ValueError("L must be > 0")
    if x_dim < 2:
        raise ValueError("x_dim must be >= 2")
    
    x = np.linspace(0.0, L, x_dim)

    initial_heat_start = initial_heat_start_proportion * L
    initial_heat_end = initial_heat_end_proportion * L
    u0 = np.zeros(x_dim, dtype=float)
    u0[(x >= initial_heat_start) & (x <= initial_heat_end)] = 1.0
    
    def heat_T(D: float, T: float) -> np.ndarray:
        num_elements = x_dim - 1
        K, M = heatfem(x, np.ones(num_elements), D * np.ones(num_elements))
        L = - np.linalg.solve(M, K)

        def heat_ode(t: float, u: np.ndarray) -> np.ndarray:
            udot = L @ u
            udot[0] = 0.0
            udot[-1] = 0.0
            return udot

        sol = solve_ivp(
            heat_ode,
            t_span=(0.0, T),
            y0=u0,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )

        u_final = sol.y[:, -1]
        return u_final[1:-1]

    return heat_T


def mc_diffusion_T(u_data: np.ndarray, heat_T: callable, D0: float, T0: float, s: float, w_D: float, w_T: float, n: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Metropolis sampler with uniform proposal for diffusivity D 
    in the 1D heat equation, given observed data u_data and a function heat(D) 
    that computes the solution u(., T) at final time T for diffusivity D.

    Args:
        u_data (np.ndarray): Observed data (with noise) for u(., T) at spatial points excluding boundaries.
        heat_T (callable): Function that takes diffusivity D and time T and returns u(., T) at spatial points excluding boundaries.
        D0 (float): Initial guess for the diffusivity.
        T0 (float): Initial guess for the time.
        s (float): Standard deviation of the noise in the observed data.
        w_D (float): Proposal window size for the uniform proposal for D.
        w_T (float): Proposal window size for the uniform proposal for T.
        n (int): Number of samples to draw.
        rng (np.random.Generator | None, optional): Random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray, float, float]: Tuple array of samples of D, 
        array of samples of T, acceptance rate for D, acceptance rate for T.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if D0 < 0:
        raise ValueError("D0 must be >= 0")
    if T0 < 1.9 or T0 > 2.1:
        raise ValueError("T0 must be in [1.9, 2.1]")
    if s <= 0:
        raise ValueError("s must be > 0")
    if w_D <= 0:
        raise ValueError("w_D must be > 0")
    if w_T <= 0:
        raise ValueError("w_T must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    
    rng = np.random.default_rng() if rng is None else rng

    # perform MCMC to infer D and T
    D_current = D0
    D_chain = np.empty(n + 1, dtype=float)
    D_chain[0] = D_current
    
    T_current = T0
    T_chain = np.empty(n + 1, dtype=float)
    T_chain[0] = T_current

    u_current = heat_T(D_current, T_current)
    log_like_current = np.sum(-(u_current - u_data) ** 2) / (2.0 * s**2)
    
    num_accept_T = 0
    num_accept_D = 0

    for k in range(1, n + 1):
        # uniform proposal for D
        D_proposal = D_current + w_D * (2 * rng.random() - 1)  

        # prior and so target is 0 for D < 0, so always reject
        if D_proposal >= 0:
            u_proposal = heat_T(D_proposal, T_current)
            log_like_proposal = np.sum(-(u_proposal - u_data) ** 2) / (2.0 * s**2)

            # log(target(x_proposal)) - log(target(x_current))
            log_ratio = log_like_proposal - log_like_current
            # accept with probability min(1, exp(log_ratio))
            if np.log(rng.random()) < np.minimum(0.0, log_ratio):
                D_current = D_proposal
                log_like_current = log_like_proposal
                num_accept_D += 1
                
        # uniform proposal for T
        T_current = T_chain[k - 1]
        T_proposal = T_current + w_T * (2 * rng.random() - 1)  

        # prior and so target is 0 for T outside [1.9, 2.1], so always reject
        if 1.9 <= T_proposal <= 2.1:
            u_proposal = heat_T(D_current, T_proposal)
            log_like_proposal = np.sum(-(u_proposal - u_data) ** 2) / (2.0 * s**2)

            # log(target(x_proposal)) - log(target(x_current))
            log_ratio = log_like_proposal - log_like_current
            # accept with probability min(1, exp(log_ratio))
            if np.log(rng.random()) < np.minimum(0.0, log_ratio):
                T_current = T_proposal
                log_like_current = log_like_proposal
                num_accept_T += 1
        
        D_chain[k] = D_current
        T_chain[k] = T_current

    acc_rate_D = num_accept_D / n
    acc_rate_T = num_accept_T / n
    return D_chain, T_chain, acc_rate_D, acc_rate_T
    
    
def main() -> None:
    """Method to simulate data for the 1D heat equation with diffusivity D, run a Metropolis-within-Gibbs sampler
    with uniform proposal to infer D and T, plot the trace and histogram of D and T samples, 
    and print the mean and SD of D and T samples. Note that the window sizes for D and T are fixed, and not tuned.
    """
    rng = np.random.default_rng(0)
    
    # simulate data
    D = 0.5
    L = 10
    s = 0.03
    T = 2
    x_dim = 25
    
    heat_T = make_heat_T(L=L, x_dim=x_dim)
    u_data = heat_T(D, T) + s * rng.standard_normal(x_dim - 2)
    
    # perform MCMC to infer D and T
    D0 = 1.0
    T0 = 2.0
    w_D = 0.16
    w_T = 0.16
    n = 10_000
    burn_in = 1_000
    
    D_chain, T_chain, acc_rate_D, acc_rate_T = mc_diffusion_T(u_data, heat_T, D0, T0, s, w_D, w_T, n, rng)
    
    # plot final trace of D samples
    plt.figure(1)
    plt.clf()
    plt.plot(D_chain)
    plt.title("Final trace of D samples")
    plt.tight_layout()
    plt.show()
    
    # plot final trace of T samples
    plt.figure(2)
    plt.clf()
    plt.plot(T_chain)
    plt.title("Final trace of T samples")
    plt.tight_layout()
    plt.show()
    
    # plot normalized histogram of D samples
    samples = D_chain[burn_in:]
    plt.figure(3)
    plt.clf()
    plt.hist(samples, bins=60, density=True, alpha=0.6)
    plt.title(f"Normalized histogram of D samples\n(D0={D0:g}, w_D={w_D:g}, w_T={w_T:g}, n={n}, burn_in={burn_in})")
    plt.xlabel("D")
    plt.ylabel("density")
    plt.show()
    
    # plot normalized histogram of T samples
    samples = T_chain[burn_in:]
    plt.figure(4)
    plt.clf()
    plt.hist(samples, bins=60, density=True, alpha=0.6)
    plt.title(f"Normalized histogram of T samples\n(T0={T0:g}, w_D={w_D:g}, w_T={w_T:g}, n={n}, burn_in={burn_in})")
    plt.xlabel("T")
    plt.ylabel("density")
    plt.show()
    
    # print mean and SD of D and T samples
    mean_D = float(np.mean(D_chain[burn_in:]))
    sd_D = float(np.std(D_chain[burn_in:], ddof=1))
    mean_T = float(np.mean(T_chain[burn_in:]))
    sd_T = float(np.std(T_chain[burn_in:], ddof=1))
    print(f"D: mean = {mean_D:.3f}, SD = {sd_D:.4f}")
    print(f"T: mean = {mean_T:.3f}, SD = {sd_T:.4f}")


if __name__ == "__main__":
    main()