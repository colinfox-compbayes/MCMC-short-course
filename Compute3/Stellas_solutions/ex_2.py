#!/usr/bin/env python3
"""
Exercise 2 for compute 3. 
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
        K (np.ndarray): (N+1, N+1) stiffness matrix.
        M (np.ndarray): (N+1, N+1) mass matrix.
        
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


def make_heat(
    *,
    L: float = 10.0,
    T: float = 2.0,
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
        T (float, optional): Final time to solve heat equation to.
        x_dim (int, optional): Number of spatial points (including boundaries).
        initial_heat_start_proportion (float, optional): Proportion of L where initial heat starts.
        initial_heat_end_proportion (float, optional): Proportion of L where initial heat ends.
    Returns:
        heat (callable): Function that takes diffusivity array D (which holds values 
        D_i = D(x) on each element (x_i, x_{i+1})) and returns u(., T) at spatial points 
        excluding boundaries.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if L <= 0:
        raise ValueError("L must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if x_dim < 2:
        raise ValueError("x_dim must be >= 2")
    
    x = np.linspace(0.0, L, x_dim)

    initial_heat_start = initial_heat_start_proportion * L
    initial_heat_end = initial_heat_end_proportion * L
    u0 = np.zeros(x_dim, dtype=float)
    u0[(x >= initial_heat_start) & (x <= initial_heat_end)] = 1.0
    
    def heat(D: np.ndarray) -> np.ndarray:
        num_elements = x_dim - 1
        if D.size != num_elements:
            raise ValueError(f"Expected D of length {num_elements}, got {D.size}.")

        K, M = heatfem(x, np.ones(num_elements), D)
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

    return heat


def jacobian(heat: callable, D: np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian of the map D -> (y_i)_{i=1}^N, where y_i = u(x_i, T), via a finite difference approximation. 
    
    Args:
        heat (callable): Function that takes diffusivity array D and returns u(., T) at spatial points excluding boundaries.
        D (np.ndarray): (N,) D values in each element (x_i, x_{i+1}).

    Returns:
        y (np.ndarray): (N,) values of u(., T) at spatial points excluding boundaries for diffusivity D.
        J (np.ndarray): (N, D.size) Jacobian matrix, where J[k, j] = dy_k/dD_j.
    """
    y = heat(D)
    J = np.zeros((y.size, D.size), dtype=float)

    for k, j in enumerate(range(D.size)):
        step = 0.1
        D_step = D.copy()
        D_step[j] = D[j] + step

        y_step = heat(D_step)
        J[:, k] = (y_step - y) / step
    
    return y, J


def effective_rank_from_noise(J: np.ndarray, s: float, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute the effective rank of map J at noise level s based 
    on the heuristic that the effective rank is the number of singular values 
    above noise level (max(sigmas) / SNR), where SNR is the signal to noise ratio 
    computed as max(|y|) / s.

    Args:
        J (np.ndarray): Jacobian matrix.
        s (float): Noise standard deviation.
        y (np.ndarray): Output vector at diffusivity D.

    Returns:
        tuple[np.ndarray, float, int]: Tuple containing the singular values of J, 
        the signal-to-noise ratio, and the effective rank.
    """
    # compute SNR
    y_max = np.max(np.abs(y))
    SNR = y_max / s if s > 0 else np.inf
    # compute singular values of J
    sigmas = np.linalg.svd(J, compute_uv=False)
    # heuristic: effective rank is the number of singular values above noise level (max(sigmas) / SNR)
    effective_rank = int(np.sum(sigmas > sigmas[0]/SNR))
    return sigmas, SNR, effective_rank


def get_effective_rank(x_dim: int = 25) -> int:
    """Compute the effective rank of the Jacobian of the map D -> (y_i)_{i=1}^N, where y_i = u(x_i, T), 
    for a given number of spatial points x_dim. Uses a finite difference approximation to compute the 
    Jacobian, and computes effective rank with heuristic based on noise level s.
    
    Args:
        x_dim (int, optional): Number of spatial points. Defaults to 25.

    Returns:
        int: Effective rank of the Jacobian.
    """
    # problem setup
    L = 10.0
    x = np.linspace(0.0, L, x_dim)
    element_midpoints = 0.5 * (x[:-1] + x[1:])
    D = 0.5 + 0.3 * np.sin(2.0 * np.pi * element_midpoints / L)
    T = 2.0
    s = 0.03
    
    heat = make_heat(L=L, T=T, x_dim=x_dim)

    # compute Jacobian at D
    y, J = jacobian(heat, D)

    # get effective rank of J at noise level s
    sigmas, SNR, effective_rank = effective_rank_from_noise(J, s, y)
    
    return effective_rank


def main() -> None:
    """Method to compute the effective rank of the Jacobian of the map D -> (y_i)_{i=1}^N, where y_i = u(x_i, T), 
    for a range of numbers of spatial points x_dim, and plot the number of elements vs effective rank. 
    Also prints the number of elements vs effective rank.
    """
    x_dims = [2**k for k in range(4, 9)] # 16, 32, 64, 128, 256, 512
    effective_ranks = [get_effective_rank(xdim) for xdim in x_dims]
    num_elements = np.array(x_dims) - 1
        
    # plot number of elements vs effective rank
    plt.figure(3)
    plt.clf()
    plt.plot(num_elements, effective_ranks, marker="o")
    plt.xscale("log")
    plt.title("Effective rank of Jacobian vs number of elements")
    plt.xlabel("number of elements")
    plt.ylabel("effective rank")
    plt.tight_layout()
    plt.show()
    
    # print number of elements vs effective rank
    print("number of elements | effective rank")
    print("-------------------")
    for num_element, effective_rank in zip(num_elements, effective_ranks):
        print(f"{num_element:4d} | {effective_rank:4d}")


if __name__ == "__main__":
   main()