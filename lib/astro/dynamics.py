"""Clohessy-Wiltshire state-transition matrices."""

import numpy as np
from scipy.linalg import expm


def stm_full(omega: float, dt: float) -> np.ndarray:
    """6x6 CW state-transition matrix. Phi(t) = expm(A*t) solves the linearized
    relative-motion ODE in closed form, and composes exactly
    (Phi(dt)^k == Phi(k*dt)) -- which is what lets a whole coast be propagated
    as one batched matmul instead of a step loop."""
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 5] = 2 * omega        # x_ddot couples z_dot
    A[4, 1] = -omega ** 2      # y_ddot (cross-track, decoupled)
    A[5, 2] = 3 * omega ** 2   # z_ddot couples x/z
    A[5, 3] = -2 * omega
    return expm(A * dt)


def stm_2d(omega: float, dt: float) -> np.ndarray:
    """In-plane 4x4 sub-block, state = [x, z, xdot, zdot]. Cross-track y is
    fully decoupled, so those rows/cols just drop out."""
    idx = np.ix_([0, 2, 3, 5], [0, 2, 3, 5])
    return stm_full(omega, dt)[idx]


def stm_inplane(omega: float, t: float) -> np.ndarray:
    """Same as stm_2d, built directly for the 4-state in-plane system."""
    A = np.zeros((4, 4))
    A[0, 2] = A[1, 3] = 1.0
    A[2, 3] = 2 * omega
    A[3, 1] = 3 * omega ** 2
    A[3, 2] = -2 * omega
    return expm(A * t)
