import gymnasium as gym
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import solve_ivp
from typing import Optional
from libs.constants import *
from scipy.linalg import expm
import torch

class CWRendezvousEnv(gym.Env):

    def __init__(
            self, 
            omega: float, 
            dt: float = 5.0, 
            max_dv: float = 0.2, 
            boundary: float = 1000.0, 
            timeout: float = 10 * T,
            pos_tolerance: float = 1.0,
            vel_tolerance: float = 0.01,
            vel_coeff: float = 10.0,
            fuel_coeff: float = 1.0,
            bonus: float = 100.0,
        ):
        

        super().__init__()
        self.omega = omega          # mean motion [rad/s]
        self.dt = dt                # propagation time per step [s]
        self.max_dv = max_dv        # max delta-v per axis [m/s]
        self.boundary = boundary    # distance from origin that ends episode [m]
        self.timeout = timeout
        self.pos_tolerance = pos_tolerance
        self.vel_tolerance = vel_tolerance
        self.vel_coeff = vel_coeff
        self.fuel_coeff = fuel_coeff
        self.bonus = bonus

        A = np.zeros((6, 6))
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        A[3, 5] = 2 * omega
        A[4, 1] = -omega ** 2
        A[5, 2] = 3 * omega ** 2
        A[5, 3] = -2 * omega
        self.STM = torch.tensor(expm(A * dt), dtype=torch.float32)  # constant, precomputed 
        

        # Observation: [x, y, z, xdot, ydot, zdot] -- continuous, unbounded
        # (bounding it artificially isn't physically meaningful here, since
        # position/velocity can in principle take any real value)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )

        # Action: delta-v applied at the start of each step, [dvx, dvy, dvz]
        self.action_space = spaces.Box(
            low=-max_dv, high=max_dv, shape=(3,), dtype=np.float64
        )

        self.state = None

    def propagate_torch(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Differentiable CW propagation for actor training (HDP model step)."""
        v_new = state[3:6] + action
        s0 = torch.cat([state[0:3], v_new])
        return self.STM @ s0


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        theta = self.np_random.uniform(0, 2 * np.pi)
        r     = 100.0
        self.state = np.array(
            [r * np.cos(theta), r * np.sin(theta), 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )


        # # Random 2D point within 100m box
        # self.state = np.array(
        #     [self.np_random.uniform(0, 100), self.np_random.uniform(0, 100), 0.0, 0.0, 0.0, 0.0],
        #     dtype=np.float64,
        # )
        self.elapsed_time = 0.0

        observation = self.state.copy()
        info = {}
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_dv, self.max_dv)  # enforce action_space bounds
        # apply impulsive delta-v to velocity components
        self.state[3:6] += action
        # propagate freely under CW dynamics for dt

        sol = solve_ivp(
            self.cw_dynamics, (0, self.dt), self.state,
            args=(self.omega,), rtol=1e-9, atol=1e-9
        )

        self.state = sol.y[:, -1]
        self.elapsed_time += self.dt

        pos_error = np.linalg.norm(self.state[0:3])
        vel_error = np.linalg.norm(self.state[3:6])
        docked = (pos_error < self.pos_tolerance) and (vel_error < self.vel_tolerance)

        terminated = bool((pos_error > self.boundary) or docked)
        truncated = bool(self.elapsed_time > self.timeout)

        reward = -pos_error - self.vel_coeff * vel_error - self.fuel_coeff * np.linalg.norm(action)
        if docked:
            reward += self.bonus

        observation = self.state.copy()
        info = {"distance": pos_error, "docked": docked}

        return observation, reward, terminated, truncated, info

    @staticmethod
    def cw_dynamics(t, s, omega):
        r = s[0:3]
        r_dot = s[3:6]
        r_ddot = np.array([
            2 * omega * r_dot[2],
            -omega ** 2 * r[1],
            3 * omega ** 2 * r[2] - 2 * omega * r_dot[0]
        ])
        return np.concatenate([r_dot, r_ddot])