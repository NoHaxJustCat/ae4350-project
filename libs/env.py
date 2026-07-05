import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
from libs.constants import (
    ENV_BONUS,
    ENV_BOUNDARY,
    ENV_DT,
    ENV_FUEL_COEFF,
    ENV_INITIAL_STATE_VBAR,
    ENV_MAX_DV,
    ENV_POS_TOLERANCE,
    ENV_TIMEOUT,
    ENV_VEL_COEFF,
    ENV_VEL_TOLERANCE,
    OMEGA,
)
from scipy.linalg import expm
import torch

class CWRendezvousEnv(gym.Env):

    def __init__(
            self, 
            omega: float = OMEGA,
            dt: float = ENV_DT,
            max_dv: float = ENV_MAX_DV,
            boundary: float = ENV_BOUNDARY,
            timeout: float = ENV_TIMEOUT,
            pos_tolerance: float = ENV_POS_TOLERANCE,
            vel_tolerance: float = ENV_VEL_TOLERANCE,
            vel_coeff: float = ENV_VEL_COEFF,
            fuel_coeff: float = ENV_FUEL_COEFF,
            bonus: float = ENV_BONUS,
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
        self.STM_np = expm(A * dt)                       # for the numpy env step

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
        stm = self.STM.to(device=state.device, dtype=state.dtype)
        return stm @ s0


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Random circle

        # theta = self.np_random.uniform(0, 2 * np.pi)
        # r     = 100.0
        # self.state = np.array(
        #     [r * np.cos(theta), r * np.sin(theta), 0.0, 0.0, 0.0, 0.0],
        #     dtype=np.float64,
        # )

        # Shift along V-bar
        self.state = np.array(
            ENV_INITIAL_STATE_VBAR,
            dtype=np.float64,
        )
        
        self.best_distance = np.linalg.norm(self.state[0:3])

        self.elapsed_time = 0.0

        observation = self.state.copy()
        info = {}
        return observation, info

    def step(self, action: np.ndarray):

        # enforce action_space bounds
        action = np.clip(action, -self.max_dv, self.max_dv)

        # apply impulsive delta-v to velocity components
        self.state[3:6] += action
        
        # propagate freely under CW dynamics for dt

        # sol = solve_ivp(
        #     self.cw_dynamics, (0, self.dt), self.state,
        #     args=(self.omega,), rtol=1e-9, atol=1e-9
        # )

        self.state = self.STM_np @ self.state
        
        self.elapsed_time += self.dt

        pos_error = np.linalg.norm(self.state[0:3])
        vel_error = np.linalg.norm(self.state[3:6])

        docked = (pos_error < self.pos_tolerance)

        if pos_error < self.best_distance:
            reward_pos = self.best_distance - pos_error  # reward = distance closed
            self.best_distance = pos_error
        else:
            reward_pos = 0.0

        terminated = bool((pos_error > self.boundary) or docked)
        truncated = bool(self.elapsed_time > self.timeout)

        reward_fuel = - self.fuel_coeff * np.linalg.norm(action)
        reward_shaping = -0.01 * pos_error  # scaled ~100x smaller than progress bonus
        reward_bonus = self.bonus if docked else 0.0
        
        if docked:
            reward_vel = -self.vel_coeff * vel_error
        else:
            reward_vel = 0

        reward = reward_pos + reward_fuel + reward_vel + reward_shaping

        observation = self.state.copy()
        info = {
            "distance": pos_error,
            "docked": docked,
            "reward_pos": reward_pos,
            "reward_vel": reward_vel,
            "reward_fuel": reward_fuel,
            "reward_bonus": reward_bonus,
        }

        return observation, reward, terminated, truncated, info