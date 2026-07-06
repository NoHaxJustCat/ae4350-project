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
    ENV_SHAPING_COEFF,
    OMEGA,
    ENV_MAX_DIR_CHANGE,
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
            max_direction_changes: float = ENV_MAX_DIR_CHANGE
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
        self.max_direction_changes = max_direction_changes

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
        stm = self.STM
        if stm.device != state.device or stm.dtype != state.dtype:
            stm = stm.to(device=state.device, dtype=state.dtype)
            self.STM = stm

        if state.dim() == 1:
            v_new = state[3:6] + action
            s0 = torch.cat([state[0:3], v_new])
            return stm @ s0

        v_new = state[:, 3:6] + action
        s0 = torch.cat([state[:, 0:3], v_new], dim=1)
        return s0 @ stm.T


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

        self._direction_changes = 0
        self._prev_delta = None

        observation = self.state.copy()
        info = {}
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_dv, self.max_dv)
        
        # Store previous state for delta-rewards
        prev_pos_error = np.linalg.norm(self.state[0:3])
        
        # Apply action and propagate
        self.state[3:6] += action
        self.state = self.STM_np @ self.state
        self.elapsed_time += self.dt

        pos_error = np.linalg.norm(self.state[0:3])
        vel_error = np.linalg.norm(self.state[3:6])

        # --- OSCILLATION DETECTION ---
        delta = prev_pos_error - pos_error
        if hasattr(self, '_prev_delta') and self._prev_delta is not None:
            if np.sign(delta) != np.sign(self._prev_delta):
                self._direction_changes += 1
        self._prev_delta = delta
        oscillating = self._direction_changes >= self.max_direction_changes

        docked = pos_error < self.pos_tolerance
        out_of_bounds = pos_error > self.boundary
        timeout = self.elapsed_time > self.timeout

        terminated = bool(docked or out_of_bounds or oscillating)
        truncated = bool(timeout)

        # --- REWARD COMPONENTS ---

        # 1. Distance progress (DENSE - always active)
        reward_pos = ENV_SHAPING_COEFF * (prev_pos_error - pos_error)

        # 3. Fuel penalty
        reward_fuel = -ENV_FUEL_COEFF * np.linalg.norm(action)

        # 4. Terminal rewards
        if docked:
            reward_terminal = ENV_BONUS - ENV_VEL_COEFF * vel_error
        elif out_of_bounds:
            reward_terminal = -200.0
        elif oscillating:
            reward_terminal = 0.0  # No reward, no penalty
        elif truncated:
            reward_terminal = -50.0
        else:
            reward_terminal = 0.0

        reward = reward_pos + reward_fuel + reward_terminal

        observation = self.state.copy()
        info = {
            "distance": pos_error,
            "docked": docked,
            "oscillating": oscillating,
            "direction_changes": self._direction_changes,
            "reward_pos": reward_pos,
            "reward_fuel": reward_fuel,
            "reward_terminal": reward_terminal,
            "vel_error": vel_error,
            "delta_v": np.linalg.norm(action),
        }

        return observation, reward, terminated, truncated, info