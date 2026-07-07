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
    ENV_DV_BUDGET,
    ENV_FAST_DT_MULT,
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
        dv_budget: float = ENV_DV_BUDGET,
        fast_dt_mult: float = ENV_FAST_DT_MULT,
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
        self.dv_budget = dv_budget          # total delta-v available for the episode
        self.fast_dt_mult = fast_dt_mult    # how much bigger the "coasting" step is once budget is spent

        A = np.zeros((6, 6))
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        A[3, 5] = 2 * omega
        A[4, 1] = -omega ** 2
        A[5, 2] = 3 * omega ** 2
        A[5, 3] = -2 * omega

        # Normal-speed state transition matrix
        self.STM = torch.tensor(expm(A * dt), dtype=torch.float32)
        self.STM_np = expm(A * dt)
        self.STM_T = self.STM.t().contiguous()

        # Fast-forward STM used once the delta-v budget is exhausted
        fast_dt = dt * fast_dt_mult
        self.STM_fast_np = expm(A * fast_dt)

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

    def _ensure_stm(self, device, dtype):
        if self.STM.device != device or self.STM.dtype != dtype:
            self.STM = self.STM.to(device=device, dtype=dtype)
            self.STM_T = self.STM.t().contiguous()

    def propagate_torch(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self._ensure_stm(state.device, state.dtype)
        v_new = state[..., 3:6] + action
        s0 = torch.cat([state[..., 0:3], v_new], dim=-1)
        return torch.nn.functional.linear(s0, self.STM_T)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Shift along V-bar
        self.state = np.array(ENV_INITIAL_STATE_VBAR, dtype=np.float64)

        self.best_distance = np.linalg.norm(self.state[0:3])
        self.step_count = 0
        self.wrong_way_counter = 0
        self.elapsed_time = 0.0
        self.dv_used = 0.0

        observation = self.state.copy()
        info = {}
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_dv, self.max_dv)

        # Cap this burn to whatever delta-v budget remains
        dv_remaining = max(0.0, self.dv_budget - self.dv_used)
        action_norm = np.linalg.norm(action)
        budget_exhausted = dv_remaining <= 0.0
        if budget_exhausted:
            action = np.zeros_like(action)
        elif action_norm > dv_remaining:
            action = action * (dv_remaining / action_norm)

        self.dv_used += np.linalg.norm(action)

        # Store previous state for delta-rewards
        prev_pos_error = np.linalg.norm(self.state[0:3])

        # Apply action and propagate (fast STM once out of fuel)
        self.state[3:6] += action
        stm = self.STM_fast_np if budget_exhausted else self.STM_np
        step_dt = self.dt * self.fast_dt_mult if budget_exhausted else self.dt
        self.state = stm @ self.state
        self.elapsed_time += step_dt

        pos_error = np.linalg.norm(self.state[0:3])
        vel_error = np.linalg.norm(self.state[3:6])

        docked = pos_error < self.pos_tolerance
        out_of_bounds = pos_error > self.boundary
        timeout = self.elapsed_time > self.timeout

        delta = prev_pos_error - pos_error
        self.step_count += 1
        if delta < 0:
            self.wrong_way_counter += 1
        else:
            self.wrong_way_counter = 0
        terminated = bool(docked or out_of_bounds or self.wrong_way_counter > 50)
        truncated = bool(timeout)

        # --- REWARD COMPONENTS ---

        # 1. Distance progress (DENSE - always active)
        reward_pos = ENV_SHAPING_COEFF * delta

        # 2. Fuel penalty (keep this small)
        reward_fuel = -ENV_FUEL_COEFF * np.linalg.norm(action)

        # 3. Terminal rewards (clear success vs failure signals)
        if docked:
            reward_terminal = ENV_BONUS - ENV_VEL_COEFF * vel_error
        elif out_of_bounds or self.wrong_way_counter > 20:
            reward_terminal = -100.0
        elif truncated:
            reward_terminal = -50.0
        else:
            reward_terminal = 0.0

        reward = reward_pos + reward_fuel + reward_terminal

        observation = self.state.copy()
        info = {
            "distance": pos_error,
            "docked": docked,
            "reward_pos": reward_pos,
            "reward_fuel": reward_fuel,
            "reward_terminal": reward_terminal,
            "vel_error": vel_error,
            "delta_v": np.linalg.norm(action),
            "dv_used": self.dv_used,
            "dv_remaining": max(0.0, self.dv_budget - self.dv_used),
            "budget_exhausted": budget_exhausted,
        }

        return observation, reward, terminated, truncated, info