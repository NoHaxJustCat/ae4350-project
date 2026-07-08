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
    ENV_CURRICULUM_ENABLED,
    ENV_CURRICULUM_START_DISTANCE,
    ENV_CURRICULUM_MAX_DISTANCE,
    ENV_CURRICULUM_INCREMENT,
    MODE_2D,
    ACTION_DIM,
    PHYS_STATE_DIM,
    OBS_DIM,
)
from scipy.linalg import expm
import torch


# ---------------------------------------------------------------------------
#   Build State Transition Matrices
# ---------------------------------------------------------------------------
# Full 6×6 CW matrix (always built; 2-D mode selects a 4×4 sub-block).
def _build_stm_full(omega: float, dt: float) -> np.ndarray:
    A = np.zeros((6, 6))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    A[3, 5] =  2 * omega       # ẍ couples ż
    A[4, 1] = -omega ** 2      # ÿ (cross-track, decoupled)
    A[5, 2] =  3 * omega ** 2  # z̈ couples ẋ/z
    A[5, 3] = -2 * omega
    return expm(A * dt)


def _build_stm_2d(omega: float, dt: float) -> np.ndarray:
    """
    In-plane (V-bar / R-bar) CW sub-block: state = [x, z, ẋ, ż].

    The cross-track (y / H-bar) direction is fully decoupled from the
    in-plane motion, so we can just drop it.  The in-plane indices in the
    full 6-D state are [0, 2, 3, 5] → x, z, ẋ, ż.
    """
    idx = np.ix_([0, 2, 3, 5], [0, 2, 3, 5])
    return _build_stm_full(omega, dt)[idx]


class CWRendezvousEnv(gym.Env):
    """
    Clohessy-Wiltshire spacecraft rendezvous environment.

    Operates in either full 3-D (MODE_2D=False) or decoupled in-plane 2-D
    (MODE_2D=True) according to the flag in constants.py.

    2-D state  : [x, z, ẋ, ż]           — V-bar / R-bar only (y = ẏ = 0)
    3-D state  : [x, y, z, ẋ, ẏ, ż]

    Observation appends the remaining Δv budget so the policy can observe
    fuel scarcity:
        2-D obs : [x, z, ẋ, ż, dv_remaining]           (5-dim)
        3-D obs : [x, y, z, ẋ, ẏ, ż, dv_remaining]     (7-dim)

    Action:
        2-D : [dvx, dvz]   (2-dim)
        3-D : [dvx, dvy, dvz]   (3-dim)
    """

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
        curriculum_enabled: bool = ENV_CURRICULUM_ENABLED,
        curriculum_start_distance: float = ENV_CURRICULUM_START_DISTANCE,
        curriculum_max_distance: float = ENV_CURRICULUM_MAX_DISTANCE,
        curriculum_increment: float = ENV_CURRICULUM_INCREMENT,
        curriculum_boundary_mult: float = 2.0,
    ):
        super().__init__()
        self.mode_2d = MODE_2D
        self.phys_dim = PHYS_STATE_DIM   # 4 (2D) or 6 (3D)
        self.obs_dim  = OBS_DIM          # 5 (2D) or 7 (3D)
        self.action_dim = ACTION_DIM     # 2 (2D) or 3 (3D)

        self.omega = omega
        self.dt = dt
        self.max_dv = max_dv
        self.base_boundary = boundary
        self.excursion_limit = boundary
        self.curriculum_boundary_mult = curriculum_boundary_mult
        self.timeout = timeout
        self.pos_tolerance = pos_tolerance
        self.vel_tolerance = vel_tolerance
        self.vel_coeff = vel_coeff
        self.fuel_coeff = fuel_coeff
        self.bonus = bonus
        self.dv_budget = dv_budget

        # --- Curriculum ---
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_increment = curriculum_increment

        # Build initial state from constants (always a 6-D spec; we slice below)
        base_pos_full = np.asarray(ENV_INITIAL_STATE_VBAR[0:3], dtype=np.float64)
        if self.mode_2d:
            # In-plane initial position: only x (V-bar) and z (R-bar) matter.
            # ENV_INITIAL_STATE_VBAR is assumed to lie in the x-z plane (y=0).
            base_pos_2d = np.array([base_pos_full[0], base_pos_full[2]], dtype=np.float64)
            base_dist = np.linalg.norm(base_pos_2d)
            self._curriculum_dir = (
                base_pos_2d / base_dist if base_dist > 0 else np.array([1.0, 0.0])
            )
            base_vel_full = np.asarray(ENV_INITIAL_STATE_VBAR[3:6], dtype=np.float64)
            self._base_velocity = np.array([base_vel_full[0], base_vel_full[2]], dtype=np.float64)
        else:
            base_dist = np.linalg.norm(base_pos_full)
            self._curriculum_dir = (
                base_pos_full / base_dist if base_dist > 0 else np.array([0.0, 1.0, 0.0])
            )
            self._base_velocity = np.asarray(ENV_INITIAL_STATE_VBAR[3:6], dtype=np.float64)

        self.curriculum_max_distance = (
            min(curriculum_max_distance, base_dist) if base_dist > 0 else curriculum_max_distance
        )
        self.curriculum_distance = (
            min(curriculum_start_distance, self.curriculum_max_distance)
            if self.curriculum_enabled
            else self.curriculum_max_distance
        )

        # --- State Transition Matrix ---
        if self.mode_2d:
            stm_np = _build_stm_2d(omega, dt)
        else:
            stm_np = _build_stm_full(omega, dt)

        self.STM_np = stm_np
        self.STM    = torch.tensor(stm_np, dtype=torch.float32)
        self.STM_T  = self.STM.t().contiguous()

        # --- Gym spaces ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-max_dv, high=max_dv, shape=(self.action_dim,), dtype=np.float64
        )

        self.state = None

    # ── Torch helpers ────────────────────────────────────────────────────────

    def _ensure_stm(self, device, dtype):
        if self.STM.device != device or self.STM.dtype != dtype:
            self.STM   = self.STM.to(device=device, dtype=dtype)
            self.STM_T = self.STM.t().contiguous()

    def propagate_torch(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Propagate the PHYSICAL state (4-dim 2D, 6-dim 3D) only.
        Strip the obs dv_remaining column before calling."""
        self._ensure_stm(state.device, state.dtype)
        # velocity indices: last half of physical state
        half = self.phys_dim // 2
        v_new = state[..., half:] + action
        s0 = torch.cat([state[..., :half], v_new], dim=-1)
        return torch.nn.functional.linear(s0, self.STM_T)

    # ── Curriculum helpers ───────────────────────────────────────────────────

    def _initial_state_for_curriculum(self) -> np.ndarray:
        pos = self._curriculum_dir * self.curriculum_distance
        return np.concatenate([pos, self._base_velocity]).astype(np.float64)

    def set_curriculum_distance(self, distance: float):
        self.curriculum_distance = float(
            np.clip(distance, 0.0, self.curriculum_max_distance)
        )

    def _advance_curriculum(self):
        if not self.curriculum_enabled:
            return
        self.curriculum_distance = min(
            self.curriculum_distance + self.curriculum_increment,
            self.curriculum_max_distance,
        )
        # Signal to training loop that buffer should be flushed
        self.curriculum_advanced = True

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        dv_remaining = max(0.0, self.dv_budget - self.dv_used)
        return np.concatenate([self.state, [dv_remaining]]).astype(np.float64)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.curriculum_enabled:
            self.state = self._initial_state_for_curriculum()
            self.excursion_limit = min(
                self.base_boundary,
                self.curriculum_distance * self.curriculum_boundary_mult,
            )
        else:
            if self.mode_2d:
                # Slice x, z, ẋ, ż from the full 6-D constant
                s6 = np.asarray(ENV_INITIAL_STATE_VBAR, dtype=np.float64)
                self.state = np.array([s6[0], s6[2], s6[3], s6[5]], dtype=np.float64)
            else:
                self.state = np.array(ENV_INITIAL_STATE_VBAR, dtype=np.float64)
            self.excursion_limit = self.base_boundary

        # Position part is the first half of the physical state
        half = self.phys_dim // 2
        self.start_pos = self.state[:half].copy()

        self.best_distance   = np.linalg.norm(self.state[:half])
        self.step_count      = 0
        self.wrong_way_counter = 0
        self.elapsed_time    = 0.0
        self.dv_used         = 0.0
        self.curriculum_advanced = False   # ← add this

        observation = self._build_observation()
        info = {"curriculum_distance": self.curriculum_distance}
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_dv, self.max_dv)

        # Cap to remaining Δv budget
        dv_remaining = max(0.0, self.dv_budget - self.dv_used)
        action_norm  = np.linalg.norm(action)
        budget_exhausted = dv_remaining <= 0.0
        if budget_exhausted:
            action = np.zeros_like(action)
        elif action_norm > dv_remaining:
            action = action * (dv_remaining / action_norm)

        self.dv_used += np.linalg.norm(action)

        half = self.phys_dim // 2
        prev_pos_error = np.linalg.norm(self.state[:half])

        # Propagate: apply Δv to velocity, then advance with STM
        self.state[half:] += action
        self.state = self.STM_np @ self.state
        self.elapsed_time += self.dt

        pos_error  = np.linalg.norm(self.state[:half])
        vel_error  = np.linalg.norm(self.state[half:])
        # was: excursion = np.linalg.norm(self.state[:half] - self.start_pos)
        excursion = pos_error

        docked        = pos_error < self.pos_tolerance
        out_of_bounds = excursion > self.excursion_limit
        timeout       = self.elapsed_time > self.timeout

        delta      = prev_pos_error - pos_error
        terminated = bool(docked or out_of_bounds)
        truncated  = bool(timeout)

        if docked:
            self._advance_curriculum()

        # --- Reward ---
        reward_pos  = ENV_SHAPING_COEFF * delta / self.curriculum_distance
        reward_fuel = -ENV_FUEL_COEFF * np.linalg.norm(action)

        if docked:
            reward_terminal = ENV_BONUS - ENV_VEL_COEFF * vel_error
        elif out_of_bounds:
            reward_terminal = 0.0
        elif truncated: # must penalize otherwise it elarns that the timeout is great compared to going left!
            reward_terminal = - ENV_SHAPING_COEFF * (self.excursion_limit - pos_error) / self.curriculum_distance
        else:
            reward_terminal = 0.0

        reward = reward_pos + reward_fuel + reward_terminal

        observation = self._build_observation()
        info = {
            "distance":        pos_error,
            "docked":          docked,
            "reward_pos":      reward_pos,
            "reward_fuel":     reward_fuel,
            "reward_terminal": reward_terminal,
            "vel_error":       vel_error,
            "delta_v":         np.linalg.norm(action),
            "applied_action":  action.copy(),
            "dv_used":         self.dv_used,
            "dv_remaining":    max(0.0, self.dv_budget - self.dv_used),
            "budget_exhausted": budget_exhausted,
            "curriculum_distance": self.curriculum_distance,
            "excursion":       excursion,
            "excursion_limit": self.excursion_limit,
        }

        return observation, reward, terminated, truncated, info