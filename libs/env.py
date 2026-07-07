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
        curriculum_enabled: bool = ENV_CURRICULUM_ENABLED,
        curriculum_start_distance: float = ENV_CURRICULUM_START_DISTANCE,
        curriculum_max_distance: float = ENV_CURRICULUM_MAX_DISTANCE,
        curriculum_increment: float = ENV_CURRICULUM_INCREMENT,
        curriculum_boundary_mult: float = 2.0,
    ):
        super().__init__()
        self.omega = omega          # mean motion [rad/s]
        self.dt = dt                # propagation time per step [s]
        self.max_dv = max_dv        # max delta-v per axis [m/s]
        self.base_boundary = boundary  # the "full problem" boundary [m], used as a cap
        self.boundary = boundary    # distance from origin that ends episode [m] (updated per-episode under curriculum)
        self.curriculum_boundary_mult = curriculum_boundary_mult  # boundary = mult * curriculum_distance while curriculum is active
        self.timeout = timeout
        self.pos_tolerance = pos_tolerance
        self.vel_tolerance = vel_tolerance
        self.vel_coeff = vel_coeff
        self.fuel_coeff = fuel_coeff
        self.bonus = bonus
        self.dv_budget = dv_budget          # total delta-v available for the episode

        # --- Curriculum learning setup ---
        # The agent starts at a short, easy distance and the starting distance
        # is nudged further out every time it successfully docks, until it
        # reaches the "full" distance defined by ENV_INITIAL_STATE_VBAR.
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_increment = curriculum_increment

        # Direction of the original initial offset (unit vector) — curriculum
        # distances are placed along this same line so the geometry of the
        # problem doesn't change, only its difficulty.
        base_pos = np.asarray(ENV_INITIAL_STATE_VBAR[0:3], dtype=np.float64)
        base_dist = np.linalg.norm(base_pos)
        self._curriculum_dir = base_pos / base_dist if base_dist > 0 else np.array([0.0, 1.0, 0.0])
        self._base_velocity = np.asarray(ENV_INITIAL_STATE_VBAR[3:6], dtype=np.float64)

        # Cap curriculum distance at whichever is smaller: the explicit cap
        # passed in, or the original full distance (so curriculum never
        # overshoots the "real" problem).
        self.curriculum_max_distance = min(curriculum_max_distance, base_dist) if base_dist > 0 else curriculum_max_distance
        self.curriculum_distance = (
            min(curriculum_start_distance, self.curriculum_max_distance)
            if self.curriculum_enabled
            else self.curriculum_max_distance
        )

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

        # Observation: [x, y, z, xdot, ydot, zdot, dv_remaining] -- continuous,
        # unbounded. dv_remaining (the fuel budget left, in m/s) is appended
        # so the policy can actually observe when it's running low/out of
        # fuel, instead of blindly commanding burns the env will zero out.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
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
        """Propagates the 6-dim PHYSICAL state [x,y,z,xdot,ydot,zdot] only.
        If you're passing in the 7-dim observation (physical state +
        dv_remaining), slice off the last column first — this does not
        know about, or update, the fuel budget."""
        self._ensure_stm(state.device, state.dtype)
        v_new = state[..., 3:6] + action
        s0 = torch.cat([state[..., 0:3], v_new], dim=-1)
        return torch.nn.functional.linear(s0, self.STM_T)

    def _initial_state_for_curriculum(self) -> np.ndarray:
        """Build the initial state at the current curriculum distance, along
        the same direction as the original ENV_INITIAL_STATE_VBAR offset."""
        pos = self._curriculum_dir * self.curriculum_distance
        return np.concatenate([pos, self._base_velocity]).astype(np.float64)

    def _build_observation(self) -> np.ndarray:
        """Physical state (6) + remaining delta-v budget (1) = 7-dim obs.
        self.state itself stays 6-dim internally since only the physical
        part is propagated by the STM; dv_remaining is tracked separately
        and appended here for what the agent actually observes."""
        dv_remaining = max(0.0, self.dv_budget - self.dv_used)
        return np.concatenate([self.state, [dv_remaining]]).astype(np.float64)

    def set_curriculum_distance(self, distance: float):
        """Manually set the curriculum distance (e.g. for eval/resuming)."""
        self.curriculum_distance = float(np.clip(distance, 0.0, self.curriculum_max_distance))

    def _advance_curriculum(self):
        if not self.curriculum_enabled:
            return
        self.curriculum_distance = min(
            self.curriculum_distance + self.curriculum_increment,
            self.curriculum_max_distance,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.curriculum_enabled:
            self.state = self._initial_state_for_curriculum()
            # Boundary scales with curriculum distance so early (easy, close)
            # episodes end quickly if the agent flies off, and later (hard,
            # far) episodes get the room they need — capped at the original
            # full-problem boundary.
            self.boundary = min(
                self.base_boundary,
                self.curriculum_distance * self.curriculum_boundary_mult,
            )
        else:
            self.state = np.array(ENV_INITIAL_STATE_VBAR, dtype=np.float64)
            self.boundary = self.base_boundary

        self.best_distance = np.linalg.norm(self.state[0:3])
        self.step_count = 0
        self.wrong_way_counter = 0
        self.elapsed_time = 0.0
        self.dv_used = 0.0

        observation = self._build_observation()
        info = {"curriculum_distance": self.curriculum_distance}
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

        # Apply action and propagate
        self.state[3:6] += action
        self.state = self.STM_np @ self.state
        self.elapsed_time += self.dt

        pos_error = np.linalg.norm(self.state[0:3])
        vel_error = np.linalg.norm(self.state[3:6])

        docked = pos_error < self.pos_tolerance
        out_of_bounds = pos_error > self.boundary
        timeout = self.elapsed_time > self.timeout

        delta = prev_pos_error - pos_error
        terminated = bool(docked or out_of_bounds)
        truncated = bool(timeout)

        # Advance curriculum on a successful dock (only takes effect on the
        # *next* reset() call, so this episode's result is unaffected)
        if docked:
            self._advance_curriculum()

        # --- REWARD COMPONENTS ---

        # 1. Distance progress (DENSE - always active)
        reward_pos = ENV_SHAPING_COEFF * delta

        # 2. Fuel penalty (keep this small)
        reward_fuel = -ENV_FUEL_COEFF * np.linalg.norm(action)

        # 3. Terminal rewards (clear success vs failure signals)
        if docked:
            reward_terminal = ENV_BONUS - ENV_VEL_COEFF * vel_error
        elif out_of_bounds:
            reward_terminal = 0.0
        elif truncated:
            reward_terminal = 0.0
        else:
            reward_terminal = 0.0

        reward = reward_pos + reward_fuel + reward_terminal

        observation = self._build_observation()
        info = {
            "distance": pos_error,
            "docked": docked,
            "reward_pos": reward_pos,
            "reward_fuel": reward_fuel,
            "reward_terminal": reward_terminal,
            "vel_error": vel_error,
            "delta_v": np.linalg.norm(action),
            "applied_action": action.copy(),  # actual post-clip/budget-cap burn — use THIS
                                               # (not the actor's raw commanded action) for
                                               # anything that needs the action that truly
                                               # produced this transition (e.g. replay buffer).
            "dv_used": self.dv_used,
            "dv_remaining": max(0.0, self.dv_budget - self.dv_used),
            "budget_exhausted": budget_exhausted,
            "curriculum_distance": self.curriculum_distance,
            "boundary": self.boundary,
        }

        return observation, reward, terminated, truncated, info