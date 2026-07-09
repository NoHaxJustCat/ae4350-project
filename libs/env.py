import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
from libs.constants import (
    ENV_BONUS,
    ENV_BOUNDARY,
    ENV_DT,
    ENV_FUEL_COEFF,
    ENV_MAX_DV,
    ENV_POS_TOLERANCE,
    ENV_TIMEOUT,
    ENV_VEL_COEFF,
    ENV_VEL_TOLERANCE,
    ENV_SHAPING_COEFF,
    OMEGA,
    ENV_DV_CEILING_MULT,
    ENV_DV_CEILING_FLOOR,
    ENV_CURRICULUM_ENABLED,
    ENV_CURRICULUM_START_DISTANCE,
    ENV_CURRICULUM_MAX_DISTANCE,
    ENV_CURRICULUM_INCREMENT,
    SCENARIO,
    RBAR_X_TO_Z_RATIO,
    MODE_2D,
    ACTION_DIM,
    PHYS_STATE_DIM,
    OBS_DIM,
)
from libs.reference import reference_dv_for_scenario
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

    Observation appends cumulative Δv used so far so the policy can
    condition on how much fuel it has already spent:
        2-D obs : [x, z, ẋ, ż, dv_used]           (5-dim)
        3-D obs : [x, y, z, ẋ, ẏ, ż, dv_used]     (7-dim)

    Action:
        2-D : [dvx, dvz]   (2-dim)
        3-D : [dvx, dvy, dvz]   (3-dim)

    Scenario (2-D only, selects the initial-condition family; see
    CLAUDE.md goals 1 & 2). Sign/quadrant is randomized every reset() so a
    single model learns both directions rather than only ever seeing one:
        "vbar" : pure V-bar displacement, x = ±distance, z = 0
        "rbar" : coupled displacement, (x, z) = ±distance · (ratio, -1)/norm
                 restricted to the two mirrored quadrants (+x,-z)/(-x,+z)
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
        scenario: str = SCENARIO,
        dv_ceiling_mult: float = ENV_DV_CEILING_MULT,
        dv_ceiling_floor: float = ENV_DV_CEILING_FLOOR,
        curriculum_enabled: bool = ENV_CURRICULUM_ENABLED,
        curriculum_start_distance: float = ENV_CURRICULUM_START_DISTANCE,
        curriculum_max_distance: float = ENV_CURRICULUM_MAX_DISTANCE,
        curriculum_increment: float = ENV_CURRICULUM_INCREMENT,
        curriculum_boundary_mult: float = 2.0,
        rbar_x_to_z_ratio: float = RBAR_X_TO_Z_RATIO,
    ):
        super().__init__()
        if scenario not in ("vbar", "rbar"):
            raise ValueError(f"Unknown scenario: {scenario!r}")
        if not MODE_2D and scenario == "rbar":
            raise ValueError("scenario='rbar' is only defined for MODE_2D")

        self.mode_2d = MODE_2D
        self.phys_dim = PHYS_STATE_DIM   # 4 (2D) or 6 (3D)
        self.obs_dim  = OBS_DIM          # 5 (2D) or 7 (3D)
        self.action_dim = ACTION_DIM     # 2 (2D) or 3 (3D)
        self.scenario = scenario
        self.rbar_x_to_z_ratio = rbar_x_to_z_ratio

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
        self.dv_ceiling_mult = dv_ceiling_mult
        self.dv_ceiling_floor = dv_ceiling_floor

        # --- Curriculum ---
        # NOTE: this env does NOT self-advance the curriculum anymore. With
        # N parallel sub-envs each doing their own local "3 consecutive
        # docks" count, curriculum_distance drifted out of sync across
        # workers with no visibility into it — that's what looked like
        # "random" spawn distances. A single authority (CurriculumCallback
        # in training.py) now tracks dock rate across ALL envs and pushes
        # one shared distance to every sub-env via set_curriculum_distance().
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_increment = curriculum_increment
        self.curriculum_max_distance = curriculum_max_distance
        self.curriculum_distance = (
            min(curriculum_start_distance, curriculum_max_distance)
            if curriculum_enabled
            else curriculum_max_distance
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
        self._forced_sign = None

    # ── Torch helpers ────────────────────────────────────────────────────────

    def _ensure_stm(self, device, dtype):
        if self.STM.device != device or self.STM.dtype != dtype:
            self.STM   = self.STM.to(device=device, dtype=dtype)
            self.STM_T = self.STM.t().contiguous()

    def propagate_torch(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Propagate the PHYSICAL state (4-dim 2D, 6-dim 3D) only.
        Strip the obs dv_used column before calling."""
        self._ensure_stm(state.device, state.dtype)
        # velocity indices: last half of physical state
        half = self.phys_dim // 2
        v_new = state[..., half:] + action
        s0 = torch.cat([state[..., :half], v_new], dim=-1)
        return torch.nn.functional.linear(s0, self.STM_T)

    # ── Initial-condition sampling ───────────────────────────────────────────

    def _sample_direction(self) -> np.ndarray:
        """Unit direction vector for this episode's initial displacement.
        Sign/quadrant is randomized so a single model sees both cases
        required by CLAUDE.md goals 1 & 2, instead of always the same one.
        Pass reset(options={"sign": +1 or -1}) to force a side for eval."""
        if self._forced_sign is not None:
            sign = float(self._forced_sign)
        else:
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
        if self.scenario == "vbar":
            return np.array([sign, 0.0], dtype=np.float64)
        # "rbar": (+x,-z) or (-x,+z) — opposite-sign coupled displacement
        raw = np.array([sign * self.rbar_x_to_z_ratio, -sign], dtype=np.float64)
        return raw / np.linalg.norm(raw)

    def _sample_initial_state(self) -> np.ndarray:
        direction = self._sample_direction()
        pos = direction * self.curriculum_distance
        vel = np.zeros_like(pos)
        return np.concatenate([pos, vel]).astype(np.float64)

    def set_curriculum_distance(self, distance: float):
        """External control hook — called by training.py's CurriculumCallback
        (via VecEnv.env_method) so every parallel sub-env shares one
        synchronized curriculum distance instead of drifting independently."""
        self.curriculum_distance = float(
            np.clip(distance, 0.0, self.curriculum_max_distance)
        )

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        return np.concatenate([self.state, [self.dv_used]]).astype(np.float64)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._forced_sign = (options or {}).get("sign")
        self.state = self._sample_initial_state()
        self.excursion_limit = min(
            self.base_boundary,
            self.curriculum_distance * self.curriculum_boundary_mult,
        )

        # Position part is the first half of the physical state
        half = self.phys_dim // 2
        self.start_pos = self.state[:half].copy()

        self.start_distance  = np.linalg.norm(self.state[:half])
        self.best_distance   = self.start_distance
        self.worst_distance  = self.start_distance
        self.step_count      = 0
        self.elapsed_time    = 0.0
        self.dv_used         = 0.0
        self.episode_curriculum_distance = self.curriculum_distance

        dx, dz = (abs(self.state[0]), abs(self.state[1])) if self.mode_2d else (abs(self.state[0]), 0.0)
        ref_dv = reference_dv_for_scenario(self.scenario, dx, dz, self.omega)
        self.dv_ceiling = max(self.dv_ceiling_mult * ref_dv, self.dv_ceiling_floor)

        observation = self._build_observation()
        info = {"curriculum_distance": self.curriculum_distance}
        return observation, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(action, -self.max_dv, self.max_dv)
        self.dv_used += np.linalg.norm(action)

        half = self.phys_dim // 2
        prev_pos_error = np.linalg.norm(self.state[:half])

        # Propagate: apply Δv to velocity, then advance with STM
        self.state[half:] += action
        self.state = self.STM_np @ self.state
        self.elapsed_time += self.dt

        pos_error  = np.linalg.norm(self.state[:half])
        vel_error  = np.linalg.norm(self.state[half:])
        excursion  = pos_error

        docked          = pos_error < self.pos_tolerance
        out_of_bounds   = excursion > self.excursion_limit
        timeout         = self.elapsed_time > self.timeout
        # Soft safety valve (NOT a hard action clip): once an episode has
        # burned far more Δv than any sane strategy would need and still
        # hasn't docked, stop wasting steps on it. Generous multiple/floor
        # (see constants.py) means this never fires on a competent — even
        # somewhat wasteful — trajectory.
        dv_ceiling_hit  = (not docked) and (self.dv_used > self.dv_ceiling)

        delta      = prev_pos_error - pos_error
        terminated = bool(docked or out_of_bounds)
        truncated  = bool(timeout or dv_ceiling_hit)

        # --- Reward ---
        reward_pos  = ENV_SHAPING_COEFF * delta / self.curriculum_distance
        reward_fuel = -self.fuel_coeff * np.linalg.norm(action)

        # Burning through the entire fuel ceiling is always at least as bad
        # as the worst possible "never improved" milestone malus (see below),
        # regardless of how many steps it took to get there. This is what
        # actually closes the fast-burn exploit: previously a short, fast
        # burn-to-ceiling could dodge the malus by ending the episode before
        # worst_distance had a chance to grow; scaling the malus by step
        # count also worked but diluted its guiding signal for genuine slow
        # divergence. Charging the worst-case malus directly to reward_fuel
        # whenever the ceiling is hit removes the incentive to race there
        # without softening the milestone term itself.
        if dv_ceiling_hit:
            max_milestone_penalty = (
                ENV_SHAPING_COEFF * (self.excursion_limit - self.start_distance)
                / self.episode_curriculum_distance
            )
            reward_fuel -= max_milestone_penalty

        if docked:
            reward_terminal = self.bonus - self.vel_coeff * vel_error
        elif out_of_bounds:
            reward_terminal = 0.0
        elif truncated:  # penalize so the agent can't "hide" by running out the clock / burning wastefully
            reward_terminal = -ENV_SHAPING_COEFF * (self.excursion_limit - pos_error) / self.curriculum_distance
        else:
            reward_terminal = 0.0

        if pos_error > self.worst_distance:
            self.worst_distance = pos_error

        if pos_error < self.best_distance:
            improvement = self.best_distance - pos_error
            reward_milestone = ENV_SHAPING_COEFF * improvement / self.episode_curriculum_distance
            self.best_distance = pos_error
        elif (terminated or truncated) and self.best_distance >= self.start_distance:
            # Never once improved on the starting distance this whole
            # episode (mid-episode excursions away from the target are NOT
            # penalized — only fires at the terminal step, so a trajectory
            # that legitimately needs to move away first before it can
            # approach is untouched as long as it improves on start_distance
            # by the end). The fast-burn-to-ceiling dodge this malus used to
            # be vulnerable to is now closed separately by the dv_ceiling_hit
            # charge to reward_fuel above, so this stays a direct function of
            # how far it drifted — no step-count normalization needed.
            reward_milestone = (
                -ENV_SHAPING_COEFF * (self.worst_distance - self.start_distance)
                / self.episode_curriculum_distance
            )
        else:
            reward_milestone = 0.0

        reward = reward_pos + reward_fuel + reward_terminal + reward_milestone

        observation = self._build_observation()
        info = {
            "state":           self.state.copy(),
            "distance":        pos_error,
            "docked":          docked,
            "reward_pos":      reward_pos,
            "reward_fuel":     reward_fuel,
            "reward_terminal": reward_terminal,
            "reward_milestone": reward_milestone,
            "vel_error":       vel_error,
            "delta_v":         np.linalg.norm(action),
            "applied_action":  action.copy(),
            "dv_used":         self.dv_used,
            "dv_ceiling":      self.dv_ceiling,
            "dv_ceiling_hit":  dv_ceiling_hit,
            "curriculum_distance": self.curriculum_distance,
            "excursion":       excursion,
            "excursion_limit": self.excursion_limit,
        }

        return observation, reward, terminated, truncated, info
