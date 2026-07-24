import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional
from libs.constants import (
    ENV_BONUS,
    ENV_BOUNDARY,
    ENV_DT,
    ENV_DT_PHYS,
    ENV_BURN_DEADZONE_FRAC,
    ENV_FUEL_COEFF,
    ENV_MAX_DV_COEFF,
    ENV_POS_TOLERANCE,
    ENV_TIMEOUT,
    ENV_VEL_COEFF,
    ENV_STOP_COEFF,
    ENV_STOP_VEL_SCALE_FRAC,
    ENV_FUEL_OPT_FLOOR,
    ENV_SHAPING_COEFF,
    OMEGA,
    ENV_CURRICULUM_ENABLED,
    ENV_CURRICULUM_START_DISTANCE,
    ENV_CURRICULUM_MAX_DISTANCE,
    ENV_CURRICULUM_INCREMENT,
    ENV_CURRICULUM_BOUNDARY_MULT,
    SCENARIO,
    RBAR_X_TO_Z_RATIO,
    ENV_RANDOM_DV_REF_MULT,
    ENV_RANDOM_ANGLE_TABLE_N,
    MODE_2D,
    ACTION_DIM,
    ACTION_IMPULSE_DIM,
    ENV_COAST_MIN_UNITS,
    ENV_COAST_MAX_UNITS,
    PHYS_STATE_DIM,
    OBS_DIM,
)
from scipy.linalg import expm

from libs.reference import (
    dv_rbar_strategy_rv,
    dv_vbar_two_impulse_rr,
    optimal_two_impulse_stop_dv_per_m,
    dv_opt_per_m_table,
    dv_opt_per_m_lookup,
)


# ---------------------------------------------------------------------------
#   State transition matrices (Clohessy-Wiltshire)
# ---------------------------------------------------------------------------

def _build_stm_full(omega: float, dt: float) -> np.ndarray:
    """Full 6x6 CW state-transition matrix for one timestep dt."""
    A = np.zeros((6, 6))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    A[3, 5] = 2 * omega        # x_ddot couples z_dot
    A[4, 1] = -omega ** 2      # y_ddot (cross-track, decoupled)
    A[5, 2] = 3 * omega ** 2   # z_ddot couples x/z
    A[5, 3] = -2 * omega
    return expm(A * dt)


def _build_stm_2d(omega: float, dt: float) -> np.ndarray:
    """In-plane (V-bar/R-bar) 4x4 sub-block, state = [x, z, xdot, zdot].
    Cross-track (y) is fully decoupled, so we just drop those rows/cols."""
    idx = np.ix_([0, 2, 3, 5], [0, 2, 3, 5])
    return _build_stm_full(omega, dt)[idx]


class CWRendezvousEnv(gym.Env):
    """
    Clohessy-Wiltshire spacecraft rendezvous environment.

    State (2-D): [x, z, xdot, zdot]. Observation adds cumulative Δv used and a
    braking-phase flag. Action = [impulse per axis, coast-duration scalar]:
    one impulsive burn is applied, then the state coasts ballistically for an
    agent-chosen number of steps. Reaching the target opens a one-step braking
    phase (see _brake_step) instead of ending the episode immediately, so the
    agent fires a dedicated final impulse to null its arrival velocity.

    Reward = dense distance shaping every step, plus three bonuses paid only
    on a completed dock (in _brake_step): a flat dock bonus, a fuel-efficiency
    bonus graded against the true achievable optimum Δv (dv_opt), and a
    stopping bonus graded on arrival speed.

    Scenarios (2-D only):
        "vbar"   : pure V-bar (x) displacement, random sign each episode.
        "rbar"   : coupled x/z displacement, random sign each episode.
        "random" : displacement at a uniformly random angle each episode.
    """

    def __init__(
        self,
        omega: float = OMEGA,
        dt: float = ENV_DT,
        dt_phys: float = ENV_DT_PHYS,
        max_dv_coeff: float = ENV_MAX_DV_COEFF,
        dv_budget_coeff: Optional[float] = None,
        burn_deadzone_frac: float = ENV_BURN_DEADZONE_FRAC,
        boundary: float = ENV_BOUNDARY,
        timeout: float = ENV_TIMEOUT,
        pos_tolerance: float = ENV_POS_TOLERANCE,
        vel_coeff: float = ENV_VEL_COEFF,
        fuel_coeff: float = ENV_FUEL_COEFF,
        stop_coeff: float = ENV_STOP_COEFF,
        stop_vel_scale_frac: float = ENV_STOP_VEL_SCALE_FRAC,
        bonus: float = ENV_BONUS,
        scenario: str = SCENARIO,
        curriculum_enabled: bool = ENV_CURRICULUM_ENABLED,
        curriculum_start_distance: float = ENV_CURRICULUM_START_DISTANCE,
        curriculum_max_distance: float = ENV_CURRICULUM_MAX_DISTANCE,
        curriculum_increment: float = ENV_CURRICULUM_INCREMENT,
        curriculum_boundary_mult: float = ENV_CURRICULUM_BOUNDARY_MULT,
        rbar_x_to_z_ratio: float = RBAR_X_TO_Z_RATIO,
    ):
        super().__init__()
        if scenario not in ("vbar", "rbar", "random"):
            raise ValueError(f"Unknown scenario: {scenario!r}")
        if not MODE_2D and scenario in ("rbar", "random"):
            raise ValueError(f"scenario={scenario!r} is only defined for MODE_2D")

        self.mode_2d = MODE_2D
        self.phys_dim = PHYS_STATE_DIM     # 4 (2D) or 6 (3D)
        self.obs_dim = OBS_DIM             # phys_dim + dv_used + braking flag
        self.action_dim = ACTION_DIM       # impulse_dim + 1 coast scalar
        self.impulse_dim = ACTION_IMPULSE_DIM
        self.scenario = scenario
        self.rbar_x_to_z_ratio = rbar_x_to_z_ratio

        self.omega = omega
        # dt = agent step (one impulse + coast decision); dt_phys = fine
        # physics substep it's decomposed into for docking/OOB sampling.
        self.dt = dt
        self.dt_phys = dt_phys
        n_sub = dt / dt_phys
        self.n_substeps = int(round(n_sub))
        if self.n_substeps < 1 or abs(n_sub - self.n_substeps) > 1e-9:
            raise ValueError(
                f"dt ({dt}) must be a positive integer multiple of dt_phys "
                f"({dt_phys}); got ratio {n_sub}."
            )

        self.max_dv_coeff = max_dv_coeff
        # dv_budget caps TOTAL episode dv_used once set (see set_dv_budget_coeff).
        self.dv_budget_coeff = dv_budget_coeff
        self.dv_budget = None              # set in reset(): dv_budget_coeff * dv_ref
        self.burn_deadzone_frac = burn_deadzone_frac
        self.burn_deadzone = 0.0           # set in reset(): burn_deadzone_frac * max_dv
        self.base_boundary = boundary
        self.excursion_limit = boundary
        self.curriculum_boundary_mult = curriculum_boundary_mult
        self.timeout = timeout
        self.pos_tolerance = pos_tolerance
        self.vel_coeff = vel_coeff
        self.fuel_coeff = fuel_coeff
        self.stop_coeff = stop_coeff
        self.stop_vel_scale_frac = stop_vel_scale_frac
        self.bonus = bonus

        # Curriculum distance is pushed in externally (training.py's
        # CurriculumCallback), not self-advanced — see set_curriculum_distance.
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_increment = curriculum_increment
        self.curriculum_max_distance = curriculum_max_distance
        self.curriculum_distance = (
            min(curriculum_start_distance, curriculum_max_distance)
            if curriculum_enabled
            else curriculum_max_distance
        )

        # STM for the fine physics substep. n_substeps of dt_phys == one dt
        # exactly (expm(A*dt_phys)**n == expm(A*n*dt_phys)); this only buys
        # finer sampling for the docking/OOB test, not different physics.
        if self.mode_2d:
            self.stm = _build_stm_2d(omega, dt_phys)
        else:
            self.stm = _build_stm_full(omega, dt_phys)

        # Precompute stm^k for k = 1..max_coast_substeps so a whole coast (up
        # to ~1 orbit) is one batched matmul in step() instead of a Python loop.
        max_total_substeps = ENV_COAST_MAX_UNITS * self.n_substeps
        powers = np.empty((max_total_substeps, self.phys_dim, self.phys_dim))
        acc = np.eye(self.phys_dim)
        for k in range(max_total_substeps):
            acc = self.stm @ acc
            powers[k] = acc
        self._coast_powers = powers

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float64
        )

        # True achievable optimum Δv per metre of distance (CW is linear, so it
        # scales exactly with distance). Used to grade the fuel bonus. Fixed
        # direction (up to sign) for vbar/rbar -> one scalar; "random" varies
        # the direction every episode, so we build a per-angle table instead
        # and look it up in reset() (see libs/reference.py).
        if self.scenario == "random":
            self._dv_opt_per_m = None
            self._dv_opt_table = dv_opt_per_m_table(omega, ENV_RANDOM_ANGLE_TABLE_N)
        else:
            self._dv_opt_per_m = optimal_two_impulse_stop_dv_per_m(
                self._sample_direction_for(sign=1.0), omega
            )
            self._dv_opt_table = None

        self.state = None
        self._forced_sign = None
        self._forced_angle = None

    # ── Initial-condition sampling ──────────────────────────────────────────

    def _sample_direction_for(self, sign: float) -> np.ndarray:
        """Unit in-plane displacement direction ([x, z]) for a given sign."""
        if self.scenario == "vbar":
            return np.array([sign, 0.0], dtype=np.float64)
        raw = np.array([sign * self.rbar_x_to_z_ratio, -sign], dtype=np.float64)
        return raw / np.linalg.norm(raw)

    def _sample_direction(self) -> np.ndarray:
        """Unit direction for this episode's initial displacement. vbar/rbar
        randomize only the sign (force with reset(options={"sign": +-1}));
        random draws a uniform angle (force with options={"angle": radians})."""
        if self.scenario == "random":
            if self._forced_angle is not None:
                theta = float(self._forced_angle)
            else:
                theta = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        if self._forced_sign is not None:
            sign = float(self._forced_sign)
        else:
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
        return self._sample_direction_for(sign)

    def _sample_initial_state(self) -> np.ndarray:
        direction = self._sample_direction()
        pos = direction * self.curriculum_distance
        vel = np.zeros_like(pos)
        return np.concatenate([pos, vel]).astype(np.float64)

    def set_curriculum_distance(self, distance: float):
        """Set by training.py's CurriculumCallback so every parallel sub-env
        shares one synchronized curriculum distance."""
        self.curriculum_distance = float(
            np.clip(distance, 0.0, self.curriculum_max_distance)
        )

    def set_dv_budget_coeff(self, coeff: Optional[float]):
        """Set by training.py's DvBudgetCurriculumCallback. None disables the
        budget; takes effect from the next reset() onward."""
        self.dv_budget_coeff = None if coeff is None else max(float(coeff), 1e-3)

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        flag = 1.0 if self.braking_phase else 0.0
        return np.concatenate([self.state, [self.dv_used, flag]]).astype(np.float64)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._forced_sign = (options or {}).get("sign")
        self._forced_angle = (options or {}).get("angle")
        self.state = self._sample_initial_state()
        self.excursion_limit = min(
            self.base_boundary,
            self.curriculum_distance * self.curriculum_boundary_mult,
        )
        self.elapsed_time = 0.0
        self.dv_used = 0.0
        # Reaching the target opens a one-step braking phase (see step() /
        # _brake_step) instead of ending the episode immediately.
        self.braking_phase = False

        half = self.phys_dim // 2
        dist = float(np.linalg.norm(self.state[:half]))

        # dv_opt: true achievable optimum for this episode's start, used to
        # grade the fuel bonus (see _brake_step). Fixed scalar for vbar/rbar;
        # looked up from the actual direction for "random".
        if self.scenario == "random":
            direction = self.state[:half] / dist
            dv_opt_per_m = dv_opt_per_m_lookup(direction, self._dv_opt_table)
        else:
            dv_opt_per_m = self._dv_opt_per_m
        self.dv_opt = dv_opt_per_m * dist

        # dv_ref: actuator/observation scale only (max_dv, deadzone, Δv budget,
        # dv_used obs norm) — NOT the fuel-grading target. Analytic reference
        # maneuver for vbar/rbar; derived from dv_opt for "random" since no
        # analytic formula applies there.
        if self.scenario == "vbar":
            self.dv_ref = 0.5 * dv_vbar_two_impulse_rr(dist, self.omega)
        elif self.scenario == "rbar":
            self.dv_ref = dv_rbar_strategy_rv(dist, self.omega)
        else:
            self.dv_ref = ENV_RANDOM_DV_REF_MULT * self.dv_opt

        self.max_dv = self.dv_ref * self.max_dv_coeff
        self.dv_budget = (
            None if self.dv_budget_coeff is None else self.dv_ref * self.dv_budget_coeff
        )
        self.burn_deadzone = self.burn_deadzone_frac * self.max_dv

        observation = self._build_observation()
        info = {"curriculum_distance": self.curriculum_distance, "state": self.state.copy()}
        return observation, info

    @staticmethod
    def _segment_closest_to_origin(p0: np.ndarray, p1: np.ndarray):
        """Closest point on segment [p0, p1] to the origin, and its distance —
        catches a fast fly-through the tolerance circle between samples."""
        d = p1 - p0
        denom = float(d @ d)
        if denom <= 0.0:
            closest = p0
        else:
            t = float(np.clip(-(p0 @ d) / denom, 0.0, 1.0))
            closest = p0 + t * d
        return closest, float(np.linalg.norm(closest))

    def _coast_units_from_cmd(self, coast_cmd: float) -> int:
        """Map the coast action scalar [-1, 1] to an integer number of
        agent-dt coast units in [ENV_COAST_MIN_UNITS, ENV_COAST_MAX_UNITS]."""
        frac01 = (float(np.clip(coast_cmd, -1.0, 1.0)) + 1.0) * 0.5
        span = ENV_COAST_MAX_UNITS - ENV_COAST_MIN_UNITS
        return ENV_COAST_MIN_UNITS + int(round(frac01 * span))

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if self.braking_phase:
            return self._brake_step(action)

        coast_units = self._coast_units_from_cmd(action[self.impulse_dim])
        u = np.clip(action[: self.impulse_dim], -1.0, 1.0)
        action = u * self.max_dv

        # Below-deadzone burns are zeroed (free coasting); otherwise clip to
        # whatever remains of the total-episode Δv budget, if one is active.
        burn = float(np.linalg.norm(action))
        if burn < self.burn_deadzone:
            action = np.zeros_like(action)
            burn = 0.0
        elif self.dv_budget is not None:
            remaining = self.dv_budget - self.dv_used
            if remaining <= 0.0:
                action = np.zeros_like(action)
                burn = 0.0
            elif burn > remaining:
                action = action * (remaining / burn)
                burn = remaining
        self.dv_used += burn

        half = self.phys_dim // 2
        prev_pos_error = np.linalg.norm(self.state[:half])

        # Impulse changes velocity only (instantaneous burn); position moves
        # only during the coast that follows.
        self.state[half:] += action
        s0 = self.state.copy()
        total_substeps = coast_units * self.n_substeps

        # Whole coast as one batched matmul: seq[i] = stm^(i+1) @ s0.
        seq = self._coast_powers[:total_substeps] @ s0
        pos_seq = seq[:, :half]
        prev_seq = np.vstack([s0[:half], pos_seq[:-1]])

        # Vectorized closest-approach test over every substep chord.
        dvec = pos_seq - prev_seq
        denom = np.einsum("ij,ij->i", dvec, dvec)
        with np.errstate(invalid="ignore", divide="ignore"):
            tparam = np.where(
                denom > 0.0,
                np.clip(-np.einsum("ij,ij->i", prev_seq, dvec) / denom, 0.0, 1.0),
                0.0,
            )
        closest = prev_seq + tparam[:, None] * dvec
        closest_dist = np.linalg.norm(closest, axis=1)
        new_dist = np.linalg.norm(pos_seq, axis=1)

        # First substep index at which each terminal condition fires.
        BIG = total_substeps + 1
        dock_hits = np.flatnonzero(closest_dist < self.pos_tolerance)
        oob_hits = np.flatnonzero(new_dist > self.excursion_limit)
        i_dock = int(dock_hits[0]) if dock_hits.size else BIG
        i_oob = int(oob_hits[0]) if oob_hits.size else BIG
        thresh = (self.timeout - self.elapsed_time) / self.dt_phys
        i_timeout = int(np.floor(thresh))
        if not (0 <= i_timeout < total_substeps):
            i_timeout = BIG

        # Earliest event wins; ties break dock > out-of-bounds > timeout.
        i_event = min(i_dock, i_oob, i_timeout)
        if i_event == BIG:
            i_end = total_substeps - 1
            docked = out_of_bounds = timeout = False
        else:
            i_end = i_event
            docked = i_event == i_dock
            out_of_bounds = (not docked) and i_event == i_oob
            timeout = (not docked) and (not out_of_bounds) and i_event == i_timeout

        self.elapsed_time += (i_end + 1) * self.dt_phys
        self.state = seq[i_end].copy()
        if docked:
            # Snap to the true closest-approach point; keep the coasted
            # velocity for the terminal vel_error / stop bonus.
            self.state[:half] = closest[i_end]

        # Coarse agent-dt samples of the coast, for trajectory plotting.
        substates = [seq[i].copy() for i in range(self.n_substeps - 1, i_end, self.n_substeps)]
        substates.append(self.state.copy())

        pos_error = np.linalg.norm(self.state[:half])
        vel_error = np.linalg.norm(self.state[half:])
        delta = prev_pos_error - pos_error

        # Dense shaping only; terminal bonuses are paid in _brake_step once the
        # dock is actually completed (docking here just opens that phase).
        reward_pos = ENV_SHAPING_COEFF * delta / self.curriculum_distance
        reward_fuel = reward_stop = reward_terminal = 0.0

        if docked:
            self.braking_phase = True
            terminated = truncated = False
        else:
            terminated = bool(out_of_bounds)
            truncated = bool(timeout)

        reward = reward_pos

        observation = self._build_observation()
        info = {
            "state": self.state.copy(),
            "substates": substates,
            "coast_units": coast_units,
            "distance": pos_error,
            "docked": False,  # credited on the terminal _brake_step instead
            "reward_pos": reward_pos,
            "reward_fuel": reward_fuel,
            "reward_stop": reward_stop,
            "reward_terminal": reward_terminal,
            "vel_error": vel_error,
            "delta_v": np.linalg.norm(action),
            "applied_action": action.copy(),
            "dv_used": self.dv_used,
            "dv_ref": self.dv_ref,
            "dv_opt": self.dv_opt,
            "dv_budget_coeff": self.dv_budget_coeff,
            "curriculum_distance": self.curriculum_distance,
            "excursion_limit": self.excursion_limit,
            "braking_phase": self.braking_phase,
        }
        return observation, reward, terminated, truncated, info

    def _brake_step(self, action: np.ndarray):
        """Terminal braking impulse, entered the step after the target is
        reached. No coast: applied instantly at the target, episode ends here.
        The deadzone is skipped (the optimal brake can be smaller than it) but
        the Δv budget still applies."""
        half = self.phys_dim // 2
        u = np.clip(action[: self.impulse_dim], -1.0, 1.0)
        impulse = u * self.max_dv
        burn = float(np.linalg.norm(impulse))

        if self.dv_budget is not None:
            remaining = self.dv_budget - self.dv_used
            if remaining <= 0.0:
                impulse = np.zeros_like(impulse)
                burn = 0.0
            elif burn > remaining:
                impulse = impulse * (remaining / burn)
                burn = remaining
        self.dv_used += burn

        # Velocity change only; position stays at the snapped dock point.
        self.state[half:] += impulse
        pos_error = float(np.linalg.norm(self.state[:half]))
        vel_error = float(np.linalg.norm(self.state[half:]))

        # Fuel bonus graded against dv_opt (true optimum for this geometry),
        # floored at ENV_FUEL_OPT_FLOOR (see constants.py — do not remove the
        # floor or raise it to 1.0, both break the incentive). Stop bonus
        # graded on post-brake speed.
        v_scale = self.stop_vel_scale_frac * self.dv_ref
        stop_quality = 1.0 / (1.0 + vel_error / v_scale)
        ratio_opt = max(self.dv_used / self.dv_opt, ENV_FUEL_OPT_FLOOR)
        reward_fuel = self.fuel_coeff / ratio_opt
        reward_stop = self.stop_coeff * stop_quality
        reward_terminal = self.bonus
        reward_pos = 0.0
        reward = reward_pos + reward_fuel + reward_stop + reward_terminal

        self.braking_phase = False
        observation = self._build_observation()
        info = {
            "state": self.state.copy(),
            "substates": [self.state.copy()],
            "coast_units": 0,
            "distance": pos_error,
            "docked": True,
            "reward_pos": reward_pos,
            "reward_fuel": reward_fuel,
            "reward_stop": reward_stop,
            "reward_terminal": reward_terminal,
            "vel_error": vel_error,
            "delta_v": burn,
            "applied_action": impulse.copy(),
            "dv_used": self.dv_used,
            "dv_ref": self.dv_ref,
            "dv_opt": self.dv_opt,
            "dv_budget_coeff": self.dv_budget_coeff,
            "curriculum_distance": self.curriculum_distance,
            "excursion_limit": self.excursion_limit,
            "braking_phase": False,
        }
        return observation, reward, True, False, info
