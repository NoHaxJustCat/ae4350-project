"""Clohessy-Wiltshire rendezvous environment."""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    ACTION_DIM, ACTION_IMPULSE_DIM, ENV_ANGLE_CURRICULUM_MAX_DEG, ENV_BONUS,
    ENV_BOUNDARY, ENV_BURN_DEADZONE_FRAC, ENV_COAST_MAX_UNITS, ENV_COAST_MIN_UNITS,
    ENV_CURRICULUM_BOUNDARY_MULT, ENV_CURRICULUM_ENABLED, ENV_CURRICULUM_MAX_DISTANCE,
    ENV_CURRICULUM_START_DISTANCE, ENV_DT, ENV_DT_PHYS, ENV_FUEL_COEFF,
    ENV_FUEL_OPT_FLOOR, ENV_MAX_DV_COEFF, ENV_POS_TOLERANCE, ENV_RANDOM_ANGLE_TABLE_N,
    ENV_RANDOM_DV_REF_MULT, ENV_SHAPING_COEFF, ENV_STOP_COEFF, ENV_STOP_VEL_SCALE_FRAC,
    ENV_TIMEOUT, MODE_2D, OBS_DIM, OMEGA, PHYS_STATE_DIM, RBAR_X_TO_Z_RATIO, SCENARIO,
)
from lib.astro.dynamics import stm_2d, stm_full
from lib.astro.reference import (
    dv_opt_per_m_lookup, dv_opt_per_m_table, dv_rbar_strategy_rv,
    dv_vbar_two_impulse_rr, optimal_two_impulse_stop_dv_per_m,
)


class CWRendezvousEnv(gym.Env):
    """State (2-D): [x, z, xdot, zdot]; observation adds cumulative dv and a
    braking-phase flag. Action = [impulse per axis, coast-duration scalar]: one
    impulsive burn, then a ballistic coast of agent-chosen length. Reaching the
    target opens a one-step braking phase so the agent fires a dedicated final
    impulse to null its arrival velocity.

    Reward = dense distance shaping each step, plus a dock bonus, a fuel bonus
    graded against the true achievable optimum, and a stopping bonus graded on
    arrival speed -- the last three paid only on a completed dock.
    """

    def __init__(
        self,
        omega: float = OMEGA,
        dt: float = ENV_DT,
        dt_phys: float = ENV_DT_PHYS,
        max_dv_coeff: float = ENV_MAX_DV_COEFF,
        burn_deadzone_frac: float = ENV_BURN_DEADZONE_FRAC,
        boundary: float = ENV_BOUNDARY,
        timeout: float = ENV_TIMEOUT,
        pos_tolerance: float = ENV_POS_TOLERANCE,
        fuel_coeff: float = ENV_FUEL_COEFF,
        stop_coeff: float = ENV_STOP_COEFF,
        stop_vel_scale_frac: float = ENV_STOP_VEL_SCALE_FRAC,
        bonus: float = ENV_BONUS,
        scenario: str = SCENARIO,
        curriculum_enabled: bool = ENV_CURRICULUM_ENABLED,
        curriculum_start_distance: float = ENV_CURRICULUM_START_DISTANCE,
        curriculum_max_distance: float = ENV_CURRICULUM_MAX_DISTANCE,
        curriculum_boundary_mult: float = ENV_CURRICULUM_BOUNDARY_MULT,
        angle_half_width_deg: float = ENV_ANGLE_CURRICULUM_MAX_DEG,
        rbar_x_to_z_ratio: float = RBAR_X_TO_Z_RATIO,
        dv_ref_mult: float = None,
    ):
        super().__init__()
        if scenario not in ("vbar", "rbar", "random"):
            raise ValueError(f"Unknown scenario: {scenario!r}")
        if not MODE_2D and scenario in ("rbar", "random"):
            raise ValueError(f"scenario={scenario!r} is only defined for MODE_2D")

        self.phys_dim = PHYS_STATE_DIM
        self.impulse_dim = ACTION_IMPULSE_DIM
        self.scenario = scenario
        self.rbar_x_to_z_ratio = rbar_x_to_z_ratio
        self.omega = omega

        # dt = agent decision interval; dt_phys = the fine substep it is
        # decomposed into for docking / out-of-bounds sampling.
        self.dt, self.dt_phys = dt, dt_phys
        n_sub = dt / dt_phys
        self.n_substeps = int(round(n_sub))
        if self.n_substeps < 1 or abs(n_sub - self.n_substeps) > 1e-9:
            raise ValueError(f"dt ({dt}) must be an integer multiple of dt_phys ({dt_phys})")

        # None -> each scenario's analytic reference. A float sizes the actuator
        # as a multiple of the TRUE optimum instead, for every scenario, the way
        # "random" already does. vbar's analytic reference gives max_dv =
        # 3.54*dv_opt, which is the oversized cap that was removed for "random";
        # dv_ref_mult=1.0 gives 1.5*dv_opt to match it.
        self.dv_ref_mult = dv_ref_mult
        self.max_dv_coeff = max_dv_coeff
        self.burn_deadzone_frac = burn_deadzone_frac
        self.burn_deadzone = 0.0
        self.base_boundary = boundary
        self.excursion_limit = boundary
        self.curriculum_boundary_mult = curriculum_boundary_mult
        self.timeout = timeout
        self.pos_tolerance = pos_tolerance
        self.fuel_coeff = fuel_coeff
        self.stop_coeff = stop_coeff
        self.stop_vel_scale_frac = stop_vel_scale_frac
        self.bonus = bonus

        # Curriculum state is pushed in by training.py's callbacks, not
        # self-advanced, so every parallel sub-env stays synchronized.
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_max_distance = curriculum_max_distance
        self.curriculum_distance = (min(curriculum_start_distance, curriculum_max_distance)
                                    if curriculum_enabled else curriculum_max_distance)
        self.angle_half_width_deg = float(np.clip(angle_half_width_deg, 0.0,
                                                  ENV_ANGLE_CURRICULUM_MAX_DEG))

        self.stm = (stm_2d if MODE_2D else stm_full)(omega, dt_phys)
        # Precompute stm^k so a whole coast is one batched matmul.
        powers = np.empty((ENV_COAST_MAX_UNITS * self.n_substeps, self.phys_dim, self.phys_dim))
        acc = np.eye(self.phys_dim)
        for k in range(len(powers)):
            acc = self.stm @ acc
            powers[k] = acc
        self._coast_powers = powers

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float64)

        # True optimum dv per metre. Fixed direction for vbar/rbar -> one
        # scalar; "random" varies per episode -> interpolate a per-angle table.
        if scenario == "random":
            self._dv_opt_per_m = None
            self._dv_opt_table = dv_opt_per_m_table(omega, ENV_RANDOM_ANGLE_TABLE_N)
        else:
            self._dv_opt_per_m = optimal_two_impulse_stop_dv_per_m(
                self._direction_for(sign=1.0), omega)
            self._dv_opt_table = None

        self.state = None
        self._forced_sign = None
        self._forced_angle = None

    # -- curriculum setters (called via env_method on every sub-env) ----------

    def set_curriculum_distance(self, distance: float):
        self.curriculum_distance = float(np.clip(distance, 0.0, self.curriculum_max_distance))

    def set_angle_half_width(self, half_width_deg: float):
        """"random" only; a harmless no-op elsewhere."""
        self.angle_half_width_deg = float(np.clip(half_width_deg, 0.0,
                                                  ENV_ANGLE_CURRICULUM_MAX_DEG))

    # -- initial conditions ---------------------------------------------------

    def _direction_for(self, sign: float) -> np.ndarray:
        if self.scenario == "vbar":
            return np.array([sign, 0.0])
        raw = np.array([sign * self.rbar_x_to_z_ratio, -sign])
        return raw / np.linalg.norm(raw)

    def _sample_direction(self) -> np.ndarray:
        """vbar/rbar randomize only the sign; random draws from the curriculum
        sector centred on a V-bar axis (either sign, so half-width 90 deg is
        exactly the uniform draw). Force with reset(options=...)."""
        if self.scenario == "random":
            if self._forced_angle is not None:
                theta = float(self._forced_angle)
            else:
                hw = np.radians(self.angle_half_width_deg)
                base = 0.0 if self.np_random.random() < 0.5 else np.pi
                theta = base + float(self.np_random.uniform(-hw, hw))
            return np.array([np.cos(theta), np.sin(theta)])
        sign = (self._forced_sign if self._forced_sign is not None
                else (1.0 if self.np_random.random() < 0.5 else -1.0))
        return self._direction_for(float(sign))

    # -- gym API --------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        flag = 1.0 if self.braking_phase else 0.0
        return np.concatenate([self.state, [self.dv_used, flag]])

    def _info(self, **extra) -> dict:
        base = {
            "state": self.state.copy(),
            "distance": float(np.linalg.norm(self.state[:self.phys_dim // 2])),
            "dv_used": self.dv_used,
            "dv_ref": self.dv_ref,
            "dv_opt": self.dv_opt,
            "curriculum_distance": self.curriculum_distance,
            "angle_half_width_deg": self.angle_half_width_deg,
            "excursion_limit": self.excursion_limit,
            "braking_phase": self.braking_phase,
        }
        base.update(extra)
        return base

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        opts = options or {}
        self._forced_sign = opts.get("sign")
        self._forced_angle = opts.get("angle")

        direction = self._sample_direction()
        pos = direction * self.curriculum_distance
        self.state = np.concatenate([pos, np.zeros_like(pos)])
        self.excursion_limit = min(self.base_boundary,
                                   self.curriculum_distance * self.curriculum_boundary_mult)
        self.elapsed_time = 0.0
        self.dv_used = 0.0
        self.braking_phase = False

        half = self.phys_dim // 2
        dist = float(np.linalg.norm(self.state[:half]))
        if self.scenario == "random":
            per_m = dv_opt_per_m_lookup(self.state[:half] / dist, self._dv_opt_table)
        else:
            per_m = self._dv_opt_per_m
        self.dv_opt = per_m * dist

        # dv_ref is the actuator/observation scale only, NOT the fuel target.
        if self.dv_ref_mult is not None:
            self.dv_ref = self.dv_ref_mult * self.dv_opt
        elif self.scenario == "vbar":
            self.dv_ref = 0.5 * dv_vbar_two_impulse_rr(dist, self.omega)
        elif self.scenario == "rbar":
            self.dv_ref = dv_rbar_strategy_rv(dist, self.omega)
        else:
            self.dv_ref = ENV_RANDOM_DV_REF_MULT * self.dv_opt

        self.max_dv = self.dv_ref * self.max_dv_coeff
        self.burn_deadzone = self.burn_deadzone_frac * self.max_dv
        return self._observation(), self._info(docked=False)

    def _coast_units(self, coast_cmd: float) -> int:
        frac = (float(np.clip(coast_cmd, -1.0, 1.0)) + 1.0) * 0.5
        span = ENV_COAST_MAX_UNITS - ENV_COAST_MIN_UNITS
        return ENV_COAST_MIN_UNITS + int(round(frac * span))

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if self.braking_phase:
            return self._brake_step(action)

        half = self.phys_dim // 2
        coast_units = self._coast_units(action[self.impulse_dim])
        impulse = np.clip(action[:self.impulse_dim], -1.0, 1.0) * self.max_dv
        burn = float(np.linalg.norm(impulse))
        if burn < self.burn_deadzone:      # below the minimum impulse bit
            impulse, burn = np.zeros_like(impulse), 0.0
        self.dv_used += burn

        prev_error = np.linalg.norm(self.state[:half])
        self.state[half:] += impulse       # impulsive: velocity only
        s0 = self.state.copy()
        total_substeps = coast_units * self.n_substeps

        seq = self._coast_powers[:total_substeps] @ s0
        pos_seq = seq[:, :half]
        prev_seq = np.vstack([s0[:half], pos_seq[:-1]])

        # Vectorized closest-approach over every substep chord, so a fast
        # fly-through of the tolerance circle between samples still counts.
        d = pos_seq - prev_seq
        denom = np.einsum("ij,ij->i", d, d)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(denom > 0.0,
                         np.clip(-np.einsum("ij,ij->i", prev_seq, d) / denom, 0.0, 1.0), 0.0)
        closest = prev_seq + t[:, None] * d
        closest_dist = np.linalg.norm(closest, axis=1)
        new_dist = np.linalg.norm(pos_seq, axis=1)

        BIG = total_substeps + 1
        dock_hits = np.flatnonzero(closest_dist < self.pos_tolerance)
        oob_hits = np.flatnonzero(new_dist > self.excursion_limit)
        i_dock = int(dock_hits[0]) if dock_hits.size else BIG
        i_oob = int(oob_hits[0]) if oob_hits.size else BIG
        i_timeout = int(np.floor((self.timeout - self.elapsed_time) / self.dt_phys))
        if not (0 <= i_timeout < total_substeps):
            i_timeout = BIG

        # Earliest event wins; ties break dock > out-of-bounds > timeout.
        i_event = min(i_dock, i_oob, i_timeout)
        if i_event == BIG:
            i_end, docked, out_of_bounds, timeout = total_substeps - 1, False, False, False
        else:
            i_end = i_event
            docked = i_event == i_dock
            out_of_bounds = (not docked) and i_event == i_oob
            timeout = not (docked or out_of_bounds)

        self.elapsed_time += (i_end + 1) * self.dt_phys
        self.state = seq[i_end].copy()
        if docked:
            self.state[:half] = closest[i_end]   # snap to true closest approach
            self.braking_phase = True

        substates = [seq[i].copy() for i in range(self.n_substeps - 1, i_end, self.n_substeps)]
        substates.append(self.state.copy())

        reward = ENV_SHAPING_COEFF * (prev_error - np.linalg.norm(self.state[:half])) / self.curriculum_distance
        info = self._info(
            docked=False,                    # credited on the terminal brake
            substates=substates, coast_units=coast_units,
            reward_pos=reward, reward_fuel=0.0, reward_stop=0.0, reward_terminal=0.0,
            vel_error=float(np.linalg.norm(self.state[half:])),
            delta_v=burn, applied_action=impulse.copy(),
        )
        return self._observation(), reward, bool(out_of_bounds), bool(timeout), info

    def _brake_step(self, action: np.ndarray):
        """Terminal braking impulse, applied instantly at the target with no
        coast. The deadzone is skipped -- the optimal brake can be smaller."""
        half = self.phys_dim // 2
        impulse = np.clip(action[:self.impulse_dim], -1.0, 1.0) * self.max_dv
        burn = float(np.linalg.norm(impulse))
        self.dv_used += burn
        self.state[half:] += impulse

        vel_error = float(np.linalg.norm(self.state[half:]))
        stop_quality = 1.0 / (1.0 + vel_error / (self.stop_vel_scale_frac * self.dv_ref))
        reward_fuel = self.fuel_coeff / max(self.dv_used / self.dv_opt, ENV_FUEL_OPT_FLOOR)
        reward_stop = self.stop_coeff * stop_quality
        reward = reward_fuel + reward_stop + self.bonus

        self.braking_phase = False
        info = self._info(
            docked=True, substates=[self.state.copy()], coast_units=0,
            reward_pos=0.0, reward_fuel=reward_fuel, reward_stop=reward_stop,
            reward_terminal=self.bonus, vel_error=vel_error,
            delta_v=burn, applied_action=impulse.copy(),
        )
        return self._observation(), reward, True, False, info
