"""
Single-process batched VecEnv for CWRendezvousEnv.

Why this exists (see remote_training.ps1 for the measured history): both
SubprocVecEnv (serial pipe round-trips in the main process) and DummyVecEnv
(serial Python loop over n_envs .step() calls) pay per-env Python/IPC
overhead for an env whose physics is one 4x4 matmul. The fix is to stop
looping over envs in Python entirely: hold all n_envs physical states as one
(n_envs, 4) array and advance them with a single `states @ STM.T` per
vec-step, with every reward/termination term computed as elementwise numpy
ops over the batch.

This class replicates, vectorized and to floating-point tolerance, the exact
per-env wrapper stack training.py builds for dummy/subproc:

    Monitor(NormalizedObsEnv(CanonicalizeDirectionEnv(CWRendezvousEnv)))

plus DummyVecEnv's auto-reset semantics (on done: terminal obs goes in
infos[i]["terminal_observation"], "TimeLimit.truncated" is set, only that row
resets, reset uses NO seed/options so each row's RNG persists across
episodes). Physics/reward formulas are copied verbatim from
libs/env.py::CWRendezvousEnv.step()/reset() — if you change one, change both
(scratchpad verify script: step both stacks with identical seeds/actions and
compare trajectories before trusting a change).

Info-dict shape decision: step_wait() still manufactures the list-of-n_envs
dicts that TrainingCallback / CurriculumCallback / SB3's collect_rollouts
consume, built from the batched arrays via .tolist() (one C call per field,
not n_envs numpy scalar extractions). Adapting the callbacks to consume raw
batched arrays would shave a little more, but the dict build is a small
fraction of the win and this keeps every existing consumer untouched.

Curriculum: curriculum_distance is ONE shared python float here (no per-row
copy to drift), so CurriculumCallback's
env_method("set_curriculum_distance", d) is a single attribute write.
"""

import time

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

from libs.constants import (
    ACTION_DIM,
    ENV_BONUS,
    ENV_BOUNDARY,
    ENV_CURRICULUM_ENABLED,
    ENV_CURRICULUM_INCREMENT,
    ENV_CURRICULUM_MAX_DISTANCE,
    ENV_CURRICULUM_START_DISTANCE,
    ENV_DT,
    ENV_FUEL_COEFF,
    ENV_MAX_DV,
    ENV_POS_TOLERANCE,
    ENV_SHAPING_COEFF,
    ENV_TIMEOUT,
    ENV_VEL_COEFF,
    MODE_2D,
    OBS_DIM,
    OMEGA,
    PHYS_STATE_DIM,
    RBAR_X_TO_Z_RATIO,
    SCENARIO,
)
from libs.env import _build_stm_2d
from libs.normalization import STATE_SCALE, NormalizedObsEnv
from libs.symmetry import CanonicalizeDirectionEnv


class BatchedCWVecEnv(VecEnv):
    """Drop-in stable_baselines3 VecEnv that steps all n_envs CW rendezvous
    environments as one batched numpy operation per vec-step."""

    def __init__(
        self,
        n_envs: int,
        scenario: str = SCENARIO,
        omega: float = OMEGA,
        dt: float = ENV_DT,
        max_dv: float = ENV_MAX_DV,
        boundary: float = ENV_BOUNDARY,
        timeout: float = ENV_TIMEOUT,
        pos_tolerance: float = ENV_POS_TOLERANCE,
        vel_coeff: float = ENV_VEL_COEFF,
        fuel_coeff: float = ENV_FUEL_COEFF,
        bonus: float = ENV_BONUS,
        curriculum_enabled: bool = ENV_CURRICULUM_ENABLED,
        curriculum_start_distance: float = ENV_CURRICULUM_START_DISTANCE,
        curriculum_max_distance: float = ENV_CURRICULUM_MAX_DISTANCE,
        curriculum_increment: float = ENV_CURRICULUM_INCREMENT,
        curriculum_boundary_mult: float = 2.0,
        rbar_x_to_z_ratio: float = RBAR_X_TO_Z_RATIO,
    ):
        if not MODE_2D:
            raise ValueError("BatchedCWVecEnv currently supports MODE_2D only")
        if scenario not in ("vbar", "rbar"):
            raise ValueError(f"Unknown scenario: {scenario!r}")

        self.scenario = scenario
        self.rbar_x_to_z_ratio = rbar_x_to_z_ratio
        self.phys_dim = PHYS_STATE_DIM
        self.half = PHYS_STATE_DIM // 2

        self.omega = omega
        self.dt = dt
        self.max_dv = max_dv
        self.base_boundary = boundary
        self.curriculum_boundary_mult = curriculum_boundary_mult
        self.timeout = timeout
        self.pos_tolerance = pos_tolerance
        self.vel_coeff = vel_coeff
        self.fuel_coeff = fuel_coeff
        self.bonus = bonus

        self.curriculum_enabled = curriculum_enabled
        self.curriculum_increment = curriculum_increment
        self.curriculum_max_distance = curriculum_max_distance
        self.curriculum_distance = (
            min(curriculum_start_distance, curriculum_max_distance)
            if curriculum_enabled
            else curriculum_max_distance
        )

        self.STM = _build_stm_2d(omega, dt)          # (4, 4) float64
        self.STM_T = np.ascontiguousarray(self.STM.T)

        # Spaces mirror the wrapped single-env stack: NormalizedObsEnv maps
        # obs into [0, 1]; actions stay in raw physical units.
        observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float64
        )
        action_space = spaces.Box(
            low=-max_dv, high=max_dv, shape=(ACTION_DIM,), dtype=np.float64
        )
        self.render_mode = None  # read by VecEnv.__init__ via get_attr
        super().__init__(n_envs, observation_space, action_space)

        n = n_envs
        # --- batched physical + per-episode state (row i == env i) ---
        self.states = np.zeros((n, self.phys_dim), dtype=np.float64)
        self.dv_used = np.zeros(n, dtype=np.float64)
        self.elapsed_time = np.zeros(n, dtype=np.float64)
        self.excursion_limit = np.full(n, boundary, dtype=np.float64)
        self._forced_sign = np.full(n, np.nan)  # nan = sample randomly

        # CanonicalizeDirectionEnv state: which rows are currently mirrored.
        self._mirror = np.zeros(n, dtype=bool)
        self._norm_obs = np.zeros((n, OBS_DIM), dtype=np.float64)

        # Monitor state (episode return/length accounting).
        self._ep_rew = np.zeros(n, dtype=np.float64)
        self._ep_len = np.zeros(n, dtype=np.int64)
        self._t_start = time.time()

        # One gymnasium-style Generator per row so sign draws follow the
        # exact same stream a seeded per-env stack would (seeded in reset()
        # from VecEnv._seeds, persists across auto-resets like DummyVecEnv).
        self._rngs = [seeding.np_random(None)[0] for _ in range(n)]

        self._actions = None

    # ── Curriculum hook (same name/signature as CWRendezvousEnv) ──────────

    def set_curriculum_distance(self, distance: float):
        self.curriculum_distance = float(
            np.clip(distance, 0.0, self.curriculum_max_distance)
        )

    # ── Episode reset (vectorized CWRendezvousEnv.reset) ──────────────────

    def _reset_rows(self, idx: np.ndarray) -> None:
        cd = self.curriculum_distance
        for i in idx:
            forced = self._forced_sign[i]
            if not np.isnan(forced):
                sign = float(forced)
            else:
                sign = 1.0 if self._rngs[i].random() < 0.5 else -1.0
            if self.scenario == "vbar":
                direction = np.array([sign, 0.0], dtype=np.float64)
            else:  # "rbar": (+x,-z) or (-x,+z) coupled displacement
                raw = np.array(
                    [sign * self.rbar_x_to_z_ratio, -sign], dtype=np.float64
                )
                direction = raw / np.linalg.norm(raw)
            pos = direction * cd
            self.states[i, : self.half] = pos
            self.states[i, self.half:] = 0.0

        self.excursion_limit[idx] = min(
            self.base_boundary, cd * self.curriculum_boundary_mult
        )
        self.elapsed_time[idx] = 0.0
        self.dv_used[idx] = 0.0
        self._ep_rew[idx] = 0.0
        self._ep_len[idx] = 0

    # ── Observation pipeline (CanonicalizeDirectionEnv + NormalizedObsEnv) ─

    def _recompute_obs(self) -> None:
        raw = np.concatenate([self.states, self.dv_used[:, None]], axis=1)
        self._mirror = raw[:, 0] < 0.0
        raw[self._mirror, :-1] *= -1.0  # mirror all but the dv_used scalar
        np.clip(raw, -STATE_SCALE, STATE_SCALE, out=raw)
        self._norm_obs = (raw + STATE_SCALE) / (2.0 * STATE_SCALE)

    # ── VecEnv API ─────────────────────────────────────────────────────────

    def reset(self):
        for i in range(self.num_envs):
            if self._seeds[i] is not None:
                self._rngs[i] = seeding.np_random(self._seeds[i])[0]
            sign = (self._options[i] or {}).get("sign")
            self._forced_sign[i] = np.nan if sign is None else float(sign)
        self._reset_rows(np.arange(self.num_envs))
        self._recompute_obs()
        self.reset_infos = [
            {"curriculum_distance": self.curriculum_distance}
            for _ in range(self.num_envs)
        ]
        self._reset_seeds()
        self._reset_options()
        return self._norm_obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        n = self.num_envs
        actions = np.asarray(self._actions, dtype=np.float64).reshape(n, ACTION_DIM)
        # Mirror-canonicalization: rows flagged at the previous obs act in
        # the x>=0 canonical frame; map their action back before applying.
        real = np.where(self._mirror[:, None], -actions, actions)
        real = np.clip(real, -self.max_dv, self.max_dv)

        # --- physics + bookkeeping (verbatim from CWRendezvousEnv.step) ---
        dv_norm = np.linalg.norm(real, axis=1)
        self.dv_used += dv_norm

        prev_pos_error = np.linalg.norm(self.states[:, : self.half], axis=1)
        self.states[:, self.half:] += real
        self.states = self.states @ self.STM_T   # row i -> STM @ state_i
        self.elapsed_time += self.dt

        pos_error = np.linalg.norm(self.states[:, : self.half], axis=1)
        vel_error = np.linalg.norm(self.states[:, self.half:], axis=1)

        docked = pos_error < self.pos_tolerance
        out_of_bounds = pos_error > self.excursion_limit
        timeout = self.elapsed_time > self.timeout

        delta = prev_pos_error - pos_error
        terminated = docked | out_of_bounds
        truncated = timeout

        # --- reward (verbatim from CWRendezvousEnv.step): dense
        # distance-shaping + docking bonus + a plain, smooth per-step fuel
        # cost. No ceiling, no truncation tied to it.
        reward_pos = ENV_SHAPING_COEFF * delta / self.curriculum_distance
        reward_fuel = -self.fuel_coeff * dv_norm
        reward_terminal = np.where(docked, self.bonus - self.vel_coeff * vel_error, 0.0)
        reward = reward_pos + reward_fuel + reward_terminal
        dones = terminated | truncated

        # Monitor accounting (float64 running sum == Monitor's sum of floats).
        self._ep_rew += reward
        self._ep_len += 1

        # Obs after the step; for done rows this is the terminal obs.
        self._recompute_obs()

        # --- per-env info dicts (bulk .tolist() then plain-python indexing) ---
        states_copy = self.states.copy()
        real_copy = real.copy()
        dist_l = pos_error.tolist()
        dock_l = docked.tolist()
        rpos_l = reward_pos.tolist()
        rfuel_l = reward_fuel.tolist()
        rterm_l = reward_terminal.tolist()
        verr_l = vel_error.tolist()
        dvn_l = dv_norm.tolist()
        dvu_l = self.dv_used.tolist()
        exlim_l = self.excursion_limit.tolist()
        term_l = terminated.tolist()
        trunc_l = truncated.tolist()
        cd = self.curriculum_distance

        infos = [
            {
                "state": states_copy[i],
                "distance": dist_l[i],
                "docked": dock_l[i],
                "reward_pos": rpos_l[i],
                "reward_fuel": rfuel_l[i],
                "reward_terminal": rterm_l[i],
                "vel_error": verr_l[i],
                "delta_v": dvn_l[i],
                "applied_action": real_copy[i],
                "dv_used": dvu_l[i],
                "curriculum_distance": cd,
                "excursion_limit": exlim_l[i],
                "TimeLimit.truncated": trunc_l[i] and not term_l[i],
            }
            for i in range(n)
        ]

        done_idx = np.flatnonzero(dones)
        if done_idx.size:
            t_now = time.time()
            for i in done_idx:
                infos[i]["terminal_observation"] = self._norm_obs[i].copy()
                infos[i]["episode"] = {
                    "r": round(float(self._ep_rew[i]), 6),
                    "l": int(self._ep_len[i]),
                    "t": round(t_now - self._t_start, 6),
                }
                self.reset_infos[i] = {"curriculum_distance": cd}
            # DummyVecEnv auto-resets with no seed/options -> forced sign
            # clears and each row's RNG stream continues.
            self._forced_sign[done_idx] = np.nan
            self._reset_rows(done_idx)
            self._recompute_obs()

        return (
            self._norm_obs.copy(),
            reward.astype(np.float32),
            dones.copy(),
            infos,
        )

    def close(self) -> None:
        pass

    # ── Introspection API (shared-state: one value fans out to all rows) ───

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None):
        indices = self._get_indices(indices)
        if not hasattr(self, attr_name):
            raise AttributeError(f"{type(self).__name__} has no attribute {attr_name!r}")
        value = getattr(self, attr_name)
        return [value for _ in indices]

    def set_attr(self, attr_name: str, value, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args,
                   indices: VecEnvIndices = None, **method_kwargs):
        indices = list(self._get_indices(indices))
        result = getattr(self, method_name)(*method_args, **method_kwargs)
        return [result for _ in indices]

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None):
        indices = list(self._get_indices(indices))
        # We emulate these wrappers' behavior internally.
        emulated = wrapper_class in (Monitor, NormalizedObsEnv, CanonicalizeDirectionEnv)
        return [emulated for _ in indices]
