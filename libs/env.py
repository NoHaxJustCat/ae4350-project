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
    ENV_SHAPING_COEFF,
    OMEGA,
    ENV_CURRICULUM_ENABLED,
    ENV_CURRICULUM_START_DISTANCE,
    ENV_CURRICULUM_MAX_DISTANCE,
    ENV_CURRICULUM_INCREMENT,
    ENV_CURRICULUM_BOUNDARY_MULT,
    SCENARIO,
    RBAR_X_TO_Z_RATIO,
    MODE_2D,
    ACTION_DIM,
    ACTION_IMPULSE_DIM,
    ENV_COAST_MIN_UNITS,
    ENV_COAST_MAX_UNITS,
    PHYS_STATE_DIM,
    OBS_DIM,
)
from scipy.linalg import expm

from libs.reference import dv_rbar_strategy_rv, dv_vbar_two_impulse_rr


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

    Observation appends cumulative Δv used so far:
        2-D obs : [x, z, ẋ, ż, dv_used]           (5-dim)
        3-D obs : [x, y, z, ẋ, ẏ, ż, dv_used]     (7-dim)

    Action:
        2-D : [dvx, dvz]   (2-dim)
        3-D : [dvx, dvy, dvz]   (3-dim)

    Reward is intentionally barebones: dense distance-shaping every step,
    a docking bonus, and a fuel cost — nothing else. No milestone
    bookkeeping, no dv ceiling/truncation tied to fuel, no malus — those
    were the source of repeated reward-exploit debugging (fast-burn-to-
    ceiling loops, reward discontinuities).

    The fuel term is a terminal bonus on cumulative dv_used, paid only on a
    successful dock: ENV_FUEL_COEFF * (dv_used + eps)**-1 (currently
    25 * (dv_used + 0.01)**-1). Earlier versions tried a per-step linear
    cost (-coeff * ||action||), then a per-step log1p-telescoped cost — both
    replaced. The linear cost gives equal reward for equal *absolute* dv
    reductions, so its gradient vanishes once dv_used is already small
    (cutting 4->2 m/s earns far more than the equally impressive 2x cut from
    0.05->0.025 m/s), and a real run measurably drifted back up once it
    reached that low-signal regime. The inverse terminal bonus keeps that
    "smaller is disproportionately better" pressure alive at low dv_used
    without ever letting fuel cost make *failing* to dock look better than a
    wasteful dock (it's strictly additive on top of the dock bonus, never
    subtracted elsewhere). The small eps (0.01, ~the optimal dv scale) keeps
    it finite as dv_used -> 0 while leaving the curve steep near the optimum;
    see the ENV_FUEL_COEFF comment in constants.py.

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
        if scenario not in ("vbar", "rbar"):
            raise ValueError(f"Unknown scenario: {scenario!r}")
        if not MODE_2D and scenario == "rbar":
            raise ValueError("scenario='rbar' is only defined for MODE_2D")

        self.mode_2d = MODE_2D
        self.phys_dim = PHYS_STATE_DIM   # 4 (2D) or 6 (3D)
        self.obs_dim  = OBS_DIM          # 5 (2D) or 7 (3D)
        # Full action = impulse components + 1 coast-duration scalar.
        self.action_dim = ACTION_DIM             # 3 (2D) or 4 (3D)
        self.impulse_dim = ACTION_IMPULSE_DIM    # 2 (2D) or 3 (3D)
        self.scenario = scenario
        self.rbar_x_to_z_ratio = rbar_x_to_z_ratio

        self.omega = omega
        # dt is the AGENT step (one impulse + one observation + one stored
        # transition); dt_phys is the fine physics/collision substep it is
        # decomposed into. Must divide evenly — the substep count is exact.
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
        # None = unconstrained (default — matches all prior behavior). Once
        # set (training.py's DvBudgetCurriculumCallback, via
        # set_dv_budget_coeff()), dv_budget = dv_ref * dv_budget_coeff caps
        # TOTAL cumulative dv_used for the whole episode, not just one burn
        # — see the ENV_DV_BUDGET_COEFF_START comment in constants.py.
        self.dv_budget_coeff = dv_budget_coeff
        self.dv_budget = None  # set for real in reset(), as dv_budget_coeff * dv_ref
        self.burn_deadzone_frac = burn_deadzone_frac
        self.burn_deadzone = 0.0  # set for real in reset(), as burn_deadzone_frac * max_dv
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
        # Built for the fine PHYSICS substep, not the agent step. Propagating
        # n_substeps of dt_phys is exactly propagating one dt_agent (the CW
        # STM is a matrix exponential: expm(A*dt_phys)**n == expm(A*n*dt_phys)),
        # so this changes nothing about the physics — it only lets us sample
        # the trajectory finely for the docking/collision test.
        if self.mode_2d:
            self.STM_np = _build_stm_2d(omega, dt_phys)
        else:
            self.STM_np = _build_stm_full(omega, dt_phys)

        # Precompute STM_phys^k for k = 1..(max coast length in substeps). A
        # coast decision can span a whole orbit (~1440 fine substeps); stepping
        # that as a Python loop dominated runtime. With these stacked powers the
        # entire coast is ONE batched matmul (seq = powers[:total] @ s0) plus
        # vectorized docking/OOB/timeout tests — byte-for-byte the same substep
        # trajectory as the loop, just computed at once. Built once per env.
        max_total_substeps = ENV_COAST_MAX_UNITS * self.n_substeps
        powers = np.empty((max_total_substeps, self.phys_dim, self.phys_dim))
        acc = np.eye(self.phys_dim)
        for _k in range(max_total_substeps):
            acc = self.STM_np @ acc
            powers[_k] = acc
        self._coast_powers = powers

        # --- Gym spaces ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float64
        )

        self.state = None
        self._forced_sign = None

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

    def set_dv_budget_coeff(self, coeff: Optional[float]):
        """External control hook — called by training.py's
        DvBudgetCurriculumCallback (via VecEnv.env_method), mirroring
        set_curriculum_distance(). None disables the budget (unconstrained,
        the default); a float takes effect from the NEXT reset() onward
        (dv_budget is recomputed from dv_budget_coeff * dv_ref there), not
        retroactively on whatever episode is already in flight."""
        self.dv_budget_coeff = None if coeff is None else max(float(coeff), 1e-3)

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        flag = 1.0 if self.braking_phase else 0.0
        return np.concatenate([self.state, [self.dv_used, flag]]).astype(np.float64)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._forced_sign = (options or {}).get("sign")
        self.state = self._sample_initial_state()
        self.excursion_limit = min(
            self.base_boundary,
            self.curriculum_distance * self.curriculum_boundary_mult,
        )
        self.elapsed_time = 0.0
        self.dv_used      = 0.0
        # Reaching the target opens a one-step "braking phase" (see step() /
        # _brake_step): the episode does NOT end at arrival — the agent gets
        # one more action to fire a terminal impulse nulling its velocity, and
        # only then terminates. False during the whole transfer, True on the
        # single observation the agent acts on to brake.
        self.braking_phase = False

        half = self.phys_dim // 2

        if self.scenario == "vbar":
            self.dv_ref = 0.5 * dv_vbar_two_impulse_rr(np.linalg.norm(self.state[:half]), self.omega)
            # We divide by half since the output of the function is the TOTAL delta v required for also stopping
        else:
            self.dv_ref = dv_rbar_strategy_rv(np.linalg.norm(self.state[:half]), self.omega)
            # TODO: divide by the amount required actually to reach... could use half as approximation
            
        self.max_dv = self.dv_ref * self.max_dv_coeff
        self.dv_budget = (
            None if self.dv_budget_coeff is None else self.dv_ref * self.dv_budget_coeff
        )

        self.burn_deadzone = self.burn_deadzone_frac * self.max_dv

        observation = self._build_observation()
        # "state" mirrors step()'s info so callers (e.g. training.py's
        # trajectory-plot accumulator) can seed a new episode's start point
        # from vec_env.reset_infos, instead of the plot only ever showing
        # each episode's first POST-action position.
        info = {"curriculum_distance": self.curriculum_distance, "state": self.state.copy()}
        return observation, info

    @staticmethod
    def _segment_closest_to_origin(p0: np.ndarray, p1: np.ndarray):
        """Closest point on the segment [p0, p1] to the origin, and its
        distance. Used for the docking test so a fast fly-through of the
        pos_tolerance circle registers a dock even if neither endpoint sample
        happens to land inside it (the tunnelling the larger agent step would
        otherwise cause). Exact for a straight chord; over one fine dt_phys the
        real curved CW arc is essentially straight, so this is accurate."""
        d = p1 - p0
        denom = float(d @ d)
        if denom <= 0.0:
            closest = p0
        else:
            t = float(np.clip(-(p0 @ d) / denom, 0.0, 1.0))
            closest = p0 + t * d
        return closest, float(np.linalg.norm(closest))

    def _coast_units_from_cmd(self, coast_cmd: float) -> int:
        """Map the coast-duration action scalar in [-1, 1] to an integer
        number of agent-dt coast units in [ENV_COAST_MIN_UNITS,
        ENV_COAST_MAX_UNITS]. Linear so the physically meaningful coasts (a
        half-orbit ~29, a full orbit ~58) sit at well-resolved interior points
        rather than saturated at an endpoint (see ENV_COAST_* in constants.py)."""
        frac01 = (float(np.clip(coast_cmd, -1.0, 1.0)) + 1.0) * 0.5
        span = ENV_COAST_MAX_UNITS - ENV_COAST_MIN_UNITS
        return ENV_COAST_MIN_UNITS + int(round(frac01 * span))

    def step(self, action: np.ndarray):
        # Action = [impulse..., coast_cmd]. The impulse components are a
        # NORMALIZED fraction u in [-1, 1] per axis, scaled to the physical
        # impulse for this episode's distance (max_dv set in reset()); the
        # trailing scalar chooses how many agent-dt units to coast AFTER the
        # burn (see _coast_units_from_cmd).
        action = np.asarray(action, dtype=np.float64)
        # If the target was reached on the previous step, this action is the
        # terminal braking impulse — a different, no-coast code path.
        if self.braking_phase:
            return self._brake_step(action)

        coast_units = self._coast_units_from_cmd(action[self.impulse_dim])
        u = np.clip(action[: self.impulse_dim], -1.0, 1.0)
        action = u * self.max_dv

        # Burn deadzone / minimum-impulse-bit: a commanded burn below the
        # threshold is treated as EXACTLY zero — no Δv charged and no velocity
        # applied — so the agent can coast for free instead of leaking a
        # little fuel every step it can't output an exact zero (see
        # ENV_BURN_DEADZONE_FRAC in constants.py). Applied to the norm so it's the
        # total impulse magnitude that must clear the threshold, and used
        # everywhere below (dv_used, the state update, and the info's
        # delta_v/applied_action) so what's charged == what's applied.
        burn = float(np.linalg.norm(action))
        if burn < self.burn_deadzone:
            action = np.zeros_like(action)
            burn = 0.0
        elif self.dv_budget is not None:
            # Total-episode Δv BUDGET (distinct from max_dv above, which
            # only caps THIS one burn): clip — don't zero — any burn that
            # would push cumulative dv_used past dv_budget down to whatever
            # remains, preserving direction. The tank runs genuinely dry
            # instead of the agent being free to keep re-burning at the
            # per-burst cap indefinitely (see the ENV_DV_BUDGET_COEFF_START
            # comment in constants.py for why a per-burst cap alone doesn't
            # limit burn COUNT).
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

        # One impulsive Δv is applied ONCE at the start of the agent step,
        # then the state coasts BALLISTICALLY for the agent-chosen number of
        # agent-dt units (coast_units), each decomposed into n_substeps of the
        # fine physics dt. Docking (segment test) and out-of-bounds are checked
        # at EVERY substep so a fast pass-through the target can't tunnel
        # between samples, and we stop the instant a terminal event happens.
        # This is the whole point of the coast-duration action: the agent
        # spends ONE decision on a possibly orbit-long coast instead of having
        # to emit ~30 consecutive near-zero actions to reproduce it.
        self.state[half:] += action
        s0 = self.state.copy()  # state at the burn point (start of the coast)
        total_substeps = coast_units * self.n_substeps

        # Whole coast in one batched matmul: seq[i] = STM_phys^(i+1) @ s0 for
        # i = 0..total-1. EXACTLY the substep sequence the old per-substep loop
        # produced (verified byte-identical), just vectorized.
        seq = self._coast_powers[:total_substeps] @ s0     # (total, phys_dim)
        pos_seq = seq[:, :half]                             # pos after each substep
        prev_seq = np.vstack([s0[:half], pos_seq[:-1]])     # pos before each substep

        # Vectorized segment-closest-to-origin for every (prev, new) chord, so a
        # fast fly-through the tolerance circle still registers between samples.
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

        # First substep index (0-based) at which each terminal condition fires.
        BIG = total_substeps + 1
        dock_hits = np.flatnonzero(closest_dist < self.pos_tolerance)
        oob_hits = np.flatnonzero(new_dist > self.excursion_limit)
        i_dock = int(dock_hits[0]) if dock_hits.size else BIG
        i_oob = int(oob_hits[0]) if oob_hits.size else BIG
        # timeout: elapsed + (i+1)*dt_phys > timeout  ->  first i is floor(thresh)
        thresh = (self.timeout - self.elapsed_time) / self.dt_phys
        i_timeout = int(np.floor(thresh))
        if not (0 <= i_timeout < total_substeps):
            i_timeout = BIG

        # Earliest event wins; within one substep the old loop's priority was
        # dock > out-of-bounds > timeout, reproduced by the tie ordering here.
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
            # Snap position to the closest-approach point (true miss distance);
            # keep the coasted velocity for the terminal vel_error / stop bonus.
            self.state[:half] = closest[i_end]

        # Coarse (agent-dt) samples of the coast leg for trajectory plotting, so
        # a decision spanning a whole orbit still renders as a smooth arc rather
        # than one straight chord. Interior agent-dt boundaries reached before
        # the terminal substep, plus the true terminal endpoint (matching the
        # old loop's substate bookkeeping exactly).
        substates = [seq[i].copy() for i in range(self.n_substeps - 1, i_end, self.n_substeps)]
        substates.append(self.state.copy())

        pos_error = np.linalg.norm(self.state[:half])
        vel_error = np.linalg.norm(self.state[half:])

        delta = prev_pos_error - pos_error

        # Dense distance shaping only. Terminal bonuses (dock / fuel / stopping)
        # are NOT paid here: reaching the target opens the braking phase, and
        # _brake_step pays them on the post-brake terminal state after the agent
        # fires its velocity-nulling impulse next step. This mirrors the analytic
        # two-impulse maneuver whose SECOND burn brakes exactly at the target.
        # OOB / timeout still end the episode immediately (no brake, no dock).
        reward_pos = ENV_SHAPING_COEFF * delta / self.curriculum_distance
        reward_fuel = reward_stop = reward_terminal = 0.0

        if docked:
            self.braking_phase = True
            terminated = False
            truncated = False
        else:
            terminated = bool(out_of_bounds)
            truncated = bool(timeout)

        reward = reward_pos

        observation = self._build_observation()
        info = {
            "state":           self.state.copy(),
            "substates":       substates,
            "coast_units":     coast_units,
            "distance":        pos_error,
            # Dock is completed only after the brake; report False here so the
            # episode is credited as docked on the terminal _brake_step.
            "docked":          False,
            "reward_pos":      reward_pos,
            "reward_fuel":     reward_fuel,
            "reward_stop":     reward_stop,
            "reward_terminal": reward_terminal,
            "vel_error":       vel_error,
            "delta_v":         np.linalg.norm(action),
            "applied_action":  action.copy(),
            "dv_used":         self.dv_used,
            "dv_ref":          self.dv_ref,
            "dv_budget_coeff": self.dv_budget_coeff,
            "curriculum_distance": self.curriculum_distance,
            "excursion_limit": self.excursion_limit,
            "braking_phase":   self.braking_phase,
        }

        return observation, reward, terminated, truncated, info

    def _brake_step(self, action: np.ndarray):
        """Terminal braking phase (entered the step AFTER the target is
        reached; see step()). This action is the agent's final impulse to null
        its arrival velocity — a true rendezvous rather than a fly-through.
        There is NO coast: the impulse is applied at the target and the episode
        terminates immediately, scored on the post-brake speed.

        The minimum-impulse deadzone is deliberately NOT applied here: the
        optimal brake can be well below it (a fly-through arrives at ~0.21·dv_ref
        while the coast deadzone is 0.3·dv_ref), so zeroing sub-deadzone burns
        would make a full stop physically unrepresentable. The total-episode Δv
        budget still applies (clip, don't zero), same as the transfer step."""
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

        # Impulsive brake changes velocity only; position stays at the snapped
        # closest-approach point recorded at arrival.
        self.state[half:] += impulse
        pos_error = float(np.linalg.norm(self.state[:half]))
        vel_error = float(np.linalg.norm(self.state[half:]))

        # Terminal reward, now that the dock is complete: flat dock bonus + fuel
        # bonus on TOTAL dv (transfer + brake, floored at ratio 1 — see
        # ENV_FUEL_COEFF) + stopping bonus on the post-brake speed (smooth toward
        # vel_error -> 0, sized to outweigh the brake's Δv — see ENV_STOP_COEFF).
        ratio = max(self.dv_used / self.dv_ref, 1.0)
        reward_fuel = self.fuel_coeff / ratio
        v_scale = self.stop_vel_scale_frac * self.dv_ref
        reward_stop = self.stop_coeff / (1.0 + vel_error / v_scale)
        reward_terminal = self.bonus
        reward_pos = 0.0
        reward = reward_pos + reward_fuel + reward_stop + reward_terminal

        self.braking_phase = False
        observation = self._build_observation()
        info = {
            "state":           self.state.copy(),
            "substates":       [self.state.copy()],
            "coast_units":     0,
            "distance":        pos_error,
            "docked":          True,
            "reward_pos":      reward_pos,
            "reward_fuel":     reward_fuel,
            "reward_stop":     reward_stop,
            "reward_terminal": reward_terminal,
            "vel_error":       vel_error,
            "delta_v":         burn,
            "applied_action":  impulse.copy(),
            "dv_used":         self.dv_used,
            "dv_ref":          self.dv_ref,
            "dv_budget_coeff": self.dv_budget_coeff,
            "curriculum_distance": self.curriculum_distance,
            "excursion_limit": self.excursion_limit,
            "braking_phase":   False,
        }
        return observation, reward, True, False, info
