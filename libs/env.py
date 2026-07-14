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
        burn_deadzone_frac: float = ENV_BURN_DEADZONE_FRAC,
        boundary: float = ENV_BOUNDARY,
        timeout: float = ENV_TIMEOUT,
        pos_tolerance: float = ENV_POS_TOLERANCE,
        vel_coeff: float = ENV_VEL_COEFF,
        fuel_coeff: float = ENV_FUEL_COEFF,
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
        self.action_dim = ACTION_DIM     # 2 (2D) or 3 (3D)
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
        self.burn_deadzone_frac = burn_deadzone_frac
        self.burn_deadzone = 0.0  # set for real in reset(), as burn_deadzone_frac * max_dv
        self.base_boundary = boundary
        self.excursion_limit = boundary
        self.curriculum_boundary_mult = curriculum_boundary_mult
        self.timeout = timeout
        self.pos_tolerance = pos_tolerance
        self.vel_coeff = vel_coeff
        self.fuel_coeff = fuel_coeff
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
        self.elapsed_time = 0.0
        self.dv_used      = 0.0

        half = self.phys_dim // 2

        if self.scenario == "vbar":
            self.dv_ref = 0.5 * dv_vbar_two_impulse_rr(np.linalg.norm(self.state[:half]), self.omega)
            # We divide by half since the output of the function is the TOTAL delta v required for also stopping
        else:
            self.dv_ref = dv_rbar_strategy_rv(np.linalg.norm(self.state[:half]), self.omega)
            # TODO: divide by the amount required actually to reach... could use half as approximation
            
        self.max_dv = self.dv_ref * self.max_dv_coeff

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

    def step(self, action: np.ndarray):
        # Action is a NORMALIZED fraction u in [-1, 1] per axis; scale it to the
        # physical impulse for this episode's distance (max_dv_ep set in reset).
        u = np.clip(action, -1.0, 1.0)
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
        self.dv_used += burn

        half = self.phys_dim // 2
        prev_pos_error = np.linalg.norm(self.state[:half])

        # One impulsive Δv is applied ONCE at the start of the agent step,
        # then the state coasts through n_substeps of the fine physics dt.
        # Docking (segment test) and out-of-bounds are checked at EVERY
        # substep so a fast pass-through the target can't tunnel between the
        # coarse agent samples, and we stop the instant a terminal event
        # happens instead of only looking at the agent-step endpoint.
        self.state[half:] += action

        docked = out_of_bounds = timeout = False
        for _ in range(self.n_substeps):
            prev_pos = self.state[:half].copy()
            self.state = self.STM_np @ self.state
            self.elapsed_time += self.dt_phys
            new_pos = self.state[:half]

            closest, closest_dist = self._segment_closest_to_origin(prev_pos, new_pos)
            if closest_dist < self.pos_tolerance:
                docked = True
                # Snap the recorded position to the closest-approach point so
                # pos_error reflects the true miss distance; keep the coasted
                # velocity for the terminal (vel_error) penalty.
                self.state[:half] = closest
                break
            if np.linalg.norm(new_pos) > self.excursion_limit:
                out_of_bounds = True
                break
            if self.elapsed_time > self.timeout:
                timeout = True
                break

        pos_error = np.linalg.norm(self.state[:half])
        vel_error = np.linalg.norm(self.state[half:])

        delta      = prev_pos_error - pos_error
        terminated = bool(docked or out_of_bounds)
        truncated  = bool(timeout)

        # --- Reward: dense distance-shaping + docking bonus + a fuel bonus
        # paid only on a successful dock, inverse-sqrt in cumulative
        # dv_used so smaller dv_used is disproportionately rewarded even
        # once it's already small (see class docstring). eps floors it so
        # an extremely low-fuel dock can't spike to +inf / divide by zero.
        reward_pos = ENV_SHAPING_COEFF * delta / self.curriculum_distance

        if docked:
            # Reward is coeff/ratio, ratio = dv_used/dv_ref (the analytic
            # two-impulse reference computed in reset()) floored at 1 so
            # matching-or-beating the reference all saturates at the same
            # peak (fuel_coeff) instead of blowing up as dv_used -> 0.
            #
            # A plain inverse (not inverse-CUBE, as an earlier version used)
            # is a deliberate choice: 1/ratio has constant elasticity
            # (d(reward)/reward == -d(ratio)/ratio everywhere), so a given
            # PROPORTIONAL fuel improvement is worth the same reward change
            # at ratio=1.2 as at ratio=20 — unlike 1/ratio**3, which is steep
            # near ratio=1 but has essentially zero gradient by ratio~10
            # (100/10**3=0.1, 100/11**3=0.075 — a 10% fuel cut is worth
            # 0.025 points, invisible next to ordinary terminal-velocity
            # noise). That flat-tail problem was real: a live run sitting at
            # ratio~13-17x reference showed total dv NOT improving over
            # 26k+ episodes because the fuel term had nothing left to say.
            # With 1/ratio: ratio=1->100, 1.5->66.7, 2->50, 3->33.3, 5->20,
            # 10->10, 15->6.7, 20->5 — strong near the optimum (as before)
            # AND still a meaningful, non-vanishing gradient at 20x+.
            ratio = max(self.dv_used / self.dv_ref, 1.0)
            reward_fuel = self.fuel_coeff / ratio
        else:
            reward_fuel = 0.0


        reward_terminal = self.bonus if docked else 0.0
        reward = reward_pos + reward_fuel + reward_terminal

        observation = self._build_observation()
        info = {
            "state":           self.state.copy(),
            "distance":        pos_error,
            "docked":          docked,
            "reward_pos":      reward_pos,
            "reward_fuel":     reward_fuel,
            "reward_terminal": reward_terminal,
            "vel_error":       vel_error,
            "delta_v":         np.linalg.norm(action),
            "applied_action":  action.copy(),
            "dv_used":         self.dv_used,
            "dv_ref":          self.dv_ref,
            "curriculum_distance": self.curriculum_distance,
            "excursion_limit": self.excursion_limit,
        }

        return observation, reward, terminated, truncated, info
