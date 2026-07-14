import os
import numpy as np

MODE_2D: bool = True

# Derived convenience constants (do not edit these directly)
# Physical state dim: 4 (2D) or 6 (3D)
PHYS_STATE_DIM: int = 4 if MODE_2D else 6
# Observation dim = physical state + dv_used scalar
OBS_DIM:    int = PHYS_STATE_DIM + 1       # 5 (2D) or 7 (3D)
ACTION_DIM: int = 2 if MODE_2D else 3

# --- Orbit / physics constants ---
EARTH_MU = 3.986 * 10 ** 14  # m^3 / s^2
ORBIT_RADIUS = (6378 + 600) * 10 ** 3  # m
OMEGA = np.sqrt(EARTH_MU / ORBIT_RADIUS ** 3)
ORBIT_PERIOD = 2 * np.pi / OMEGA

# --- Curriculum learning defaults ---
# Defined before ENV_BOUNDARY below so the excursion-limit safety ceiling can
# be derived from the curriculum range instead of hand-tuned separately —
# raising ENV_CURRICULUM_MAX_DISTANCE used to silently desync from a fixed
# ENV_BOUNDARY and break training past ~200 m (see libs/env.py reset()).
ENV_CURRICULUM_ENABLED = True
ENV_CURRICULUM_START_DISTANCE = 30.0   # starting distance [m]
ENV_CURRICULUM_MAX_DISTANCE = 1000.0   # final distance [m] (matches ENV_INITIAL_STATE_VBAR norm)
ENV_CURRICULUM_INCREMENT = 10.0        # distance added per successful dock [m]
# Excursion-limit headroom: CWRendezvousEnv.reset() sets
# excursion_limit = curriculum_distance * ENV_CURRICULUM_BOUNDARY_MULT, so an
# episode can drift up to this many multiples of its own starting distance
# before being flagged out-of-bounds (see libs/env.py's curriculum_boundary_mult).
ENV_CURRICULUM_BOUNDARY_MULT = 2.0

# --- Environment defaults ---
# Two decoupled timesteps (see libs/env.py):
#   ENV_DT_PHYS  : the fine PHYSICS/collision substep. The CW state-transition
#                  matrix is the exact closed-form solution of the linear
#                  dynamics, so this introduces NO integration error at any
#                  size — it exists only to sample the trajectory finely
#                  enough that (a) the straight-chord docking test is accurate
#                  and (b) a fast pass-through the 1 m circle can't tunnel
#                  between samples. Kept at the historical 5 s so the physics
#                  is byte-identical to every prior run.
#   ENV_DT_AGENT : the interval at which the RL agent actually acts (applies
#                  one impulsive Δv), observes, and writes a transition to the
#                  replay buffer. Larger than dt_phys so an episode spans far
#                  fewer agent steps, which lets a MUCH lower gamma still
#                  "see" the terminal reward (see GAMMA below). Must be an
#                  integer multiple of ENV_DT_PHYS.
ENV_DT_PHYS = 5.0
ENV_DT_AGENT = 100.0
# Back-compat alias: ENV_DT means the AGENT step everywhere downstream
# (MAX_STEPS, evaluate.py, the env's default dt).
ENV_DT = ENV_DT_AGENT
# Excursion / out-of-bounds safety ceiling. Derived from the curriculum range
# (not a hand-picked absolute) so it always covers the largest possible
# excursion_limit CWRendezvousEnv.reset() can compute
# (curriculum_distance * ENV_CURRICULUM_BOUNDARY_MULT, maxed out at
# ENV_CURRICULUM_MAX_DISTANCE). Previously a fixed 200 m tuned for the old
# 100 m curriculum max — once the curriculum max was raised to 1000 m this
# silently clamped excursion_limit at 200 regardless of curriculum_distance,
# so every episode past ~200 m curriculum distance spawned already outside
# its own boundary and terminated out-of-bounds on the very first step.
ENV_BOUNDARY = ENV_CURRICULUM_MAX_DISTANCE * ENV_CURRICULUM_BOUNDARY_MULT
ENV_TIMEOUT = 2 * ORBIT_PERIOD
ENV_POS_TOLERANCE = 5.0
ENV_VEL_COEFF = 10.0
ENV_SHAPING_COEFF = 10.0
ENV_BONUS = 50.0
# Fuel bonus: paid only on a successful dock, inverse (constant-elasticity)
# decay in dv_used/dv_ref (reward_fuel = coeff / max(dv_used/dv_ref, 1) —
# see libs/env.py::step). dv_ref is the analytic two-impulse reference for
# this episode's scenario/distance (libs/reference.py), so the bonus is
# graded against "how close to the optimal transfer" rather than an
# absolute m/s scale. Strictly additive on top of the dock bonus, so it can
# never make failing to dock look better than a wasteful dock (zero unless
# docked=True).
#
# coeff=100 sets the peak (dv_used <= dv_ref): 100/1=100. ratio=1.5->66.7,
# 2->50, 3->33.3, 5->20, 10->10, 15->6.7, 20->5 — a plain 1/ratio (not
# 1/ratio**3, an earlier version) so the reward has constant elasticity: a
# given PROPORTIONAL fuel improvement is worth the same relative reward
# change at ratio=20 as at ratio=1.2, instead of the gradient vanishing once
# dv_used is already many multiples of optimal. A live run sitting at
# ratio~13-17x showed total dv flat for 26k+ episodes under the old cubic
# (100/15**3=0.03, 100/16**3=0.024 — a real fuel improvement was worth
# 0.006 points, invisible next to ordinary terminal-velocity noise); 1/ratio
# is worth 6.7 vs 6.25 at the same two points, an actually learnable signal.
ENV_FUEL_COEFF = 500.0

# Physical per-burn actuator cap: max_dv = dv_ref * ENV_MAX_DV_COEFF, set per
# episode in CWRendezvousEnv.reset() from that scenario/distance's analytic
# reference Δv (libs/reference.py). Relative to dv_ref rather than a fixed
# m/s value on purpose — dv_ref scales linearly with distance, so this one
# coefficient gives the same proportional headroom over the optimal transfer
# at every curriculum distance (30 m through ENV_CURRICULUM_MAX_DISTANCE)
# without needing retuning when the curriculum range changes.
ENV_MAX_DV_COEFF = 1.5

# --- Total Δv BUDGET curriculum, opt-in via --fuel-curriculum ---
# A SECOND, independent curriculum axis (training.py's
# DvBudgetCurriculumCallback), gated behind the distance curriculum reaching
# ENV_CURRICULUM_MAX_DISTANCE. Unlike ENV_MAX_DV_COEFF above (a PER-BURST cap
# on any single impulse), this caps the TOTAL cumulative dv_used allowed
# across the WHOLE episode: dv_budget = dv_ref * dv_budget_coeff, enforced in
# CWRendezvousEnv.step() by clipping (not zeroing) any commanded burn down to
# whatever budget remains once dv_used would exceed it — the tank runs
# genuinely dry, direction preserved, magnitude reduced. A per-burst cap
# alone doesn't limit burn COUNT: an earlier attempt capping only
# ENV_MAX_DV_COEFF found the agent just chained many separate near-cap burns
# and kept total dv_used/dv_ref sitting around ~20x even at a tight 1.2x
# per-burst cap, since the flat docking bonus doesn't scale with efficiency.
#
# Starts, once activated, at a deliberately generous 50x dv_ref (looser than
# the ~20x ratio observed in practice, so it's initially non-binding) and
# ratchets down MULTIPLICATIVELY (not linearly — see ENV_DV_BUDGET_SHRINK)
# toward a tight 3x floor as dock rate stays high, mirroring
# CurriculumCallback's dock-rate-gated advance/stall-regress shape.
ENV_DV_BUDGET_COEFF_START = 50.0
ENV_DV_BUDGET_COEFF_FLOOR = 3.0
ENV_DV_BUDGET_SHRINK = 0.85  # multiplicative ratchet per dock-rate window

# Burn deadzone / minimum-impulse-bit, as a FRACTION of the episode's max_dv
# (CWRendezvousEnv.reset() sets burn_deadzone = ENV_BURN_DEADZONE_FRAC *
# max_dv — relative to dv_ref for the same reason as ENV_MAX_DV_COEFF above,
# so it stays proportionally sized across the whole curriculum range instead
# of needing a hand-picked absolute m/s value). Any commanded burn whose
# magnitude ‖a‖ is below this is treated as EXACTLY zero — no Δv charged, no
# velocity change. Two reasons:
#   1. Physical: real thrusters have a minimum impulse bit and an off state;
#      an arbitrarily small continuous burn isn't realizable anyway.
#   2. RL-critical: dv_used sums ‖aₜ‖ over EVERY agent step, but a neural-net
#      actor can't output exactly zero, so without a deadzone every "coast"
#      step leaks a little Δv and a long low-fuel coast accumulates MORE waste
#      than a fast burn-straight-in dock — which is exactly why the policy
#      refuses to coast and docks fast (~23x optimal). The deadzone makes
#      coasting free and representable (the actor just aims below it), which
#      is what lets the fuel-optimal slow two-impulse transfer actually win.
# 0.2 of max_dv leaves genuine correction burns well clear of the deadzone
# while covering the actor's near-zero coast-leakage floor. Set to 0 to disable.
ENV_BURN_DEADZONE_FRAC = 0.2

# Base (max-curriculum) initial condition. Only the sign/quadrant is
# randomized per episode (see CWRendezvousEnv._sample_initial_position) —
# this vector just fixes the V-bar magnitude used at full curriculum
# distance for the default "vbar" scenario.
ENV_INITIAL_STATE_VBAR = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# --- Scenario config (mandatory goals 1 & 2 from CLAUDE.md) ---
# "vbar": pure V-bar (x) displacement, sign randomized each episode (+x/-x).
# "rbar": coupled x/z displacement with opposite signs, sign-combo
#         randomized each episode between (+x,-z) and (-x,+z).
SCENARIO = os.environ.get("AE4350_SCENARIO", "vbar")
RBAR_X_TO_Z_RATIO = 2.0  # matches the Δx = 2·Δz relation in goal 2 strategy 1

# Normalization multiplier for the dv_used observation element (see
# libs/normalization.py): the norm scale is DV_USED_NORM_MULT * max_dv for
# that episode, not a fixed absolute value — max_dv (= dv_ref * max_dv_coeff)
# already varies ~12x between "vbar" and "rbar" scenarios and linearly with
# curriculum distance, so a single fixed scale can't cover both without
# saturating dv_used to 1.0 for most of the episode in whichever case it
# wasn't tuned for. 5x max_dv gives headroom for a multi-burn trajectory to
# spend a few times the single-burn cap before the feature saturates.
DV_USED_NORM_MULT = 5.0

# --- Training defaults ---
# The discount has to make the SLOW fuel-optimal transfer's terminal reward
# survive back to the early "coast, don't burn" decisions, otherwise the
# agent just fast-burns (see the ENV_FUEL_COEFF note). The number of AGENT
# steps to that payoff is what matters, and that is now set by ENV_DT_AGENT,
# not the physics dt:
#   old: dt = 5 s  -> a ~1 orbit (~5800 s) transfer is ~1160 steps, so you
#        needed gamma ~0.9999 (0.9999**1160 ~= 0.89) to see the reward — and
#        that long an effective horizon (1/(1-gamma) ~= 10000) diverges the
#        critic on anything but a tiny net.
#   now: dt_agent = 100 s -> the same transfer is ~58 agent steps, so a MUCH
#        gentler gamma sees it: 0.99**58 ~= 0.56. Effective horizon
#        1/(1-0.99) = 100 steps ~ one max episode (ENV_TIMEOUT/dt_agent), so
#        credit assignment is easy and wide/deep nets train stably.
# This is the whole point of raising ENV_DT_AGENT: trade the (physically
# free) fine control cadence for a short-horizon MDP a smart net can learn.
GAMMA = 0.99
MAX_STEPS = int(ENV_TIMEOUT / ENV_DT) + 1

ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
GRAD_CLIP_NORM = 1.0
LOG_EVERY = 10
TAU = 0.005
BATCH_SIZE = 256 * 5
MIN_BUFFER = 5_000
REPLAY_BUFFER_SIZE = 300_000

# Parallel envs for rollout collection (the single biggest, safest lever for
# wall-clock speed — env.step() is a cheap analytic matrix multiply, so
# throughput scales ~linearly with cores). Override with --n-envs.
NUM_ENVS = max(1, min((os.cpu_count() or 4) - 1, 16))

# Exploration noise (OU, TD3-style — NOT tied to any fuel budget). Kept as
# OU rather than i.i.d. Gaussian on purpose: exploration is a temporally
# correlated random walk, so it can linger near zero (or hold a sustained
# push) for many consecutive agent steps — needed to ever stumble into the
# true-optimal V-bar transfer's ~230+ consecutive steps of coasting, which
# i.i.d. per-step noise essentially cannot produce by chance.
#
# ACTION_NOISE_STD_* is the quantity that actually matters — the noise's
# stationary std in the env's native [-1,1] action units (always [-1,1]
# regardless of ENV_MAX_DV_COEFF; CWRendezvousEnv.step() scales u to the
# physical impulse internally). It is NOT the same as the `sigma` parameter
# OrnsteinUhlenbeckActionNoise takes: for the discrete OU update SB3 uses
# (x += theta*(mean-x)*dt + sigma*sqrt(dt)*noise), the stationary std is
# sigma / sqrt(2*theta - theta**2*dt), ~1.83x the sigma parameter at
# OU_THETA/OU_DT below — verified empirically (see scratch check). The
# previous ACTION_NOISE_SIGMA_* constants were passed straight into `sigma`
# without this correction, so the *actual* std was ~1.83x bigger than the
# number suggested — with the old 0.5 that meant a stationary std of ~0.94
# on a box of half-width 1 (i.e. saturating near +-1 almost constantly),
# leaving the burn deadzone (0.2, see ENV_BURN_DEADZONE_FRAC) essentially
# unreachable through nearly all of training (decay only completes at
# NOISE_DECAY_FRAC * TOTAL_TIMESTEPS). STD_START=0.25 instead gives noise a
# genuine chance to dip under the deadzone (~30% instantaneous probability
# of the 2-D noise norm alone landing under 0.2) while still producing
# meaningful directed excursions (real optimal actions rarely need more than
# a fraction of max_dv anyway).
OU_THETA = 0.15   # SB3 OrnsteinUhlenbeckActionNoise default; pinned explicitly
OU_DT    = 0.01   # so the amplification-factor math below can't silently drift
OU_STD_PER_SIGMA = (2 * OU_THETA - OU_THETA ** 2 * OU_DT) ** -0.5

ACTION_NOISE_STD_START = 0.15
ACTION_NOISE_STD_END   = 0.001
ACTION_NOISE_SIGMA_START = ACTION_NOISE_STD_START / OU_STD_PER_SIGMA
ACTION_NOISE_SIGMA_END   = ACTION_NOISE_STD_END / OU_STD_PER_SIGMA
NOISE_DECAY_FRAC = 0.25  # ~3x faster than the old 0.7 — noise was still near
# sigma_start at 20% of training, starving the agent of low-noise fuel-
# efficiency practice for the back 80% of the run.

# TD3 target-policy-smoothing noise. SB3 defaults (0.2 / 0.5) already assume
# the actual action range here, [-1, 1] — no rescaling needed (see above).
TD3_TARGET_POLICY_NOISE = 0.2
TD3_TARGET_NOISE_CLIP   = 0.5

TOTAL_TIMESTEPS = 10_000_000

# --- Evaluation / plotting defaults ---
SMOOTHING_WINDOW = 20
DOCK_RATE_WINDOW = LOG_EVERY
CLOSE_NOTE_THRESHOLD = 5.0
BOUNDARY_WARNING_FACTOR = 0.8
TRAINED_MODEL_DIR = "trained"
DIAGNOSTICS_PLOT_PATH = "out/diagnostics.png"
TRAINING_HISTORY_PATH = "out/training_history.npz"
