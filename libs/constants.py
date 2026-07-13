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
ENV_BOUNDARY = 200.0
ENV_TIMEOUT = 2 * ORBIT_PERIOD
ENV_POS_TOLERANCE = 1.0
ENV_VEL_COEFF = 10.0
ENV_SHAPING_COEFF = 10.0
ENV_BONUS = 50.0
# Fuel bonus: paid only on a successful dock, inverse in cumulative dv_used
# (reward_fuel = coeff * (dv_used + eps)**-1 — see libs/env.py::step).
# Strictly additive on top of the dock bonus, so it can never make failing
# to dock look better than a wasteful dock (it's zero unless docked=True).
#
# Retuned for the fuel push: coeff 150 -> 25 and eps 1.0 -> 0.01. The old
# eps=1.0 flattened the whole low-dv end — 150/(0.01+1)=148.5 vs
# 150/(0.05+1)=142.9, only ~6 reward points between a near-optimal 0.01 m/s
# dock and a 5x-worse 0.05 m/s one, so the agent had almost no incentive to
# chase the last order of magnitude of fuel. With eps=0.01 the curve stays
# steep exactly where the optimum lives: 25/(0.0115+0.01)=1163 at the
# optimal V-bar dv vs 25/(0.5+0.01)=49 at a wasteful 0.5 m/s dock — a ~24x
# reward gap that makes low fuel strongly worth it. coeff dropped to 25 so
# the peak reward stays in a sane range for the critic to represent given
# the smaller eps blows the magnitude up. Earlier per-step linear / log1p
# costs were tried and dropped (vanishing low-end gradient / flat cost).
ENV_FUEL_COEFF = 25.0

# Physical per-burn actuator cap (m/s). Independent of any fuel budget —
# sized to comfortably cover a single optimal impulse for the largest
# curriculum distance (worst case ~0.03 m/s for the R-bar two-V-bar-impulse
# strategy at 100 m), with headroom for the policy to correct errors.
ENV_MAX_DV = 0.05

# Burn deadzone / minimum-impulse-bit (m/s). Any commanded burn whose
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
# Sized BELOW the smallest real impulse the optimal maneuver needs (~0.006
# m/s per burn at 100 m) so genuine burns still count, but above the actor's
# near-zero coast-leakage floor. Set to 0 to disable.
ENV_BURN_DEADZONE = 0.002

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

# Normalization scale for the dv_used observation element (see
# libs/normalization.py) — a fixed reference scale covering the typical
# range of cumulative Δv spent during an episode across both scenarios.
DV_USED_NORM_SCALE = 0.3

# --- Curriculum learning defaults ---
ENV_CURRICULUM_ENABLED = True
ENV_CURRICULUM_START_DISTANCE = 10.0   # starting distance [m]
ENV_CURRICULUM_MAX_DISTANCE = 100.0    # final distance [m] (matches ENV_INITIAL_STATE_VBAR norm)
ENV_CURRICULUM_INCREMENT = 5.0         # distance added per successful dock [m]

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

# Exploration noise (Gaussian, TD3-style — NOT tied to any fuel budget).
# Linearly decayed from *_START to *_END over the first NOISE_DECAY_FRAC of
# training so the policy can rely on its own (learned) near-zero output
# later in training instead of noise perpetually forcing nonzero burns.
ACTION_NOISE_SIGMA_START = 0.5 * ENV_MAX_DV
ACTION_NOISE_SIGMA_END   = 0.02 * ENV_MAX_DV
NOISE_DECAY_FRAC = 0.7

# TD3 target-policy-smoothing noise. SB3 defaults (0.2 / 0.5) assume an
# action range of roughly [-1, 1]; ours is [-ENV_MAX_DV, ENV_MAX_DV], so the
# defaults must be scaled down or they dwarf the action space entirely.
TD3_TARGET_POLICY_NOISE = 0.2 * ENV_MAX_DV
TD3_TARGET_NOISE_CLIP   = 0.5 * ENV_MAX_DV

TOTAL_TIMESTEPS = 10_000_000

# --- Evaluation / plotting defaults ---
SMOOTHING_WINDOW = 20
DOCK_RATE_WINDOW = LOG_EVERY
CLOSE_NOTE_THRESHOLD = 5.0
BOUNDARY_WARNING_FACTOR = 0.8
TRAINED_MODEL_DIR = "trained"
DIAGNOSTICS_PLOT_PATH = "out/diagnostics.png"
TRAINING_HISTORY_PATH = "out/training_history.npz"
