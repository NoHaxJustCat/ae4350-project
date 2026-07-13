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
ENV_DT = 5.0
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
# gamma=0.97 gives an effective horizon of ~33 steps (1/(1-gamma)) — but a
# classical V-bar two-impulse hop takes close to a full orbital period
# (~1160 steps here). Under the old gamma the terminal dock bonus and any
# payoff more than ~30 steps out was invisible to the critic, so the agent
# could never learn that an early burn pays off hundreds of steps later.
# gamma=0.999 gives an effective horizon of ~1000 steps, matching the
# actual task timescale.
#
# Raised 0.9995 -> 0.9999 to attack the FUEL problem specifically. With no
# per-step time cost, the only thing implicitly rewarding a fast dock is the
# discount: the fuel-optimal V-bar transfer pays off ~1160 steps out, and at
# gamma=0.9995 that terminal reward is discounted by 0.9995**1160 ~= 0.56,
# while a wasteful 28-step burn is discounted by only ~0.99 — so the two
# score about the same despite a 100x fuel difference, and the agent
# rationally picks the fast burn. At gamma=0.9999, 0.9999**1160 ~= 0.89, so
# the slow low-fuel transfer's terminal payoff survives the discount and
# actually wins. Trade-off: a longer effective horizon makes credit
# assignment harder for the critic — pairs best with the LayerNorm --arch
# smart critic; plain MLPs may need the extra capacity of the wider sweep
# members to cope. Drop back to 0.9995 if runs stop learning.
GAMMA = 0.9999
MAX_STEPS = int(ENV_TIMEOUT / ENV_DT) + 1

ACTOR_LR = 1e-5
CRITIC_LR = 1e-5
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
