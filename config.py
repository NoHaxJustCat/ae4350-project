import os
import numpy as np

MODE_2D: bool = True

PHYS_STATE_DIM: int = 4 if MODE_2D else 6
OBS_DIM: int = PHYS_STATE_DIM + 2          # + dv_used + braking flag
ACTION_IMPULSE_DIM: int = 2 if MODE_2D else 3
ACTION_DIM: int = ACTION_IMPULSE_DIM + 1   # + coast duration

# --- Orbit / physics ---
EARTH_MU = 3.986e14                        # m^3/s^2
ORBIT_RADIUS = (6378 + 600) * 1e3          # m
OMEGA = np.sqrt(EARTH_MU / ORBIT_RADIUS ** 3)
ORBIT_PERIOD = 2 * np.pi / OMEGA

# --- Timesteps ---
ENV_DT_PHYS = 5.0                          # collision/docking sampling substep
ENV_DT_AGENT = 100.0                       # decision interval
ENV_DT = ENV_DT_AGENT
# 3 orbits, not 2: a correction leg needs a full orbit and t* is independent
# of distance, so at 2 a missed transfer can never be corrected.
ENV_TIMEOUT = 3 * ORBIT_PERIOD
MAX_STEPS = int(ENV_TIMEOUT / ENV_DT) + 1

# Coast length in agent-dt units; 58 units = 1 orbit = the V-bar coast leg.
ENV_COAST_MIN_UNITS = 1
ENV_COAST_MAX_UNITS = 72

# --- Distance curriculum ---
ENV_CURRICULUM_ENABLED = True
ENV_CURRICULUM_START_DISTANCE = 30.0       # m
ENV_CURRICULUM_MAX_DISTANCE = 1000.0       # m
ENV_CURRICULUM_INCREMENT = 10.0            # m per cleared window
# 4, not 2: the optimal coasting trajectory peaks at 1.64x the initial
# distance off-axis, and a +-15% burn error reaches 4.1x, so at 2 the agent
# was punished for exploring near the correct solution. Safe to change only
# because OBS_POS_SCALE below is no longer derived from it.
ENV_CURRICULUM_BOUNDARY_MULT = 4.0
ENV_BOUNDARY = ENV_CURRICULUM_MAX_DISTANCE * ENV_CURRICULUM_BOUNDARY_MULT

# Observation position scale. Pinned, NOT derived from ENV_BOUNDARY: coupling
# them meant raising the boundary rescaled every observation and invalidated
# every saved model (100% -> 0% dock rate, measured).
OBS_POS_SCALE = 2000.0
DV_USED_NORM_MULT = 5.0                    # dv_used obs scale = this * max_dv

# --- Angle curriculum ("random" only) ---
# Sector half-width around the V-bar axis, widened as dock rate holds. MAX = 90
# is the uniform draw. START must be tiny: the V-bar specialist's usable
# envelope is only ~+-0.12 deg.
ENV_ANGLE_CURRICULUM_START_DEG = 0.1
ENV_ANGLE_CURRICULUM_MAX_DEG = 90.0
ENV_ANGLE_CURRICULUM_INCREMENT_DEG = 1.0

# --- Rewards ---
ENV_POS_TOLERANCE = 5.0                    # m, docking closest-approach test
ENV_SHAPING_COEFF = 10.0
ENV_BONUS = 50.0
ENV_FUEL_COEFF = 500.0                     # reward = coeff / max(dv/dv_opt, floor)
ENV_FUEL_OPT_FLOOR = 0.9                   # must stay < 1; un-floored, training collapses
ENV_STOP_COEFF = 500.0                     # rewards low arrival speed
ENV_STOP_VEL_SCALE_FRAC = 0.05
ENV_MAX_DV_COEFF = 1.5                     # max_dv = dv_ref * this
ENV_BURN_DEADZONE_FRAC = 0.05              # smaller commands are applied as zero

# --- Scenarios ---
SCENARIO = os.environ.get("AE4350_SCENARIO", "vbar")
RBAR_X_TO_Z_RATIO = 2.0
# "random" actuator scale: max_dv = ENV_MAX_DV_COEFF * this * dv_opt.
#
# 1.0 (max_dv = 1.5*dv_opt), not the old 3*pi/4 (3.53*dv_opt). dv_opt varies
# 23.6x with direction, so the action scale does too -- one unit of normalized
# action does 23.6x more physical damage off-axis while the 5 m tolerance stays
# fixed. Measured dock rate of the OPTIMAL action under 0.02 burn noise at
# 30 m, as the cap shrinks:
#       cap    0deg  15deg  45deg  90deg
#      3.53     82%    10%     3%    15%
#      1.50    100%    28%    12%    34%
# The largest optimal burn is 0.746*dv_opt (at 90 deg), so a 1.5 cap still
# leaves 2x headroom for corrections.
#
# This drops the old property that a vbar model transfers into "random"
# unchanged -- deliberate: seeding from the V-bar specialist starts at 1000 m,
# where the basin is narrowest, which is the worst place to begin. Random now
# trains from scratch up the distance curriculum instead.
# random-only: vbar/rbar set dv_ref from their analytic references.
ENV_RANDOM_DV_REF_MULT = 1.0
ENV_RANDOM_ANGLE_TABLE_N = 180

# --- TD3 ---
GAMMA = 0.99
LEARNING_RATE = 1e-4
TAU = 0.005
BATCH_SIZE = 256 * 5
MIN_BUFFER = 5_000
REPLAY_BUFFER_SIZE = 300_000
TRAIN_FREQ = 1
GRADIENT_STEPS = -1
TD3_TARGET_POLICY_NOISE = 0.2
TD3_TARGET_NOISE_CLIP = 0.5
TOTAL_TIMESTEPS = 10_000_000

# --- Network ---
# n_blocks 2 with relu is the deepest that trains; 3 saturates the actor.
NET_ARCH = [256, 256]
FEATURES_DIM = NET_ARCH[0]
N_BLOCKS = 2
ACTIVATION = "relu"

# --- Compute ---
DEVICE = "cuda"
NUM_ENVS = 6
VEC_ENV = "dummy"
TORCH_THREADS = 5

# --- Exploration noise (OU) ---
# ACTION_NOISE_STD_* is the stationary std in native action units, not SB3's
# sigma; OU_STD_PER_SIGMA converts.
OU_THETA = 0.15
OU_DT = 0.01
OU_STD_PER_SIGMA = (2 * OU_THETA - OU_THETA ** 2 * OU_DT) ** -0.5
ACTION_NOISE_STD_START = 0.05
ACTION_NOISE_STD_END = 0.01
NOISE_DECAY_FRAC = 0.5

# --- Checkpointing / evaluation ---
CHECKPOINT_FREQ = 10_000
KEEP_LAST_CHECKPOINTS = 3
EVAL_FREQ = 1_000
EVAL_EPISODES = 20

# --- Logging ---
LOG_EVERY = 10
SMOOTHING_WINDOW = 20
DOCK_RATE_WINDOW = LOG_EVERY
THROUGHPUT_LOG_EVERY = 2000
# 60s, not 30: the diagnostics set is 11 separate figures (~5.7s to write), so
# a 30s cadence spent ~19% of training on plotting.
DIAG_UPDATE_SECONDS = 60.0
OUT_DIR = "tmp"        # every run writes to tmp/<run_tag>/
