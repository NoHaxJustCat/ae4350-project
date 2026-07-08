import numpy as np
MODE_2D: bool = True

# Derived convenience constants (do not edit these directly)
# Physical state dim: 4 (2D) or 6 (3D)
PHYS_STATE_DIM: int = 4 if MODE_2D else 6
# Observation dim = physical state + dv_remaining scalar
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
ENV_TIMEOUT = 1 * ORBIT_PERIOD
ENV_POS_TOLERANCE = 1.0
ENV_VEL_TOLERANCE = 0.01
ENV_VEL_COEFF = 10.0
ENV_FUEL_COEFF = 200.0
ENV_SHAPING_COEFF = 10.0
ENV_BONUS = 20.0
ENV_INITIAL_STATE_VBAR = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
ENV_DV_BUDGET = 0.05
ENV_MAX_DV = ENV_DV_BUDGET/2

# --- Curriculum learning defaults ---
ENV_CURRICULUM_ENABLED = True
ENV_CURRICULUM_START_DISTANCE = 2.0   # starting distance [m]
ENV_CURRICULUM_MAX_DISTANCE = 100.0    # final distance [m] (matches ENV_INITIAL_STATE_VBAR norm)
ENV_CURRICULUM_INCREMENT = 2.0         # distance added per successful dock [m]

# --- Training defaults ---
GAMMA = 0.97
NUM_EPISODES = 50000
MAX_STEPS = int(ENV_TIMEOUT/ENV_DT) + 1
ACTOR_LR = 1e-5
CRITIC_LR = 1e-4
GRAD_CLIP_NORM = 1.0
LOG_EVERY = 1
TAU = 0.01  # soft update rate — add to constants.py
BATCH_SIZE = 256
MIN_BUFFER = 1000
REPLAY_BUFFER_SIZE = 50_000
# --- Evaluation / plotting defaults ---
SMOOTHING_WINDOW = 20
DOCK_RATE_WINDOW = LOG_EVERY
CLOSE_NOTE_THRESHOLD = 5.0
BOUNDARY_WARNING_FACTOR = 0.8
TRAINED_ACTOR_PATH = "trained/actor.pt"
TRAINED_CRITIC_PATH = "trained/critic.pt"
DIAGNOSTICS_PLOT_PATH = "out/diagnostics.png"
TRAINING_HISTORY_PATH = "out/training_history.npz"