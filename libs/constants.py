import numpy as np

# --- Orbit / physics constants ---
EARTH_MU = 3.986 * 10 ** 14  # m^3 / s^2
ORBIT_RADIUS = (6378 + 600) * 10 ** 3  # m
OMEGA = np.sqrt(EARTH_MU / ORBIT_RADIUS ** 3)
ORBIT_PERIOD = 2 * np.pi / OMEGA

# --- Environment defaults ---
ENV_DT = 5.0
ENV_MAX_DV = 0.2
ENV_BOUNDARY = 1000.0
ENV_TIMEOUT = 1 * ORBIT_PERIOD
ENV_POS_TOLERANCE = 1.0
ENV_VEL_TOLERANCE = 0.01
ENV_VEL_COEFF = 5.0
ENV_FUEL_COEFF = 10.0
ENV_SHAPING_COEFF = 10.0
ENV_BONUS = 500.0
ENV_INITIAL_STATE_VBAR = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
ENV_INITIAL_STATE_XPLUS = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
ENV_MAX_DIR_CHANGE = 4

# --- Training defaults ---
GAMMA = 0.97
NUM_EPISODES = 15000
MAX_STEPS = int(ENV_TIMEOUT/ENV_DT)
ACTOR_LR = 1e-5
CRITIC_LR = 1e-3
GRAD_CLIP_NORM = 1.0
LOG_EVERY = 10
TAU = 0.008  # soft update rate — add to constants.py
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 64
MIN_BUFFER = 1000  # don't train until buffer has this many samples

# --- Evaluation / plotting defaults ---
SMOOTHING_WINDOW = 20
DOCK_RATE_WINDOW = 50
CLOSE_NOTE_THRESHOLD = 5.0
BOUNDARY_WARNING_FACTOR = 0.8
TRAINED_ACTOR_PATH = "trained/actor.pt"
TRAINED_CRITIC_PATH = "trained/critic.pt"
DIAGNOSTICS_PLOT_PATH = "out/diagnostics.png"
TRAINING_HISTORY_PATH = "out/training_history.npz"