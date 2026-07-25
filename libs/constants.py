import os
import numpy as np

MODE_2D: bool = True

# Derived dims (do not edit directly)
PHYS_STATE_DIM: int = 4 if MODE_2D else 6   # [x,z,xdot,zdot] or [x,y,z,xdot,ydot,zdot]
# obs = phys state + dv_used + braking-phase flag. The flag is an explicit
# feature (not inferred from "position ~ 0") because at the target the
# normalized position is too close to other small positions for LayerNorm to
# distinguish reliably — see CWRendezvousEnv.braking_phase / _brake_step.
OBS_DIM: int = PHYS_STATE_DIM + 2           # 6 (2D) or 8 (3D)
# action = impulse per axis + 1 coast-duration scalar (see ENV_COAST_* below
# and CWRendezvousEnv.step()) so an episode is a handful of decisions instead
# of one every 100 s for 100+ steps.
ACTION_IMPULSE_DIM: int = 2 if MODE_2D else 3
ACTION_DIM: int = ACTION_IMPULSE_DIM + 1

# --- Orbit / physics constants ---
EARTH_MU = 3.986 * 10 ** 14            # m^3/s^2
ORBIT_RADIUS = (6378 + 600) * 10 ** 3  # m
OMEGA = np.sqrt(EARTH_MU / ORBIT_RADIUS ** 3)
ORBIT_PERIOD = 2 * np.pi / OMEGA

# --- Curriculum learning ---
ENV_CURRICULUM_ENABLED = True
ENV_CURRICULUM_START_DISTANCE = 30.0   # starting distance [m]
ENV_CURRICULUM_MAX_DISTANCE = 1000.0   # final distance [m]
ENV_CURRICULUM_INCREMENT = 10.0        # distance added per successful dock [m]
# excursion_limit = curriculum_distance * this (see CWRendezvousEnv.reset()).
ENV_CURRICULUM_BOUNDARY_MULT = 2.0

# --- Angle curriculum ("random" scenario only) ---
# Second curriculum axis for the "random" scenario, gated behind the distance
# ramp (training.py's AngleCurriculumCallback). Instead of drawing the initial
# displacement direction uniformly over the whole circle from step one, draw it
# from a sector of half-width `angle_half_width_deg` centred on the V-bar axis
# and widen it as the dock rate holds.
#
# Why: the achievable optimum varies 23.6x with direction and rises ~6x within
# 10 deg of V-bar, so V-bar is a CUSP in the target — and only 3.9% of uniform
# angles land within 2x of it. Worse, the optimal coasting trajectory peaks at
# 1.02x the initial distance at V-bar but 1.57-1.64x at 30-60 deg, against an
# excursion_limit of 2x: there is ~96% margin for noisy exploration of the
# efficient solution at V-bar but only ~20% off-axis. Uniform sampling
# therefore punishes exactly the coasting behaviour that makes a transfer
# cheap, and the policy settles on brute-force thrusting at ~11x optimal
# instead. Starting in the V-bar sector (where a specialist already reaches
# 1.15x) and widening keeps a working solution in the replay buffer the whole
# way out.
#
# MAX = 90 deg is the full circle, not a half of it: the sector is centred on
# BOTH V-bar directions (+x and -x) with equal probability, so half-width 90
# reproduces the uniform draw exactly (see CWRendezvousEnv._sample_direction).
#
# START is 2 deg, not wider, because the V-bar specialist's dock is an
# OPEN-LOOP point solution with almost no robustness: seeded from out/best_1
# (which matches its 1.15x-optimal dock exactly at 0 deg once dv_ref agrees —
# see ENV_RANDOM_DV_REF_MULT) it already misses by 0.5 deg and goes
# out-of-bounds past 1.5 deg, because it fires one burn and never corrects.
# So expect a LOW dock rate at first even at the start width: the opening
# phase is the policy learning a mid-coast correction burn, not free-riding on
# the seed. The sector floors at START and only widens once dock rate holds,
# so a slow start costs nothing but time.
ENV_ANGLE_CURRICULUM_START_DEG = 2.0
ENV_ANGLE_CURRICULUM_MAX_DEG = 90.0
ENV_ANGLE_CURRICULUM_INCREMENT_DEG = 5.0

# --- Environment timesteps ---
# ENV_DT_PHYS: fine physics/collision substep (CW's STM is exact, so this adds
#   no integration error — it only controls docking/OOB sampling resolution).
# ENV_DT_AGENT: the agent's decision interval (impulse + coast choice). Larger
#   than dt_phys so an episode spans far fewer agent steps, letting a lower
#   gamma still see the terminal reward (see GAMMA below).
ENV_DT_PHYS = 5.0
ENV_DT_AGENT = 100.0
ENV_DT = ENV_DT_AGENT  # alias used throughout (MAX_STEPS, evaluate.py, env default)

# --- Coast-duration action ---
# Coast length in agent-dt units. ~58 units = 1 orbit, ~29 = half orbit (the
# V-bar two-impulse transfer's coast leg); MAX comfortably exceeds a full
# orbit so both are reachable in one decision.
ENV_COAST_MIN_UNITS = 1
ENV_COAST_MAX_UNITS = 72

# Out-of-bounds safety ceiling, derived from the curriculum range so it always
# covers the largest excursion_limit reset() can compute.
ENV_BOUNDARY = ENV_CURRICULUM_MAX_DISTANCE * ENV_CURRICULUM_BOUNDARY_MULT
ENV_TIMEOUT = 2 * ORBIT_PERIOD
ENV_POS_TOLERANCE = 5.0
ENV_VEL_COEFF = 10.0
ENV_SHAPING_COEFF = 10.0
ENV_BONUS = 50.0

# Fuel bonus (paid only on a dock, in _brake_step):
#     reward_fuel = coeff / max(dv_used/dv_opt, ENV_FUEL_OPT_FLOOR)
# Constant-elasticity 1/ratio (a proportional Δv improvement is worth the same
# relative reward at any ratio, so the gradient never vanishes at high ratio).
# Graded against dv_opt, the TRUE achievable optimum for the episode's own
# geometry (libs/reference.py), not dv_ref — otherwise every solution below the
# analytic reference sits on one flat plateau, which for "rbar" would let the
# policy settle on the worse of the two classical strategies.
ENV_FUEL_COEFF = 500.0
# Floor on dv_used/dv_opt: must stay < 1.0 (matching the optimum must not
# saturate the bonus) and must not be removed (an un-floored version let the
# optimizer ride the physical feasibility edge and collapsed training — see
# git history "Restore the fuel-bonus floor..."). 0.9 gives 10% headroom to
# reward beating the optimum, capped at ~11% above the at-optimum reward.
ENV_FUEL_OPT_FLOOR = 0.9

# Stopping bonus: rewards low arrival speed after the terminal brake, so the
# agent performs a real rendezvous instead of a fly-through (docking is
# position-only, so a fly-through would otherwise be the cheapest "dock").
# Smooth 1/(1+vel_error/v_scale) decay, paid alongside the fuel bonus in
# _brake_step; COEFF is sized to always outweigh the brake's extra Δv cost.
ENV_STOP_COEFF = 500.0
ENV_STOP_VEL_SCALE_FRAC = 0.05

# Per-burn actuator cap: max_dv = dv_ref * this. Relative to dv_ref (not a
# fixed m/s value) so the headroom over the optimal transfer is the same
# proportion at every curriculum distance.
ENV_MAX_DV_COEFF = 1.5

# --- Total Δv budget curriculum, opt-in via --fuel-curriculum ---
# Second curriculum axis (training.py's DvBudgetCurriculumCallback), gated
# behind the distance curriculum reaching its max. Caps TOTAL episode dv_used
# (clipping, not zeroing, any burn that would exceed it) rather than any one
# burn, since a per-burst cap alone doesn't limit burn count.
ENV_DV_BUDGET_COEFF_START = 50.0
ENV_DV_BUDGET_COEFF_FLOOR = 3.0
# Cube-root of 0.85 rather than 0.85 directly: finer ratchet steps stopped a
# live run's dock rate oscillating 100%->0%->100% at each shrink.
ENV_DV_BUDGET_SHRINK = 0.85 ** (1 / 3)  # ~0.9473

# Minimum-impulse-bit: a commanded burn below this fraction of max_dv is
# charged and applied as exactly zero (free coasting). Was 0.2 pre-coast-
# action (needed to swallow per-step actor leakage); lowered to 0.05 once the
# coast action removed that failure mode, since 0.2 (0.3*dv_ref) was larger
# than the optimal transfer burn itself and would have zeroed it. The brake
# bypasses the deadzone entirely (CWRendezvousEnv._brake_step).
ENV_BURN_DEADZONE_FRAC = 0.05

ENV_INITIAL_STATE_VBAR = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# --- Scenarios ---
# "vbar"  : pure V-bar (x) displacement, sign randomized each episode.
# "rbar"  : coupled x/z displacement, sign-combo randomized each episode.
# "random": displacement at a uniformly random angle each episode. No analytic
#           formula applies, so dv_opt is looked up per episode from a numeric
#           per-angle table (see ENV_RANDOM_* below and libs/reference.py).
SCENARIO = os.environ.get("AE4350_SCENARIO", "vbar")
RBAR_X_TO_Z_RATIO = 2.0  # Δx = 2·Δz, matches goal 2 strategy 1

# "random" actuator scale: dv_ref = this * dv_opt (no analytic reference
# exists). Gives max_dv = 3.54*dv_opt, clearing the largest measured single
# optimal burn (0.875*dv_opt) with ample headroom to explore/correct.
#
# 3*pi/4 rather than a round 2.5 so that AT V-BAR this scenario's actuator
# scale is IDENTICAL to the "vbar" scenario's: there dv_ref = 0.5 *
# dv_vbar_two_impulse_rr = 0.25*omega*d, and dv_opt per metre at V-bar is the
# two-V-bar-impulse optimum omega/(3*pi), so the ratio is exactly 0.25*3*pi.
#
# This matters because a "vbar" model has to be loadable into this scenario as
# a starting point (runs/smart_cuda_random.ps1 resumes out/best_1 and ramps the
# angle sector outward from V-bar). Docking is a 5 m closest-approach test from
# up to 1000 m — 0.5% of the initial distance — so the learned open-loop burn
# is far more sensitive to max_dv than 0.5%: at the old 2.5 the two scenarios'
# max_dv differed by 6% at the SAME geometry, which was enough to turn the
# specialist's 1.15x-optimal dock into an out-of-bounds miss on transfer.
ENV_RANDOM_DV_REF_MULT = 3.0 * np.pi / 4.0
# Per-direction dv_opt table resolution over [0, pi) (period-pi by the CW
# point-reflection symmetry). 180 = 1 deg; dv_opt is smooth so linear
# interpolation error is negligible.
ENV_RANDOM_ANGLE_TABLE_N = 180

# dv_used observation scale = this * max_dv (per-episode, since max_dv varies
# with scenario/distance). Headroom for a multi-burn trajectory to exceed the
# single-burn cap a few times before the feature saturates.
DV_USED_NORM_MULT = 5.0

# --- Training defaults ---
# gamma must let the terminal reward survive back to early "coast, don't burn"
# decisions. Raising ENV_DT_AGENT (fewer, longer agent steps per transfer) is
# what makes a low, stable gamma sufficient instead of needing gamma~0.9999
# with the fine physics dt, which diverges the critic on anything but a tiny
# net.
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

# Parallel envs for rollout collection; override with --n-envs.
NUM_ENVS = max(1, min((os.cpu_count() or 4) - 1, 16))

# Exploration noise (OU, TD3-style). OU rather than i.i.d. Gaussian so a
# sustained push or near-zero stretch can persist for many consecutive steps
# — needed to reach the long coasting stretches the optimal transfer requires.
#
# ACTION_NOISE_STD_* is the noise's stationary std in the env's native [-1,1]
# action units — NOT the same as OrnsteinUhlenbeckActionNoise's `sigma` param.
# For SB3's discrete OU update, stationary std = sigma / sqrt(2*theta -
# theta**2*dt) = sigma * OU_STD_PER_SIGMA (~1.83x at the theta/dt below), so
# ACTION_NOISE_SIGMA_* below applies that correction.
OU_THETA = 0.15
OU_DT = 0.01
OU_STD_PER_SIGMA = (2 * OU_THETA - OU_THETA ** 2 * OU_DT) ** -0.5

ACTION_NOISE_STD_START = 0.1
ACTION_NOISE_STD_END = 0.01
ACTION_NOISE_SIGMA_START = ACTION_NOISE_STD_START / OU_STD_PER_SIGMA
ACTION_NOISE_SIGMA_END = ACTION_NOISE_STD_END / OU_STD_PER_SIGMA
NOISE_DECAY_FRAC = 0.35  # fraction of training over which noise decays to END

# TD3 target-policy-smoothing noise; SB3 defaults already assume [-1,1] actions.
TD3_TARGET_POLICY_NOISE = 0.2
TD3_TARGET_NOISE_CLIP = 0.5

TOTAL_TIMESTEPS = 10_000_000

# --- Evaluation / plotting defaults ---
SMOOTHING_WINDOW = 20
DOCK_RATE_WINDOW = LOG_EVERY
CLOSE_NOTE_THRESHOLD = 5.0
BOUNDARY_WARNING_FACTOR = 0.8
TRAINED_MODEL_DIR = "trained"
DIAGNOSTICS_PLOT_PATH = "out/diagnostics.png"
TRAINING_HISTORY_PATH = "out/training_history.npz"
