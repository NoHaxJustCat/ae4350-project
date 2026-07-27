"""
TD3 training for the CW rendezvous env, one model per scenario.

Network, device, vec-env and optimizer settings live in config.py; the CLI
carries only what varies between runs. Launchers are in runs/*.ps1.
Outputs go to tmp/<run-tag>/ (see lib/paths.py).
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ACTION_NOISE_STD_END, ACTION_NOISE_STD_START, ACTIVATION, BATCH_SIZE, CHECKPOINT_FREQ,
    DEVICE, DIAG_UPDATE_SECONDS, ENV_ANGLE_CURRICULUM_INCREMENT_DEG,
    ENV_ANGLE_CURRICULUM_MAX_DEG, ENV_ANGLE_CURRICULUM_START_DEG, ENV_CURRICULUM_ENABLED,
    ENV_CURRICULUM_INCREMENT, ENV_CURRICULUM_MAX_DISTANCE, ENV_CURRICULUM_START_DISTANCE,
    EVAL_EPISODES, EVAL_FREQ, FEATURES_DIM, GAMMA, GRADIENT_STEPS, KEEP_LAST_CHECKPOINTS,
    LEARNING_RATE, MIN_BUFFER, NET_ARCH, NOISE_DECAY_FRAC, NUM_ENVS, N_BLOCKS, OMEGA,
    OU_DT, OU_STD_PER_SIGMA, OU_THETA, REPLAY_BUFFER_SIZE, TAU, TD3_TARGET_NOISE_CLIP,
    TD3_TARGET_POLICY_NOISE, THROUGHPUT_LOG_EVERY, TORCH_THREADS, TOTAL_TIMESTEPS,
    TRAIN_FREQ, VEC_ENV,
)
from lib.paths import RunPaths
from lib.plots.diagnostics import write_panels
from lib.rl.callbacks.checkpoint import BestModelEval, Checkpointer
from lib.rl.callbacks.curriculum import AngleCurriculum, DistanceCurriculum
from lib.rl.callbacks.monitor import EpisodeLogger, LiveDiagnostics, NoiseDecay, Throughput
from lib.rl.env import CWRendezvousEnv
from lib.rl.net import build_smart_policy_kwargs, shrink_actor_output_init
from lib.rl.obs import NormalizedObsEnv
from lib.rl.symmetry import CanonicalizeDirectionEnv


def make_env(scenario, distance=None, angle=None):
    def _init():
        # Each worker does only a small matmul per step, so a multi-threaded
        # pool just means n_envs processes fighting for the same cores.
        torch.set_num_threads(1)
        kwargs = {}
        if distance is not None:
            kwargs["curriculum_start_distance"] = distance
        if angle is not None:
            kwargs["angle_half_width_deg"] = angle
        env = CWRendezvousEnv(omega=OMEGA, scenario=scenario, **kwargs)
        return Monitor(NormalizedObsEnv(CanonicalizeDirectionEnv(env)))
    return _init


def parse_args():
    p = argparse.ArgumentParser(description="TD3 training for CW rendezvous.")
    p.add_argument("--scenario", choices=["vbar", "rbar", "random"], default="vbar")
    p.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
                   help="Absolute target, INCLUDING a resumed model's own steps.")
    p.add_argument("--run-tag", default="", help="Output goes to tmp/<run-tag>/.")
    p.add_argument("--resume-from", default=None,
                   help="Checkpoint .zip to continue from; reads its .curriculum.json "
                        "sidecar. The replay buffer is NOT restored.")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--noise-std-start", type=float, default=ACTION_NOISE_STD_START)
    p.add_argument("--noise-std-end", type=float, default=ACTION_NOISE_STD_END)
    p.add_argument("--noise-decay-frac", type=float, default=NOISE_DECAY_FRAC)

    p.add_argument("--curriculum-start-distance", type=float, default=None,
                   help="Start (and floor) the distance curriculum here [m].")
    p.add_argument("--eval-freq", type=int, default=EVAL_FREQ, help="0 disables.")
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)

    p.add_argument("--angle-curriculum", action="store_true",
                   help="(random only) Ramp the start-direction sector outward from V-bar.")
    p.add_argument("--angle-curriculum-start", type=float, default=ENV_ANGLE_CURRICULUM_START_DEG)
    p.add_argument("--angle-curriculum-max", type=float, default=ENV_ANGLE_CURRICULUM_MAX_DEG)
    p.add_argument("--angle-curriculum-increment", type=float,
                   default=ENV_ANGLE_CURRICULUM_INCREMENT_DEG)

    args = p.parse_args()
    if args.angle_curriculum and args.scenario != "random":
        raise SystemExit("--angle-curriculum only applies to --scenario random.")
    if not args.run_tag:
        args.run_tag = f"{args.scenario}_{time.strftime('%Y%m%d_%H%M%S')}"
    return args


def load_resume_state(args):
    """(distance, angle_half_width): CLI values, overridden by a sidecar."""
    distance = args.curriculum_start_distance
    angle = args.angle_curriculum_start if args.angle_curriculum else None
    if not args.resume_from:
        return distance, angle

    sidecar = Path(args.resume_from).with_suffix("").with_suffix(".curriculum.json")
    if not sidecar.exists():
        start = distance if distance is not None else ENV_CURRICULUM_START_DISTANCE
        print(f"WARNING: no sidecar at {sidecar} - starting at {start:.1f} m")
        return distance, angle

    data = json.loads(sidecar.read_text())
    distance = data["curriculum_distance"]
    print(f"Resuming curriculum_distance = {distance:.1f} m")
    # Only honour a saved sector if THIS run wants one: resuming a graduated
    # model into a fresh narrow ramp is legitimate.
    if args.angle_curriculum and "angle_half_width_deg" in data:
        angle = data["angle_half_width_deg"]
        print(f"Resuming angle sector = +-{angle:.2f} deg")
    return distance, angle


def build_vec_env(args, distance, angle):
    if VEC_ENV == "subproc" and NUM_ENVS > 1:
        method = "spawn" if platform.system() == "Windows" else "fork"
        vec_cls, vec_kwargs = SubprocVecEnv, dict(start_method=method)
    else:
        vec_cls, vec_kwargs = DummyVecEnv, {}
    print(f"vec_env: {vec_cls.__name__} x{NUM_ENVS}")
    return make_vec_env(make_env(args.scenario, distance, angle),
                        n_envs=NUM_ENVS, vec_env_cls=vec_cls, vec_env_kwargs=vec_kwargs)


def make_action_noise(action_dim, sigma):
    """OU, not i.i.d. Gaussian: a long near-zero stretch must be a plausible
    outcome, since the optimal transfer needs a sustained coast."""
    noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_dim),
                                         sigma=sigma * np.ones(action_dim),
                                         theta=OU_THETA, dt=OU_DT)
    return VectorizedActionNoise(noise, NUM_ENVS) if NUM_ENVS > 1 else noise


def build_model(args, env, sigma_start):
    action_dim = env.action_space.shape[0]
    if args.resume_from:
        try:
            model = TD3.load(args.resume_from, env=env, device=DEVICE)
        except ValueError as exc:
            if "spaces do not match" in str(exc):
                raise SystemExit(
                    f"Cannot resume {args.resume_from}: observation/action space "
                    f"mismatch. That model predates the coast action or the braking "
                    f"flag; train fresh.") from exc
            raise
        print(f"Resumed {args.resume_from} at {model.num_timesteps} steps "
              f"({args.total_timesteps - model.num_timesteps} remaining)")
        # The saved noise is wrapped for the ORIGINAL run's env count, which
        # breaks broadcasting if NUM_ENVS changed.
        model.action_noise = make_action_noise(action_dim, sigma_start)
        return model

    model = TD3(
        policy="MlpPolicy", env=env, learning_rate=LEARNING_RATE,
        buffer_size=REPLAY_BUFFER_SIZE, learning_starts=MIN_BUFFER, batch_size=BATCH_SIZE,
        tau=TAU, gamma=GAMMA, train_freq=(TRAIN_FREQ, "step"), gradient_steps=GRADIENT_STEPS,
        action_noise=make_action_noise(action_dim, sigma_start), policy_delay=2,
        target_policy_noise=TD3_TARGET_POLICY_NOISE, target_noise_clip=TD3_TARGET_NOISE_CLIP,
        policy_kwargs=build_smart_policy_kwargs(NET_ARCH, FEATURES_DIM, N_BLOCKS, ACTIVATION),
        verbose=0, device=DEVICE, seed=args.seed)
    # Without a small final-layer init a wide net saturates the actor at
    # +-max_dv from step one and never learns.
    shrink_actor_output_init(model)
    return model


def build_callbacks(args, paths, model, remaining, sigma_start, sigma_end):
    distance_cb = DistanceCurriculum(
        NUM_ENVS, ENV_CURRICULUM_INCREMENT, ENV_CURRICULUM_MAX_DISTANCE,
        # Floor where this run starts, so a run launched at 1000 m cannot
        # regress into the ramp it was told to skip.
        min_distance=(args.curriculum_start_distance if args.curriculum_start_distance
                      is not None else ENV_CURRICULUM_START_DISTANCE),
    ) if ENV_CURRICULUM_ENABLED else None

    angle_cb = AngleCurriculum(
        NUM_ENVS, args.angle_curriculum_start, args.angle_curriculum_max,
        args.angle_curriculum_increment, distance_cb) if args.angle_curriculum else None

    logger = EpisodeLogger(paths.episodes, NUM_ENVS, distance_cb=distance_cb, angle_cb=angle_cb)
    live = LiveDiagnostics(logger, paths, args.scenario, args.total_timesteps,
                           DIAG_UPDATE_SECONDS, angle_cb=angle_cb)

    callbacks = [
        logger,
        NoiseDecay(remaining, sigma_start, sigma_end, args.noise_decay_frac,
                   model.num_timesteps if args.resume_from else 0),
        Checkpointer(paths.checkpoints, f"{args.scenario}_td3", NUM_ENVS,
                     CHECKPOINT_FREQ, KEEP_LAST_CHECKPOINTS, distance_cb, angle_cb),
        Throughput(THROUGHPUT_LOG_EVERY),
        live,
    ]
    callbacks += [cb for cb in (distance_cb, angle_cb) if cb is not None]
    # Last, so it sees the curriculum state the others settled this step. The
    # angle curriculum then gates on ITS dock rate rather than the noisy one.
    if args.eval_freq > 0:
        eval_cb = BestModelEval(args.scenario, paths.best_model, NUM_ENVS,
                                args.eval_freq, args.eval_episodes, distance_cb, angle_cb)
        callbacks.append(eval_cb)
        if angle_cb is not None:
            angle_cb.eval_source = eval_cb
    elif angle_cb is not None:
        print("WARNING: --eval-freq 0 disables the deterministic gate the angle "
              "curriculum needs; it will fall back to the noisy dock rate.")
    return callbacks, logger, live


def main():
    args = parse_args()
    sigma_start = args.noise_std_start / OU_STD_PER_SIGMA
    sigma_end = args.noise_std_end / OU_STD_PER_SIGMA

    torch.set_num_threads(max(1, TORCH_THREADS))
    print(f"Platform: {platform.system()} | device: {DEVICE} | "
          f"torch threads: {TORCH_THREADS} | cpu_count: {os.cpu_count()}")
    print(f"net_arch: {NET_ARCH} | features_dim: {FEATURES_DIM} | "
          f"n_blocks: {N_BLOCKS} | activation: {ACTIVATION}")

    check_env(CWRendezvousEnv(omega=OMEGA, scenario=args.scenario))
    paths = RunPaths(args.run_tag, args.scenario)
    print(f"output: {paths.root}")

    distance, angle = load_resume_state(args)
    env = build_vec_env(args, distance, angle)
    model = build_model(args, env, sigma_start)

    # SB3 treats learn(total_timesteps) as an INCREMENT when resuming, so pass
    # the remaining budget and keep --total-timesteps meaning the full target.
    remaining = (max(args.total_timesteps - model.num_timesteps, 0)
                 if args.resume_from else args.total_timesteps)

    callbacks, logger, live = build_callbacks(args, paths, model, remaining,
                                              sigma_start, sigma_end)
    model.learn(total_timesteps=remaining, callback=callbacks, progress_bar=False,
                reset_num_timesteps=not bool(args.resume_from))

    model.save(str(paths.final_model))
    print(f"Model saved -> {paths.final_model}.zip")

    write_panels(logger.history, args.scenario, paths.panels)
    print(f"Plots -> {paths.panels}")
    # The periodic status write is throttled, so a short run could finish
    # between updates and never get one.
    live.write_status(time.perf_counter())
    live.write_history()
    print("Done.")


if __name__ == "__main__":
    main()
