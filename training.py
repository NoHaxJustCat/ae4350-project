"""
train.py  –  TD3 via Stable-Baselines3, parallel envs, one model per scenario.

Usage:
    python -u training.py --scenario vbar
    python -u training.py --scenario rbar --n-envs 8 --total-timesteps 2000000
"""

import argparse
from collections import deque
from pathlib import Path
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from libs.constants import (
    ACTION_NOISE_SIGMA_END,
    ACTION_NOISE_SIGMA_START,
    BATCH_SIZE,
    CRITIC_LR,
    DIAGNOSTICS_PLOT_PATH,
    DOCK_RATE_WINDOW,
    ENV_CURRICULUM_ENABLED,
    ENV_CURRICULUM_INCREMENT,
    ENV_CURRICULUM_MAX_DISTANCE,
    ENV_MAX_DV,
    GAMMA,
    LOG_EVERY,
    MIN_BUFFER,
    NOISE_DECAY_FRAC,
    NUM_ENVS,
    OMEGA,
    REPLAY_BUFFER_SIZE,
    SMOOTHING_WINDOW,
    TAU,
    TD3_TARGET_NOISE_CLIP,
    TD3_TARGET_POLICY_NOISE,
    TOTAL_TIMESTEPS,
    TRAINED_MODEL_DIR,
    TRAINING_HISTORY_PATH,
)
from libs.env import CWRendezvousEnv
from libs.normalization import NormalizedObsEnv
from libs.symmetry import CanonicalizeDirectionEnv
from libs.trajectory import plot_trajectory


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_single_env(scenario: str):
    def _init():
        env = CWRendezvousEnv(omega=OMEGA, scenario=scenario)
        env = CanonicalizeDirectionEnv(env)  # must wrap the raw (unnormalized) env
        env = NormalizedObsEnv(env)
        return Monitor(env)
    return _init


def moving_average(values, window=SMOOTHING_WINDOW):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_with_smooth(ax, data, label, color, title, ylabel, window=SMOOTHING_WINDOW):
    ax.plot(data, alpha=0.25, color=color)
    smoothed = moving_average(data, window)
    offset = len(data) - len(smoothed)
    ax.plot(np.arange(offset, len(data)), smoothed, color=color, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)
    ax.legend()


# ── Callbacks ─────────────────────────────────────────────────────────────────

class NoiseDecayCallback(BaseCallback):
    """Linearly anneals the exploration noise std from START to END over the
    first `decay_frac` of training. A constant-sigma OU/Gaussian process
    forces nonzero actions every step forever, which is exactly what starves
    the agent of ever seeing a "coast, don't burn" trajectory."""

    def __init__(self, total_timesteps: int, sigma_start: float, sigma_end: float,
                 decay_frac: float = NOISE_DECAY_FRAC):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = max(1, int(total_timesteps * decay_frac))

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        sigma = self.sigma_start + progress * (self.sigma_end - self.sigma_start)
        noise = self.model.action_noise
        if noise is None:
            return True
        # With n_envs > 1, SB3 wraps action_noise in VectorizedActionNoise,
        # which holds one deep-copied NormalActionNoise per sub-env.
        sub_noises = getattr(noise, "noises", [noise])
        for sub in sub_noises:
            sub._sigma[:] = sigma
        return True


class PeriodicCheckpointCallback(BaseCallback):
    """
    Saves the model every `save_freq_timesteps` environment timesteps
    (across ALL envs, not per-env — accounts for n_envs so the cadence
    matches real sim time, since _on_step fires once per vec-step ==
    n_envs collected transitions). Keeps only the most recent `keep_last`
    checkpoints so a long unattended remote run doesn't fill the disk.

    This is what remote_training.ps1's background poller now rsyncs back
    periodically, so a dropped SSH session doesn't lose all progress —
    only whatever happened since the last checkpoint.
    """

    def __init__(self, save_dir: Path, name_prefix: str, n_envs: int,
                 save_freq_timesteps: int = 1000, keep_last: int = 20):
        super().__init__(verbose=0)
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.keep_last = keep_last
        self.save_freq_calls = max(save_freq_timesteps // n_envs, 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq_calls == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(str(path))

            checkpoints = sorted(
                self.save_dir.glob(f"{self.name_prefix}_*_steps.zip"),
                key=lambda p: p.stat().st_mtime,
            )
            for old in checkpoints[:-self.keep_last]:
                old.unlink(missing_ok=True)
        return True


class CurriculumCallback(BaseCallback):
    """
    Single authority for curriculum progression across ALL parallel sub-envs.

    Previously each CWRendezvousEnv instance advanced its own
    curriculum_distance independently after 3 local consecutive docks. With
    N sub-envs running in separate subprocesses that meant N uncoordinated,
    invisible curricula drifting apart — which is what looked like "random"
    spawn distances during training. This callback instead tracks the dock
    rate over a rolling window of episode completions pooled across every
    env, and when it clears the threshold, advances one shared distance and
    pushes it to every sub-env via env_method("set_curriculum_distance").
    """

    def __init__(self, n_envs: int, increment: float, max_distance: float,
                 window: int = 20, dock_rate_threshold: float = 0.5):
        super().__init__(verbose=0)
        self.n_envs = n_envs
        self.increment = increment
        self.max_distance = max_distance
        self.dock_rate_threshold = dock_rate_threshold
        self._recent_docks = deque(maxlen=window)
        self.curriculum_distance = None

    def _on_training_start(self) -> None:
        self.curriculum_distance = self.training_env.get_attr("curriculum_distance")[0]

    @property
    def progress(self) -> str:
        """'k/n' of the pooled rolling window — how close to the next
        curriculum advance. Read by TrainingCallback for the log line."""
        n = len(self._recent_docks)
        k = sum(self._recent_docks)
        return f"{k}/{n}"

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i in range(self.n_envs):
            if dones[i]:
                self._recent_docks.append(bool(infos[i].get("docked", False)))

        ready = len(self._recent_docks) == self._recent_docks.maxlen
        if (ready and np.mean(self._recent_docks) >= self.dock_rate_threshold
                and self.curriculum_distance < self.max_distance):
            self.curriculum_distance = min(self.curriculum_distance + self.increment, self.max_distance)
            self.training_env.env_method("set_curriculum_distance", self.curriculum_distance)
            self._recent_docks.clear()
            print(f"[curriculum] dock rate >= {self.dock_rate_threshold:.0%} -> "
                  f"advancing to {self.curriculum_distance:.1f} m")
        return True


class TrainingCallback(BaseCallback):
    """
    Per-step callback that accumulates episode diagnostics across N parallel
    vec envs. Each sub-env finishes episodes at its own pace, so per-episode
    accumulators are tracked per env index, not globally.

    Trajectories/actions are read from info["state"] / info["applied_action"]
    (raw physical values written by CWRendezvousEnv.step), NOT from
    new_obs — new_obs is normalized-to-[0,1] policy input and would corrupt
    the trajectory plots.
    """

    def __init__(self, tmp_dir: Path, n_envs: int, log_every: int = LOG_EVERY,
                 curriculum_callback: "CurriculumCallback | None" = None):
        super().__init__(verbose=0)
        self.tmp_dir   = tmp_dir
        self.n_envs    = n_envs
        self.log_every = log_every
        self.curriculum_callback = curriculum_callback

        self.episode_rewards        = []
        self.episode_steps          = []
        self.episode_delta_vs       = []
        self.episode_docked         = []
        self.episode_r_pos_totals   = []
        self.episode_r_fuel_totals  = []
        self.episode_r_term_totals  = []
        self.episode_r_mile_totals  = []
        self.episode_curriculum_distances = []

        self._acc = [self._new_accumulator() for _ in range(n_envs)]
        self._episode_num = 0

        print(
            f"{'ep':>6} | {'steps':>5} | {'reward':>9} | {'r_pos':>8} | "
            f"{'r_fuel':>8} | {'r_term':>8} | {'r_mile':>8} | "
            f"{'dv':>7} | {'cur_d':>6} | {'docked':>6} | {'cur_prog':>8} | {'ms/ep':>8}"
        )
        print("-" * 123)

    @staticmethod
    def _new_accumulator():
        return {
            "delta_v": 0.0, "r_pos": 0.0, "r_fuel": 0.0, "r_term": 0.0,
            "r_mile": 0.0, "docked": False, "states": [], "actions": [],
            "start": time.perf_counter(),
        }

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i in range(self.n_envs):
            info = infos[i]
            acc  = self._acc[i]

            acc["states"].append(info["state"])
            acc["actions"].append(info.get("applied_action", np.zeros(2)))
            acc["delta_v"] += info.get("delta_v", 0.0)
            acc["r_pos"]   += info.get("reward_pos", 0.0)
            acc["r_fuel"]  += info.get("reward_fuel", 0.0)
            acc["r_term"]  += info.get("reward_terminal", 0.0)
            acc["r_mile"]  += info.get("reward_milestone", 0.0)
            if info.get("docked", False):
                acc["docked"] = True

            if dones[i]:
                ep_info = info.get("episode", {})
                reward  = ep_info.get("r", 0.0)
                steps   = ep_info.get("l", 1)

                cur_dist = info.get("curriculum_distance", float("nan"))

                self.episode_rewards.append(reward)
                self.episode_steps.append(steps)
                self.episode_delta_vs.append(acc["delta_v"])
                self.episode_docked.append(acc["docked"])
                self.episode_r_pos_totals.append(acc["r_pos"])
                self.episode_r_fuel_totals.append(acc["r_fuel"])
                self.episode_r_term_totals.append(acc["r_term"])
                self.episode_r_mile_totals.append(acc["r_mile"])
                self.episode_curriculum_distances.append(cur_dist)

                ep = self._episode_num
                if ep % self.log_every == 0:
                    elapsed   = time.perf_counter() - acc["start"]
                    dock_rate = np.mean(self.episode_docked[-DOCK_RATE_WINDOW:]) * 100
                    cur_prog  = self.curriculum_callback.progress if self.curriculum_callback else "n/a"
                    print(
                        f"{ep:>6} | {steps:>5} | {reward:>9.2f} | "
                        f"{acc['r_pos']:>8.2f} | {acc['r_fuel']:>8.2f} | "
                        f"{acc['r_term']:>8.2f} | {acc['r_mile']:>8.2f} | "
                        f"{acc['delta_v']:>7.4f} | {cur_dist:>6.1f} | {dock_rate:>5.1f}% | "
                        f"{cur_prog:>8} | {1000 * elapsed:>8.1f}"
                    )

                if i == 0 and (ep + 1) % self.log_every == 0:
                    tag       = f"ep_{ep + 1:04d}"
                    traj_path = self.tmp_dir / f"{tag}.png"
                    plot_trajectory(
                        acc["states"],
                        acc["actions"],
                        str(traj_path),
                        min_dv_display=ENV_MAX_DV * 0.01,
                    )
                    shutil.copy(traj_path, self.tmp_dir / "latest_trajectory.png")
                    np.savez(
                        self.tmp_dir / f"{tag}.npz",
                        states=np.array(acc["states"]),
                        rewards=np.array([reward]),
                        steps=np.array([steps]),
                        delta_v=np.array([acc["delta_v"]]),
                        docked=np.array([acc["docked"]]),
                    )

                self._acc[i] = self._new_accumulator()
                self._episode_num += 1

        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["vbar", "rbar"], default="vbar")
    p.add_argument("--n-envs", type=int, default=NUM_ENVS)
    p.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--device", default="cpu",
                    help="'cpu' is usually faster than 'cuda' for a network this small.")
    p.add_argument("--checkpoint-freq", type=int, default=10000,
                    help="Save the model every N environment timesteps.")
    p.add_argument("--keep-last-checkpoints", type=int, default=5,
                    help="Only keep the N most recent checkpoints on disk.")
    return p.parse_args()


def main():
    args = parse_args()

    t0 = time.perf_counter()
    check_env(CWRendezvousEnv(omega=OMEGA, scenario=args.scenario))
    print(f"Environment check completed in {time.perf_counter() - t0:.2f}s")

    Path(TRAINED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path("out").mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_envs = max(1, args.n_envs)
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_single_env(args.scenario), n_envs=n_envs, vec_env_cls=vec_cls)

    action_dim = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=ACTION_NOISE_SIGMA_START * np.ones(action_dim),
    )

    policy_kwargs = dict(net_arch=[400, 300])

    model = TD3(
        policy               = "MlpPolicy",
        env                  = env,
        learning_rate        = CRITIC_LR,
        buffer_size          = REPLAY_BUFFER_SIZE,
        learning_starts      = MIN_BUFFER,
        batch_size           = BATCH_SIZE,
        tau                  = TAU,
        gamma                = GAMMA,
        train_freq           = (1, "step"),
        gradient_steps       = -1,   # match number of env steps collected (== n_envs)
        action_noise         = action_noise,
        policy_delay         = 2,
        target_policy_noise  = TD3_TARGET_POLICY_NOISE,
        target_noise_clip    = TD3_TARGET_NOISE_CLIP,
        policy_kwargs        = policy_kwargs,
        verbose              = 0,
        device               = args.device,
    )

    curriculum_callback = CurriculumCallback(
        n_envs=n_envs,
        increment=ENV_CURRICULUM_INCREMENT,
        max_distance=ENV_CURRICULUM_MAX_DISTANCE,
    ) if ENV_CURRICULUM_ENABLED else None

    callback = [
        TrainingCallback(tmp_dir=tmp_dir, n_envs=n_envs, curriculum_callback=curriculum_callback),
        NoiseDecayCallback(
            total_timesteps=args.total_timesteps,
            sigma_start=ACTION_NOISE_SIGMA_START,
            sigma_end=ACTION_NOISE_SIGMA_END,
        ),
        PeriodicCheckpointCallback(
            save_dir=Path(TRAINED_MODEL_DIR) / "checkpoints",
            name_prefix=f"{args.scenario}_td3",
            n_envs=n_envs,
            save_freq_timesteps=args.checkpoint_freq,
            keep_last=args.keep_last_checkpoints,
        ),
    ]
    if curriculum_callback is not None:
        callback.append(curriculum_callback)

    model.learn(
        total_timesteps = args.total_timesteps,
        callback         = callback,
        progress_bar     = False,
    )

    model_path = f"{TRAINED_MODEL_DIR}/{args.scenario}_td3"
    model.save(model_path)
    print(f"Model saved -> {model_path}.zip")

    # ── Diagnostics plot ──────────────────────────────────────────────────────
    cb = callback[0]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"TD3 (SB3) Training Diagnostics — scenario={args.scenario}", fontsize=14)

    plot_with_smooth(axes[0, 0], cb.episode_rewards,       "reward",  "tab:blue",   "Total reward",          "reward")
    plot_with_smooth(axes[0, 1], cb.episode_delta_vs,      "delta-v", "tab:orange", "Total Δv used",         "m/s")
    plot_with_smooth(axes[0, 2], cb.episode_steps,         "steps",   "tab:green",  "Episode length",        "steps")
    plot_with_smooth(axes[1, 0], cb.episode_r_pos_totals,  "r_pos",   "tab:red",    "Position reward (sum)", "reward")
    plot_with_smooth(axes[1, 1], cb.episode_r_fuel_totals, "r_fuel",  "tab:purple", "Fuel reward (sum)",     "reward")
    plot_with_smooth(axes[1, 2], cb.episode_r_term_totals, "r_term",  "tab:brown",  "Terminal reward (sum)", "reward")
    plot_with_smooth(axes[2, 0], cb.episode_r_mile_totals, "r_mile",  "tab:pink",   "Milestone reward (sum)","reward")

    dock_rate = [
        np.mean(cb.episode_docked[max(0, i - DOCK_RATE_WINDOW):i + 1]) * 100
        for i in range(len(cb.episode_docked))
    ]
    axes[2, 1].plot(dock_rate, color="tab:cyan", linewidth=2)
    axes[2, 1].set_title(f"Dock rate ({DOCK_RATE_WINDOW}-ep rolling)")
    axes[2, 1].set_ylabel("%")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_ylim(0, 100)
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(cb.episode_curriculum_distances, color="tab:olive", linewidth=2)
    axes[2, 2].set_title("Curriculum distance (shared across all envs)")
    axes[2, 2].set_ylabel("m")
    axes[2, 2].set_xlabel("Episode")
    axes[2, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    diag_path = DIAGNOSTICS_PLOT_PATH.replace(".png", f"_{args.scenario}.png")
    fig.savefig(diag_path, dpi=150)
    plt.show()

    hist_path = TRAINING_HISTORY_PATH.replace(".npz", f"_{args.scenario}.npz")
    np.savez(
        hist_path,
        rewards    = np.array(cb.episode_rewards),
        steps      = np.array(cb.episode_steps),
        delta_v    = np.array(cb.episode_delta_vs),
        docked     = np.array(cb.episode_docked),
        r_pos      = np.array(cb.episode_r_pos_totals),
        r_fuel     = np.array(cb.episode_r_fuel_totals),
        r_term     = np.array(cb.episode_r_term_totals),
        r_mile     = np.array(cb.episode_r_mile_totals),
        curriculum_distance = np.array(cb.episode_curriculum_distances),
    )
    print("Done.")


if __name__ == "__main__":
    main()
