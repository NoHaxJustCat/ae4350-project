"""
train.py  –  TD3 via Stable-Baselines3, parallel envs, one model per scenario.

Usage:
    python -u training.py --scenario vbar
    python -u training.py --scenario rbar --n-envs 8 --total-timesteps 2000000
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import platform
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
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor

from libs.constants import (
    ACTION_NOISE_SIGMA_END,
    ACTION_NOISE_SIGMA_START,
    BATCH_SIZE,
    CRITIC_LR,
    DOCK_RATE_WINDOW,
    ENV_BURN_DEADZONE_FRAC,
    ENV_CURRICULUM_ENABLED,
    ENV_CURRICULUM_INCREMENT,
    ENV_CURRICULUM_MAX_DISTANCE,
    ENV_CURRICULUM_START_DISTANCE,
    ENV_MAX_DV_COEFF,
    GAMMA,
    LOG_EVERY,
    MIN_BUFFER,
    NOISE_DECAY_FRAC,
    NUM_ENVS,
    OMEGA,
    OU_DT,
    OU_STD_PER_SIGMA,
    OU_THETA,
    REPLAY_BUFFER_SIZE,
    SMOOTHING_WINDOW,
    TAU,
    TD3_TARGET_NOISE_CLIP,
    TD3_TARGET_POLICY_NOISE,
    TOTAL_TIMESTEPS,
    TRAINED_MODEL_DIR,
)
from libs.env import CWRendezvousEnv
from libs.normalization import NormalizedObsEnv
from libs.symmetry import CanonicalizeDirectionEnv
from libs.trajectory import plot_trajectory


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_single_env(scenario: str, curriculum_start_distance: float | None = None):
    def _init():
        # Runs inside each SubprocVecEnv worker. Each worker only ever does
        # a 4x4 matmul per step — letting it default to a multi-threaded
        # torch thread pool just means n_envs processes fighting each other
        # for the same cores. The main process (real gradient compute) sets
        # its own thread count separately in main().
        #
        # CAVEAT: with DummyVecEnv these thunks run in the MAIN process, so
        # this line silently overrides --torch-threads down to 1 for the
        # gradient updates too. That accident happened to WIN on the EPYC
        # box — 1 torch thread beat 4 by ~10% steps/s (348 vs 312), because
        # batch-256 [64,64] matmuls are too small for intra-op parallelism
        # to pay for its sync overhead.
        torch.set_num_threads(1)
        env_kwargs = {}
        if curriculum_start_distance is not None:
            env_kwargs["curriculum_start_distance"] = curriculum_start_distance
        env = CWRendezvousEnv(omega=OMEGA, scenario=scenario, **env_kwargs)
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


def build_diagnostics_figure(cb: "TrainingCallback", scenario: str):
    """Builds the 3x3 training-diagnostics figure. Used both for periodic
    live snapshots (LiveDiagnosticsCallback, during training) and the final
    post-training save — one implementation so the two can't silently
    diverge. Caller owns the returned figure (save/show/close it)."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"TD3 (SB3) Training Diagnostics — scenario={scenario}", fontsize=14)

    plot_with_smooth(axes[0, 0], cb.episode_rewards,       "reward",  "tab:blue",   "Total reward",          "reward")
    plot_with_smooth(axes[0, 1], cb.episode_dv_ratios,     "dv_used/dv_ref", "tab:orange", "Δv used vs. analytic reference", "dv_used / dv_ref")
    axes[0, 1].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="reference (ratio=1)")
    axes[0, 1].legend()
    plot_with_smooth(axes[0, 2], cb.episode_steps,         "steps",   "tab:green",  "Episode length",        "steps")
    plot_with_smooth(axes[1, 0], cb.episode_r_pos_totals,  "r_pos",   "tab:red",    "Position reward (sum)", "reward")
    plot_with_smooth(axes[1, 1], cb.episode_r_fuel_totals, "r_fuel",  "tab:purple", "Fuel reward (sum)",     "reward")
    plot_with_smooth(axes[1, 2], cb.episode_r_term_totals, "r_term",  "tab:brown",  "Terminal reward (sum)", "reward")

    axes[2, 0].plot(cb.episode_noise_std, color="tab:pink", linewidth=2, label="noise std")
    axes[2, 0].axhline(ENV_BURN_DEADZONE_FRAC, color="gray", linestyle="--", linewidth=1,
                        label=f"burn deadzone ({ENV_BURN_DEADZONE_FRAC})")
    axes[2, 0].set_title("Exploration noise (stationary std, native action units)")
    axes[2, 0].set_ylabel("std")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

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
    return fig


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
        # which holds one deep-copied ActionNoise instance per sub-env
        # (OrnsteinUhlenbeckActionNoise also exposes ._sigma, same as
        # NormalActionNoise, so this decay logic works unchanged).
        sub_noises = getattr(noise, "noises", [noise])
        for sub in sub_noises:
            sub._sigma[:] = sigma
        return True


class ThroughputCallback(BaseCallback):
    """
    Standalone steps/sec and ms/step log, decoupled from TrainingCallback's
    episode-based print — episode length is wildly variable (3 steps early,
    90+ once learning kicks in), so a per-episode timer is not a clean
    throughput signal. This fires on a fixed timestep cadence instead, so
    numbers are directly comparable before/after a perf change (thread
    limits, net_arch, vec-env backend, compile, train_freq, ...).
    """

    def __init__(self, log_every_timesteps: int = 2000):
        super().__init__(verbose=0)
        self.log_every_timesteps = log_every_timesteps
        self._last_t = None
        self._last_timesteps = 0

    def _on_training_start(self) -> None:
        self._last_t = time.perf_counter()
        self._last_timesteps = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_timesteps >= self.log_every_timesteps:
            now = time.perf_counter()
            dt = now - self._last_t
            dsteps = self.num_timesteps - self._last_timesteps
            steps_per_sec = dsteps / dt if dt > 0 else float("inf")
            ms_per_step = 1000.0 * dt / dsteps if dsteps > 0 else float("nan")
            print(f"[throughput] {self.num_timesteps:>10} timesteps | "
                  f"{steps_per_sec:>8.1f} steps/s | {ms_per_step:>6.2f} ms/step")
            self._last_t = now
            self._last_timesteps = self.num_timesteps
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

    Also writes a small <name>_<steps>_steps.curriculum.json sidecar with
    the curriculum_distance at save time. SB3's model.save() only persists
    the policy/optimizer/num_timesteps — curriculum_distance lives in
    CurriculumCallback, not the model, so --resume-from needs this sidecar
    to avoid silently restarting the curriculum from the beginning.
    """

    def __init__(self, save_dir: Path, name_prefix: str, n_envs: int,
                 save_freq_timesteps: int = 1000, keep_last: int = 20,
                 curriculum_callback: "CurriculumCallback | None" = None):
        super().__init__(verbose=0)
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.keep_last = keep_last
        self.save_freq_calls = max(save_freq_timesteps // n_envs, 1)
        self.curriculum_callback = curriculum_callback

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq_calls == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(str(path))

            if self.curriculum_callback is not None:
                sidecar = path.with_suffix(".curriculum.json")
                sidecar.write_text(json.dumps(
                    {"curriculum_distance": self.curriculum_callback.curriculum_distance}
                ))

            checkpoints = sorted(
                self.save_dir.glob(f"{self.name_prefix}_*_steps.zip"),
                key=lambda p: p.stat().st_mtime,
            )
            for old in checkpoints[:-self.keep_last]:
                old.unlink(missing_ok=True)
                old.with_suffix(".curriculum.json").unlink(missing_ok=True)
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

    Advance-only used to be a trap: with no way back down, a stage the
    current policy genuinely can't clear yet (dock rate stuck under
    threshold, e.g. the cur_d=35 stall) freezes curriculum_distance for the
    rest of the run. If `stall_patience` consecutive full windows come in
    below the dock-rate threshold, distance now steps back down by one
    increment instead, letting the policy consolidate at an easier stage.

    Keyed off the dock rate of each completed window, NOT "episodes since
    the last advance" — an earlier version used the latter and had a bug:
    once curriculum_distance reached max_distance, the advance branch's own
    guard (distance < max_distance) permanently blocked it from firing, so
    its counter reset never happened again and it force-regressed after 60
    episodes regardless of dock rate — including at a sustained 100% dock
    rate, since "stalled" and "already at the ceiling with nothing left to
    advance to" aren't the same thing.
    """

    def __init__(self, n_envs: int, increment: float, max_distance: float,
                 min_distance: float, window: int = 20, dock_rate_threshold: float = 0.5,
                 stall_patience: int = 3):
        super().__init__(verbose=0)
        self.n_envs = n_envs
        self.increment = increment
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.dock_rate_threshold = dock_rate_threshold
        self.stall_patience = stall_patience
        self._recent_docks = deque(maxlen=window)
        self.curriculum_distance = None
        self._stalled_windows = 0

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

        if len(self._recent_docks) < self._recent_docks.maxlen:
            return True

        dock_rate = np.mean(self._recent_docks)
        if dock_rate >= self.dock_rate_threshold:
            self._stalled_windows = 0
            if self.curriculum_distance < self.max_distance:
                self.curriculum_distance = min(self.curriculum_distance + self.increment, self.max_distance)
                self.training_env.env_method("set_curriculum_distance", self.curriculum_distance)
                print(f"[curriculum] dock rate >= {self.dock_rate_threshold:.0%} -> "
                      f"advancing to {self.curriculum_distance:.1f} m")
            self._recent_docks.clear()
        else:
            self._stalled_windows += 1
            if self._stalled_windows >= self.stall_patience and self.curriculum_distance > self.min_distance:
                self.curriculum_distance = max(self.curriculum_distance - self.increment, self.min_distance)
                self.training_env.env_method("set_curriculum_distance", self.curriculum_distance)
                print(f"[curriculum] dock rate < {self.dock_rate_threshold:.0%} for "
                      f"{self.stall_patience} windows -> regressing to {self.curriculum_distance:.1f} m")
                self._stalled_windows = 0
            self._recent_docks.clear()
        return True


class TrainingCallback(BaseCallback):
    """
    Per-step callback that accumulates episode diagnostics across N parallel
    vec envs. Each sub-env finishes episodes at its own pace, so per-episode
    accumulators are tracked per env index, not globally.

    Trajectories/actions are read from info["state"] / info["applied_action"]
    (raw physical values written by CWRendezvousEnv.step), NOT from
    new_obs — new_obs is normalized-to-[0,1] policy input and would corrupt
    the trajectory plots. Each episode's "states" list is seeded with the
    true pre-action reset() position (from vec_env.reset_infos) before any
    step()-produced states are appended, so trajectory plots/npz dumps start
    at the real starting point and the first Δv arrow is anchored at the
    position it was actually applied from, not the position one step later.
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
        self.episode_dv_ratios      = []
        self.episode_docked         = []
        self.episode_r_pos_totals   = []
        self.episode_r_fuel_totals  = []
        self.episode_r_term_totals  = []
        self.episode_noise_std      = []
        self.episode_curriculum_distances = []

        self._acc = [self._new_accumulator() for _ in range(n_envs)]
        self._episode_num = 0

        print(
            f"{'ep':>6} | {'steps':>5} | {'reward':>9} | {'r_pos':>8} | "
            f"{'r_fuel':>8} | {'r_term':>8} | {'noise':>8} | "
            f"{'dv/ref':>7} | {'cur_d':>6} | {'docked':>6} | {'cur_prog':>8} | {'ms/ep':>8}"
        )
        print("-" * 123)

    @staticmethod
    def _new_accumulator():
        return {
            "delta_v": 0.0, "r_pos": 0.0, "r_fuel": 0.0, "r_term": 0.0,
            "docked": False, "states": [], "actions": [],
            "start": time.perf_counter(),
        }

    def _current_noise_std(self) -> float:
        """Current OU stationary std in the env's native [-1,1] action
        units (see the OU_STD_PER_SIGMA comment in constants.py) — the
        NoiseDecayCallback writes the raw `sigma` parameter directly onto
        the noise object, so this reads it back and converts rather than
        recomputing the decay schedule a second time."""
        noise = self.model.action_noise
        if noise is None:
            return 0.0
        sub_noises = getattr(noise, "noises", [noise])
        return float(sub_noises[0]._sigma[0]) * OU_STD_PER_SIGMA

    def _seed_start_state(self, i: int) -> None:
        """Seeds a fresh per-env accumulator with the true pre-action
        position from that sub-env's most recent reset() (see
        CWRendezvousEnv.reset()'s "state" info key) — without this, states[0]
        is already the position AFTER the first action, so trajectory plots
        silently skip the real starting point and the first Δv arrow.
        Falls back to leaving "states" empty if reset_infos isn't populated
        yet (shouldn't happen once training has actually started)."""
        reset_info = self.training_env.reset_infos[i]
        start_state = reset_info.get("state")
        if start_state is not None:
            self._acc[i]["states"].append(start_state.copy())

    def _on_training_start(self) -> None:
        for i in range(self.n_envs):
            self._seed_start_state(i)

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
            if info.get("docked", False):
                acc["docked"] = True

            if dones[i]:
                ep_info = info.get("episode", {})
                reward  = ep_info.get("r", 0.0)
                steps   = ep_info.get("l", 1)

                cur_dist   = info.get("curriculum_distance", float("nan"))
                noise_std  = self._current_noise_std()
                dv_ref     = info.get("dv_ref", float("nan"))
                dv_ratio   = acc["delta_v"] / dv_ref if dv_ref else float("nan")

                self.episode_rewards.append(reward)
                self.episode_steps.append(steps)
                self.episode_delta_vs.append(acc["delta_v"])
                self.episode_dv_ratios.append(dv_ratio)
                self.episode_docked.append(acc["docked"])
                self.episode_r_pos_totals.append(acc["r_pos"])
                self.episode_r_fuel_totals.append(acc["r_fuel"])
                self.episode_r_term_totals.append(acc["r_term"])
                self.episode_noise_std.append(noise_std)
                self.episode_curriculum_distances.append(cur_dist)

                ep = self._episode_num
                if ep % self.log_every == 0:
                    elapsed   = time.perf_counter() - acc["start"]
                    dock_rate = np.mean(self.episode_docked[-DOCK_RATE_WINDOW:]) * 100
                    cur_prog  = self.curriculum_callback.progress if self.curriculum_callback else "n/a"
                    print(
                        f"{ep:>6} | {steps:>5} | {reward:>9.2f} | "
                        f"{acc['r_pos']:>8.2f} | {acc['r_fuel']:>8.2f} | "
                        f"{acc['r_term']:>8.2f} | {noise_std:>8.4f} | "
                        f"{dv_ratio:>7.2f} | {cur_dist:>6.1f} | {dock_rate:>5.1f}% | "
                        f"{cur_prog:>8} | {1000 * elapsed:>8.1f}"
                    )

                if i == 0 and (ep + 1) % self.log_every == 0:
                    tag       = f"ep_{ep + 1:04d}"
                    traj_path = self.tmp_dir / f"{tag}.png"
                    plot_trajectory(
                        acc["states"],
                        acc["actions"],
                        str(traj_path),
                        min_dv_display=ENV_MAX_DV_COEFF * 0.01,
                    )
                    shutil.copy(traj_path, self.tmp_dir / "latest_trajectory.png")
                    np.savez(
                        self.tmp_dir / f"{tag}.npz",
                        states=np.array(acc["states"]),
                        rewards=np.array([reward]),
                        steps=np.array([steps]),
                        delta_v=np.array([acc["delta_v"]]),
                        dv_ratio=np.array([dv_ratio]),
                        docked=np.array([acc["docked"]]),
                    )

                self._acc[i] = self._new_accumulator()
                self._seed_start_state(i)
                self._episode_num += 1

        return True


class LiveDiagnosticsCallback(BaseCallback):
    """
    Periodically regenerates the same 3x3 diagnostics figure that used to
    only get built once at the very end, so you can watch overall training
    progress (reward/dv/dock-rate/curriculum trends) while a run — or a
    whole sweep of them — is still going, instead of waiting for it to
    finish. Also writes a small status.json snapshot (used by
    remote_training.ps1's sweep dashboard to show one summary row per
    concurrent run instead of interleaved per-episode print lines).

    Time-throttled (default every 30s of wall clock), not every episode —
    rebuilding a smoothed multi-panel figure has a real, if small, per-call
    cost that would otherwise scale with both run length and how many
    concurrent sweep members are doing it at once.
    """

    def __init__(self, training_callback: "TrainingCallback", diag_path: str,
                 status_path: Path, scenario: str, run_tag: str,
                 total_timesteps: int, update_every_seconds: float = 30.0):
        super().__init__(verbose=0)
        self.training_callback = training_callback
        self.diag_path = diag_path
        self.status_path = status_path
        self.scenario = scenario
        self.run_tag = run_tag
        self.total_timesteps = total_timesteps
        self.update_every_seconds = update_every_seconds
        self._last_update = None
        self._start_time = None

    def _on_training_start(self) -> None:
        now = time.perf_counter()
        self._last_update = now
        self._start_time = now

    def write_status(self, now: float) -> None:
        cb = self.training_callback
        dock_window = cb.episode_docked[-DOCK_RATE_WINDOW:]
        reward_window = cb.episode_rewards[-SMOOTHING_WINDOW:]
        elapsed = now - self._start_time
        status = {
            "scenario": self.scenario,
            "run_tag": self.run_tag,
            "num_timesteps": int(self.num_timesteps),
            "total_timesteps": int(self.total_timesteps),
            "episode_count": len(cb.episode_rewards),
            "recent_dock_rate": float(np.mean(dock_window)) if dock_window else None,
            "recent_avg_reward": float(np.mean(reward_window)) if reward_window else None,
            "curriculum_distance": (float(cb.episode_curriculum_distances[-1])
                                     if cb.episode_curriculum_distances else None),
            "elapsed_seconds": elapsed,
            "steps_per_sec": (self.num_timesteps / elapsed) if elapsed > 0 else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.status_path.write_text(json.dumps(status))

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self._last_update < self.update_every_seconds:
            return True
        self._last_update = now

        if self.training_callback.episode_rewards:
            fig = build_diagnostics_figure(self.training_callback, self.scenario)
            fig.savefig(self.diag_path, dpi=100)
            plt.close(fig)

        self.write_status(now)
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["vbar", "rbar"], default="vbar")
    p.add_argument("--n-envs", type=int, default=NUM_ENVS)
    p.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--seed", type=int, default=None,
                    help="Seeds TD3's own RNG (policy init, replay buffer sampling, target "
                         "noise). Independent of exploration noise/episode-init randomness, "
                         "which draw from each sub-env's own np_random. Use to run "
                         "reproducible parallel seed sweeps.")
    p.add_argument("--run-tag", default="",
                    help="Names this run's own subfolder under trained/<session-id>/ (and "
                         "tmp/) — e.g. 'utd2_seed5' — so multiple training.py processes "
                         "launched concurrently on the same machine (a hyperparameter/seed "
                         "sweep) each get their own checkpoints/model/diagnostics/history "
                         "instead of clobbering each other's. Empty (default) means this is "
                         "the only run in its session, so no extra subfolder is needed.")
    p.add_argument("--session-id", default="",
                    help="Groups related runs under trained/<session-id>/ (e.g. all members "
                         "of one --run-tag sweep launched by remote_training.ps1 share one, "
                         "passed in for you). Empty (default) auto-generates a timestamp, so "
                         "even a plain local run gets its own dated folder instead of "
                         "overwriting the previous run's checkpoints/model/diagnostics/history.")
    p.add_argument("--device", default="cpu",
                    help="'cpu' is usually faster than 'cuda' for a network this small.")
    p.add_argument("--checkpoint-freq", type=int, default=100000,
                    help="Save the model every N environment timesteps.")
    p.add_argument("--keep-last-checkpoints", type=int, default=5,
                    help="Only keep the N most recent checkpoints on disk.")
    p.add_argument("--resume-from", default=None,
                    help="Path to a saved checkpoint .zip (e.g. "
                         "trained/20260711_143022/checkpoints/vbar_td3_2299264_steps.zip) "
                         "to continue "
                         "training from instead of starting fresh. --total-timesteps is "
                         "still the ORIGINAL full target (e.g. 3000000), not a remaining "
                         "amount — the run continues until the model's num_timesteps "
                         "reaches it. Looks for a sidecar <path-without-.zip>.curriculum.json "
                         "next to the checkpoint to resume curriculum_distance too (written "
                         "automatically by PeriodicCheckpointCallback); if missing, curriculum "
                         "restarts at ENV_CURRICULUM_START_DISTANCE. The replay buffer is NOT "
                         "restored (not saved by the checkpoint callback), so gradient updates "
                         "pause again until it refills past MIN_BUFFER — network weights carry "
                         "over regardless.")

    # ── Perf knobs ──────────────────────────────────────────────────────────
    p.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto",
                    help="'auto' = subproc if n-envs>1 else dummy. Force one to A/B test "
                         "IPC overhead vs parallelism on an env this cheap to step. "
                         "dummy measurably wins on this workload (see remote_training.ps1) "
                         "since steady-state training is bottlenecked on the gradient loop, "
                         "not env stepping — a batched (single-numpy-op) vec-env was tried "
                         "and removed after confirming zero steady-state benefit for the "
                         "real maintenance cost of a second, hand-duplicated reward/physics "
                         "implementation.")
    p.add_argument("--vec-env-start-method",
                    default="fork" if platform.system() != "Windows" else "spawn",
                    choices=["fork", "spawn", "forkserver"],
                    help="'fork' (Linux/RHEL default here) skips re-importing torch/numpy "
                         "in every worker; Windows only supports 'spawn'.")
    p.add_argument("--net-arch", default="64,64",
                    help="Comma-separated hidden layer sizes, e.g. '64,64'. Was [400,300] "
                         "(from the original DDPG paper's much higher-dim tasks) — "
                         "oversized for a 5-obs/2-action problem. With --arch smart these "
                         "size the actor/critic HEADS that sit on top of the residual encoder.")
    p.add_argument("--lr", type=float, default=CRITIC_LR,
                    help="Learning rate for actor+critic (default from constants.py). Lower "
                         "(1e-4/3e-5) stabilizes a wide/deep critic; higher trains faster.")
    p.add_argument("--gamma", type=float, default=GAMMA,
                    help="Discount factor override (default from constants.py). Big nets "
                         "destabilize under the fuel-tuned 0.9999 (critic value divergence -> "
                         "actor saturates at the action bound, never learns); drop to ~0.9995 "
                         "to train a wide/deep net, at the cost of the fuel-optimal long-coast "
                         "incentive 0.9999 was chosen for.")
    p.add_argument("--arch", choices=["mlp", "smart"], default="mlp",
                    help="'mlp' = SB3's flat MlpPolicy (default, matches all prior runs). "
                         "'smart' = LayerNorm residual encoder (libs/policies.py) in front of "
                         "the actor/critic heads — added to escape the fuel-wasteful local "
                         "optimum the flat [128,128] nets converge to. Meant for --device cuda.")
    p.add_argument("--features-dim", type=int, default=256,
                    help="(--arch smart only) width of every LayerNorm-MLP encoder layer.")
    p.add_argument("--n-blocks", type=int, default=2,
                    help="(--arch smart only) number of Linear->LayerNorm->act encoder layers. "
                         "Empirically 2 is the deepest that trains cleanly with relu (3 saturates "
                         "the actor at init and never learns); silu tolerates only 1. See the A/B "
                         "isolation results — depth beyond this pins the actor to the boundary.")
    p.add_argument("--activation", choices=["silu", "relu", "gelu", "tanh"], default="relu",
                    help="(--arch smart only) hidden activation for encoder + heads. relu is more "
                         "robust to the init-saturation failure than silu at depth (silu fails at "
                         "n_blocks=2, relu survives to 2).")
    p.add_argument("--torch-threads", type=int, default=min(4, os.cpu_count() or 4),
                    help="Threads for the MAIN process's torch ops (gradient updates). "
                         "Subprocess workers are always pinned to 1 (see make_single_env).")
    p.add_argument("--compile", action="store_true",
                    help="Wrap actor/critic nets in torch.compile(). Same fixed batch "
                         "shape called millions of times -> real candidate, but more "
                         "mature on Linux than Windows; falls back gracefully if it errors.")
    p.add_argument("--train-freq", type=int, default=1,
                    help="Collect this many env-steps between training phases.")
    p.add_argument("--gradient-steps", type=int, default=-1,
                    help="Gradient steps per training phase. -1 = match --train-freq "
                         "(current 1:1 ratio). Lower e.g. --train-freq 4 --gradient-steps 1 "
                         "for a 0.25 update-to-data ratio — env steps are ~free here, "
                         "gradient steps are the expensive part.")
    p.add_argument("--throughput-log-every", type=int, default=2000,
                    help="Print steps/sec every N timesteps.")
    p.add_argument("--diag-update-every-seconds", type=float, default=30.0,
                    help="How often (wall-clock seconds) to refresh the live diagnostics.png "
                         "and status.json while training. Rebuilding the 3x3 figure has real "
                         "cost on this tiny a network — measured 859->31.6 steps/s (27x "
                         "slower) at a 1s interval on net_arch=[64,64]. Keep this at 30s+ "
                         "for real training; only lower it for a quick local smoke test.")
    return p.parse_args()


def main():
    args = parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    print(f"Platform: {platform.system()} | torch main-process threads: {args.torch_threads} "
          f"| cpu_count: {os.cpu_count()}")

    t0 = time.perf_counter()
    check_env(CWRendezvousEnv(omega=OMEGA, scenario=args.scenario))
    print(f"Environment check completed in {time.perf_counter() - t0:.2f}s")

    # Every run gets its own directory instead of flat trained/checkpoints/,
    # so re-running (or a --run-tag sweep) never overwrites a previous run's
    # checkpoints/model/diagnostics/history. session_id groups related runs
    # together (e.g. all N members of one sweep launched by
    # remote_training.ps1 share one, passed in via --session-id); it's
    # auto-generated from the wall-clock time if not given, so a plain local
    # `python training.py` still gets a fresh dated folder every time
    # instead of clobbering the previous run's outputs.
    session_id = args.session_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(TRAINED_MODEL_DIR) / session_id / args.run_tag
               if args.run_tag else Path(TRAINED_MODEL_DIR) / session_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"run_dir: {run_dir}")

    model_path  = str(run_dir / f"{args.scenario}_td3")
    diag_path   = str(run_dir / "diagnostics.png")
    hist_path   = str(run_dir / "history.npz")
    status_path = run_dir / "status.json"

    tmp_dir = Path("tmp") / args.run_tag if args.run_tag else Path("tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    resume_curriculum_distance = None
    if args.resume_from:
        sidecar = Path(args.resume_from).with_suffix("").with_suffix(".curriculum.json")
        if sidecar.exists():
            resume_curriculum_distance = json.loads(sidecar.read_text())["curriculum_distance"]
            print(f"Resuming curriculum_distance = {resume_curriculum_distance:.1f} m (from {sidecar})")
        else:
            print(f"WARNING: no curriculum sidecar at {sidecar} — curriculum will "
                  f"restart at {ENV_CURRICULUM_START_DISTANCE:.1f} m")

    n_envs = max(1, args.n_envs)
    if args.vec_env == "dummy":
        vec_cls, vec_kwargs = DummyVecEnv, {}
    elif args.vec_env == "subproc":
        vec_cls, vec_kwargs = SubprocVecEnv, dict(start_method=args.vec_env_start_method)
    else:  # auto
        if n_envs > 1:
            vec_cls, vec_kwargs = SubprocVecEnv, dict(start_method=args.vec_env_start_method)
        else:
            vec_cls, vec_kwargs = DummyVecEnv, {}
    print(f"vec_env: {vec_cls.__name__}"
          + (f" (start_method={vec_kwargs['start_method']})" if vec_kwargs else ""))
    env = make_vec_env(
        make_single_env(args.scenario, curriculum_start_distance=resume_curriculum_distance),
        n_envs=n_envs, vec_env_cls=vec_cls, vec_env_kwargs=vec_kwargs,
    )

    net_arch = [int(x) for x in args.net_arch.split(",")]
    print(f"net_arch: {net_arch} | train_freq: {args.train_freq} | "
          f"gradient_steps: {args.gradient_steps}")

    if args.resume_from:
        model = TD3.load(args.resume_from, env=env, device=args.device)
        print(f"Resumed model from {args.resume_from} at {model.num_timesteps} timesteps "
              f"({args.total_timesteps - model.num_timesteps} remaining toward "
              f"--total-timesteps {args.total_timesteps})")
    else:
        action_dim = env.action_space.shape[0]
        # OU (not i.i.d. Gaussian): exploration is a temporally-correlated
        # random walk, so a sustained multi-step push OR a long lingering
        # stretch near zero is a plausible exploration outcome instead of
        # vanishingly unlikely. The true-optimal V-bar maneuver needs ~230+
        # consecutive steps of "coasting" (near-zero action) before it pays
        # off — i.i.d. per-step noise can't produce that by chance, since
        # each step is independent of the last; a correlated walk can.
        # ACTION_NOISE_SIGMA_START/END are pre-corrected in constants.py for
        # OU's sigma->stationary-std amplification (see the comment there) —
        # theta/dt must match what that correction assumed.
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(action_dim),
            sigma=ACTION_NOISE_SIGMA_START * np.ones(action_dim),
            theta=OU_THETA,
            dt=OU_DT,
        )
        if args.arch == "smart":
            from libs.policies import build_smart_policy_kwargs
            policy_kwargs = build_smart_policy_kwargs(
                net_arch=net_arch,
                features_dim=args.features_dim,
                n_blocks=args.n_blocks,
                activation=args.activation,
            )
            print(f"arch: smart | features_dim: {args.features_dim} | "
                  f"n_blocks: {args.n_blocks} | activation: {args.activation} "
                  f"| head net_arch: {net_arch}")
        else:
            policy_kwargs = dict(net_arch=net_arch)
            print(f"arch: mlp | net_arch: {net_arch}")
        model = TD3(
            policy               = "MlpPolicy",
            env                  = env,
            learning_rate        = args.lr,
            buffer_size          = REPLAY_BUFFER_SIZE,
            learning_starts      = MIN_BUFFER,
            batch_size           = BATCH_SIZE,
            tau                  = TAU,
            gamma                = args.gamma,
            train_freq           = (args.train_freq, "step"),
            gradient_steps       = args.gradient_steps,
            action_noise         = action_noise,
            policy_delay         = 2,
            target_policy_noise  = TD3_TARGET_POLICY_NOISE,
            target_noise_clip    = TD3_TARGET_NOISE_CLIP,
            policy_kwargs        = policy_kwargs,
            verbose              = 0,
            device               = args.device,
            seed                 = args.seed,
        )

        # Small-output actor init (DDPG/TD3 fan-in trick). Required for the
        # wide/deep smart nets: without it a big net saturates the actor at
        # ±max_dv from step one and never learns (see
        # libs/policies.py::shrink_actor_output_init). Harmless for small
        # nets, so applied to every --arch smart model.
        if args.arch == "smart":
            from libs.policies import shrink_actor_output_init
            shrink_actor_output_init(model)

    if args.compile:
        try:
            model.actor = torch.compile(model.actor)
            model.actor_target = torch.compile(model.actor_target)
            model.critic = torch.compile(model.critic)
            model.critic_target = torch.compile(model.critic_target)
            print("torch.compile: enabled on actor/critic + targets")
        except Exception as e:
            print(f"torch.compile: FAILED to enable, continuing without it ({e})")

    curriculum_callback = CurriculumCallback(
        n_envs=n_envs,
        increment=ENV_CURRICULUM_INCREMENT,
        max_distance=ENV_CURRICULUM_MAX_DISTANCE,
        min_distance=ENV_CURRICULUM_START_DISTANCE,
    ) if ENV_CURRICULUM_ENABLED else None

    callback = [
        TrainingCallback(tmp_dir=tmp_dir, n_envs=n_envs, curriculum_callback=curriculum_callback),
        NoiseDecayCallback(
            total_timesteps=args.total_timesteps,
            sigma_start=ACTION_NOISE_SIGMA_START,
            sigma_end=ACTION_NOISE_SIGMA_END,
        ),
        PeriodicCheckpointCallback(
            save_dir=run_dir / "checkpoints",
            name_prefix=f"{args.scenario}_td3",
            n_envs=n_envs,
            save_freq_timesteps=args.checkpoint_freq,
            keep_last=args.keep_last_checkpoints,
            curriculum_callback=curriculum_callback,
        ),
        ThroughputCallback(log_every_timesteps=args.throughput_log_every),
    ]
    live_diag_callback = LiveDiagnosticsCallback(
        training_callback=callback[0],
        diag_path=diag_path,
        status_path=status_path,
        scenario=args.scenario,
        run_tag=args.run_tag,
        total_timesteps=args.total_timesteps,
        update_every_seconds=args.diag_update_every_seconds,
    )
    callback.append(live_diag_callback)
    if curriculum_callback is not None:
        callback.append(curriculum_callback)

    # SB3's learn(total_timesteps, reset_num_timesteps=False) treats
    # total_timesteps as an INCREMENT added to the model's existing
    # num_timesteps, not an absolute target — so on resume we pass the
    # remaining budget, not args.total_timesteps itself, to keep
    # --total-timesteps meaning "the original full target" everywhere else
    # (NoiseDecayCallback's decay schedule in particular needs the ORIGINAL
    # total, not the remaining one, since it compares against the model's
    # absolute num_timesteps).
    if args.resume_from:
        remaining_timesteps = max(args.total_timesteps - model.num_timesteps, 0)
    else:
        remaining_timesteps = args.total_timesteps

    model.learn(
        total_timesteps      = remaining_timesteps,
        callback              = callback,
        progress_bar          = False,
        reset_num_timesteps   = not bool(args.resume_from),
    )

    model.save(model_path)
    print(f"Model saved -> {model_path}.zip")

    # ── Diagnostics plot ──────────────────────────────────────────────────────
    cb = callback[0]
    fig = build_diagnostics_figure(cb, args.scenario)
    fig.savefig(diag_path, dpi=150)
    plt.show()

    # Guaranteed final status.json write — the periodic one in
    # LiveDiagnosticsCallback is time-throttled, so a run that finishes
    # between updates (or is shorter than the interval) would otherwise
    # never get one, and the dashboard would show it as stuck/missing.
    live_diag_callback.write_status(time.perf_counter())

    np.savez(
        hist_path,
        rewards    = np.array(cb.episode_rewards),
        steps      = np.array(cb.episode_steps),
        delta_v    = np.array(cb.episode_delta_vs),
        dv_ratio   = np.array(cb.episode_dv_ratios),
        docked     = np.array(cb.episode_docked),
        r_pos      = np.array(cb.episode_r_pos_totals),
        r_fuel     = np.array(cb.episode_r_fuel_totals),
        r_term     = np.array(cb.episode_r_term_totals),
        noise_std  = np.array(cb.episode_noise_std),
        curriculum_distance = np.array(cb.episode_curriculum_distances),
    )
    print("Done.")


if __name__ == "__main__":
    main()
