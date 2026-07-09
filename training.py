"""
train.py  –  DDPG via Stable-Baselines3
"""

from pathlib import Path
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor

from libs.constants import (
    ACTOR_LR,
    CRITIC_LR,
    DIAGNOSTICS_PLOT_PATH,
    DOCK_RATE_WINDOW,
    ENV_MAX_DV,
    GAMMA,
    BATCH_SIZE,
    LOG_EVERY,
    MAX_STEPS,
    MIN_BUFFER,
    NUM_EPISODES,
    OMEGA,
    REPLAY_BUFFER_SIZE,
    SMOOTHING_WINDOW,
    TAU,
    TRAINED_ACTOR_PATH,
    TRAINED_CRITIC_PATH,
    TRAINING_HISTORY_PATH,
)
from libs.env import CWRendezvousEnv
from libs.trajectory import plot_trajectory


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_env() -> Monitor:
    env = CWRendezvousEnv(omega=OMEGA)
    return Monitor(env)


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


# ── Callback ──────────────────────────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    """
    Per-step callback that accumulates episode diagnostics.

    Key design decisions vs the previous version:
    - States are buffered from `new_obs` (post-step obs), NOT from
      raw_env.state, because the env has already reset() by the time
      done=True fires, so raw_env.state would be the NEXT episode's
      initial state (the "goes back to square one" trajectory bug).
    - train_freq=1 (every step) + gradient_steps=1 is used instead of
      (1, "episode") / -1, which caused SB3 to do 0 updates per episode
      when episode length < expected, freezing the policy.
    """

    def __init__(self, tmp_dir: Path, log_every: int = LOG_EVERY):
        super().__init__(verbose=0)
        self.tmp_dir   = tmp_dir
        self.log_every = log_every

        # per-episode history lists
        self.episode_rewards        = []
        self.episode_steps          = []
        self.episode_delta_vs       = []
        self.episode_docked         = []
        self.episode_r_pos_totals   = []
        self.episode_r_fuel_totals  = []
        self.episode_r_term_totals  = []
        self.episode_r_mile_totals  = []

        self._reset_accumulators()
        self._episode_num = 0

        print(
            f"{'ep':>6} | {'steps':>5} | {'reward':>9} | {'r_pos':>8} | "
            f"{'r_fuel':>8} | {'r_term':>8} | {'r_mile':>8} | "
            f"{'dv':>7} | {'docked':>6} | {'ms/ep':>8}"
        )
        print("-" * 105)

    def _reset_accumulators(self):
        self._ep_delta_v  = 0.0
        self._ep_r_pos    = 0.0
        self._ep_r_fuel   = 0.0
        self._ep_r_term   = 0.0
        self._ep_r_mile   = 0.0
        self._ep_docked   = False
        # Store (obs_before_step, action) pairs to reconstruct trajectory.
        # We capture obs BEFORE the step (self.locals["obs_tensor"]) so
        # the final state in the buffer is the last real state, not the
        # reset state.
        self._ep_states   = []
        self._ep_actions  = []
        self._ep_start    = time.perf_counter()

    def _on_step(self) -> bool:
        info   = self.locals["infos"][0]
        done   = self.locals["dones"][0]
        action = self.locals["actions"][0]

        # Capture the observation BEFORE this step produced `done`.
        # SB3 stores it as "obs_tensor" (the input to the actor this step).
        obs = self.locals["new_obs"][0]
        self._ep_states.append(obs.copy())

        if not info.get("budget_exhausted", False):
            self._ep_actions.append(info.get("applied_action", action).copy())
        else:
            self._ep_actions.append(np.zeros_like(action))

        self._ep_delta_v += info.get("delta_v", 0.0)
        self._ep_r_pos   += info.get("reward_pos", 0.0)
        self._ep_r_fuel  += info.get("reward_fuel", 0.0)
        self._ep_r_term  += info.get("reward_terminal", 0.0)
        self._ep_r_mile  += info.get("reward_milestone", 0.0)
        if info.get("docked", False):
            self._ep_docked = True

        if done:
            # Monitor injects cumulative episode stats under info["episode"]
            ep_info = info.get("episode", {})
            reward  = ep_info.get("r", 0.0)
            steps   = ep_info.get("l", 1)

            self.episode_rewards.append(reward)
            self.episode_steps.append(steps)
            self.episode_delta_vs.append(self._ep_delta_v)
            self.episode_docked.append(self._ep_docked)
            self.episode_r_pos_totals.append(self._ep_r_pos)
            self.episode_r_fuel_totals.append(self._ep_r_fuel)
            self.episode_r_term_totals.append(self._ep_r_term)
            self.episode_r_mile_totals.append(self._ep_r_mile)

            ep = self._episode_num
            if ep % self.log_every == 0:
                elapsed   = time.perf_counter() - self._ep_start
                dock_rate = np.mean(self.episode_docked[-DOCK_RATE_WINDOW:]) * 100
                print(
                    f"{ep:>6} | {steps:>5} | {reward:>9.2f} | "
                    f"{self._ep_r_pos:>8.2f} | {self._ep_r_fuel:>8.2f} | "
                    f"{self._ep_r_term:>8.2f} | {self._ep_r_mile:>8.2f} | "
                    f"{self._ep_delta_v:>7.4f} | {dock_rate:>5.1f}% | "
                    f"{1000 * elapsed:>8.1f}"
                )

            if (ep + 1) % self.log_every == 0:
                tag       = f"ep_{ep + 1:04d}"
                traj_path = self.tmp_dir / f"{tag}.png"
                # States are physical obs (pos+vel+dv_rem); trajectory plotter
                # expects the raw state vectors — pass them as-is.
                plot_trajectory(
                    self._ep_states,
                    self._ep_actions,
                    str(traj_path),
                    min_dv_display=ENV_MAX_DV * 0.01,
                )
                shutil.copy(traj_path, self.tmp_dir / "latest_trajectory.png")
                np.savez(
                    self.tmp_dir / f"{tag}.npz",
                    states=np.array(self._ep_states),
                    rewards=np.array([reward]),
                    steps=np.array([steps]),
                    delta_v=np.array([self._ep_delta_v]),
                    docked=np.array([self._ep_docked]),
                )

            self._reset_accumulators()
            self._episode_num += 1

        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    check_env(CWRendezvousEnv(omega=OMEGA))
    print(f"Environment check completed in {time.perf_counter() - t0:.2f}s")

    Path("trained").mkdir(parents=True, exist_ok=True)
    Path("out").mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = build_env()

    # OUNoise: sigma = 30% of max_dv — generous exploration for a tiny action
    # space (max_dv = 0.025 m/s). Feel free to tune sigma down as training
    # matures, e.g. with a decaying schedule.
    action_dim = env.action_space.shape[0]
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(action_dim),
        sigma=0.30 * ENV_MAX_DV * np.ones(action_dim),
    )

    policy_kwargs = dict(net_arch=[400, 300])

    # ── DDPG model ────────────────────────────────────────────────────────────
    # train_freq=1 + gradient_steps=1: one gradient update per env step.
    # This is the standard DDPG regime and avoids the "0 updates per episode"
    # failure mode of train_freq=(1,"episode") + gradient_steps=-1.
    model = DDPG(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = CRITIC_LR,        # shared actor+critic LR in SB3 DDPG
        buffer_size     = REPLAY_BUFFER_SIZE,
        learning_starts = MIN_BUFFER,       # matches your old MIN_BUFFER gate
        batch_size      = BATCH_SIZE,
        tau             = TAU,
        gamma           = GAMMA,
        train_freq      = 1,                # update every step  ← key fix
        gradient_steps  = 1,
        action_noise    = ou_noise,
        policy_kwargs   = policy_kwargs,
        verbose         = 0,
        device          = "auto",
    )

    total_timesteps = NUM_EPISODES * MAX_STEPS
    callback = TrainingCallback(tmp_dir=tmp_dir)

    model.learn(
        total_timesteps = total_timesteps,
        callback        = callback,
        progress_bar    = False,
    )

    model.save("trained/sb3_ddpg")
    torch.save(model.actor.state_dict(),  TRAINED_ACTOR_PATH)
    torch.save(model.critic.state_dict(), TRAINED_CRITIC_PATH)
    print(f"Weights saved → {TRAINED_ACTOR_PATH}, {TRAINED_CRITIC_PATH}")

    # ── Diagnostics plot ──────────────────────────────────────────────────────
    cb  = callback
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("DDPG (SB3) Training Diagnostics", fontsize=14)

    plot_with_smooth(axes[0, 0], cb.episode_rewards,       "reward",  "tab:blue",   "Total reward",          "reward")
    plot_with_smooth(axes[0, 1], cb.episode_delta_vs,      "delta-v", "tab:orange", "Total Δv used",         "m/s")
    plot_with_smooth(axes[0, 2], cb.episode_steps,         "steps",   "tab:green",  "Episode length",        "steps")
    plot_with_smooth(axes[0, 3], cb.episode_r_pos_totals,  "r_pos",   "tab:red",    "Position reward (sum)", "reward")
    plot_with_smooth(axes[1, 0], cb.episode_r_fuel_totals, "r_fuel",  "tab:purple", "Fuel reward (sum)",     "reward")
    plot_with_smooth(axes[1, 1], cb.episode_r_term_totals, "r_term",  "tab:brown",  "Terminal reward (sum)", "reward")
    plot_with_smooth(axes[1, 2], cb.episode_r_mile_totals, "r_mile",  "tab:pink",   "Milestone reward (sum)","reward")

    dock_rate = [
        np.mean(cb.episode_docked[max(0, i - DOCK_RATE_WINDOW):i + 1]) * 100
        for i in range(len(cb.episode_docked))
    ]
    axes[1, 3].plot(dock_rate, color="tab:cyan", linewidth=2)
    axes[1, 3].set_title(f"Dock rate ({DOCK_RATE_WINDOW}-ep rolling)")
    axes[1, 3].set_ylabel("%")
    axes[1, 3].set_xlabel("Episode")
    axes[1, 3].set_ylim(0, 100)
    axes[1, 3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(DIAGNOSTICS_PLOT_PATH, dpi=150)
    plt.show()

    np.savez(
        TRAINING_HISTORY_PATH,
        rewards    = np.array(cb.episode_rewards),
        steps      = np.array(cb.episode_steps),
        delta_v    = np.array(cb.episode_delta_vs),
        docked     = np.array(cb.episode_docked),
        r_pos      = np.array(cb.episode_r_pos_totals),
        r_fuel     = np.array(cb.episode_r_fuel_totals),
        r_term     = np.array(cb.episode_r_term_totals),
        r_mile     = np.array(cb.episode_r_mile_totals),
    )
    print("Done.")


if __name__ == "__main__":
    main()