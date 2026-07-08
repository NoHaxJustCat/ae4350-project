from copy import deepcopy
from collections import deque
from pathlib import Path
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from gymnasium.utils.env_checker import check_env

from libs.actor import Actor
from libs.constants import (
    ACTOR_LR,
    CRITIC_LR,
    DIAGNOSTICS_PLOT_PATH,
    DOCK_RATE_WINDOW,
    ENV_MAX_DV,
    GAMMA,
    GRAD_CLIP_NORM,
    BATCH_SIZE,
    LOG_EVERY,
    MAX_STEPS,
    MIN_BUFFER,
    NUM_EPISODES,
    OMEGA,
    REPLAY_BUFFER_SIZE,
    TAU,
    SMOOTHING_WINDOW,
    TRAINED_ACTOR_PATH,
    TRAINED_CRITIC_PATH,
    TRAINING_HISTORY_PATH,
)
from libs.critic import Critic
from libs.env import CWRendezvousEnv
from libs.normalization import normalize_action, normalize_state
from libs.trajectory import plot_trajectory

# ── Feature flags ────────────────────────────────────────────────────────────
USE_REPLAY_BUFFER  = False
USE_ACTOR_TARGET   = False
USE_CRITIC_TARGET  = False
# ─────────────────────────────────────────────────────────────────────────────


def set_requires_grad(module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def build_env() -> CWRendezvousEnv:
    return CWRendezvousEnv(omega=OMEGA)


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


def main():
    check_start = time.perf_counter()
    check_env(build_env())
    check_seconds = time.perf_counter() - check_start
    print(f"Environment check completed in {check_seconds:.2f}s")
    print(f"Flags: USE_REPLAY_BUFFER={USE_REPLAY_BUFFER}  "
          f"USE_ACTOR_TARGET={USE_ACTOR_TARGET}  "
          f"USE_CRITIC_TARGET={USE_CRITIC_TARGET}")

    env = build_env()

    actor  = Actor(max_action=env.max_dv)
    critic = Critic()

    critic_target = deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    actor_target = deepcopy(actor)
    for p in actor_target.parameters():
        p.requires_grad_(False)

    actor_opt  = optim.Adam(actor.parameters(),  lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    buffer = deque(maxlen=REPLAY_BUFFER_SIZE) if USE_REPLAY_BUFFER else None

    Path("trained").mkdir(parents=True, exist_ok=True)
    Path("out").mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards       = []
    episode_steps         = []
    episode_delta_vs      = []
    episode_critic_losses = []
    episode_actor_losses  = []
    episode_docked        = []
    episode_r_pos_totals  = []
    episode_r_fuel_totals = []

    print(f"{'ep':>6} | {'steps':>5} | {'reward':>9} | {'r_pos':>8} | {'r_fuel':>8} | {'r_term':>8} | {'r_mile':>8} | "
      f"{'dv':>7} | {'c_loss':>9} | {'a_loss':>9} | {'J_mean':>8} | {'docked':>6} | "
      f"{'ms/step':>8}")
    print("-" * 130) 

    for episode in range(NUM_EPISODES):
        episode_start = time.perf_counter()

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        state_nn = normalize_state(state)
        episode_states  = [state.detach().numpy().copy()]
        episode_actions = []

        total_reward      = 0.0
        total_delta_v     = 0.0
        total_r_pos       = 0.0
        total_r_fuel      = 0.0
        total_r_terminal  = 0.0
        total_r_milestone = 0.0 
        total_critic_loss = 0.0
        total_actor_loss  = 0.0
        total_J           = 0.0
        update_count      = 0
        docked            = False

        for step in range(MAX_STEPS):
            if USE_REPLAY_BUFFER and buffer is not None and len(buffer) < MIN_BUFFER:
                action = torch.tensor(env.action_space.sample() * 0.1, dtype=torch.float32)
            else:
                # noise_scale = max(0.0, 0.1 * (1 - episode / NUM_EPISODES / 20))
                action = actor(state_nn).detach()
                # action += torch.tensor(np.random.normal(0, noise_scale, size=action.shape), dtype=torch.float32)
                # action = action.clamp(-env.max_dv, env.max_dv)

            next_state, reward, terminated, truncated, info = env.step(action.numpy())
            next_state    = torch.tensor(next_state, dtype=torch.float32)
            next_state_nn = normalize_state(next_state)
            episode_states.append(next_state.detach().numpy().copy())
            reward_t = torch.tensor([reward], dtype=torch.float32)

            if env.curriculum_advanced:
                # buffer.clear()
                env.curriculum_advanced = False
                print(f"  → curriculum advanced to {env.curriculum_distance:.1f}m")

            applied_action = torch.tensor(info["applied_action"], dtype=torch.float32)

            # Only record burns that actually happened — zero-action transitions
            # (budget exhausted) are meaningless for the plot and add visual noise.
            if not info["budget_exhausted"]:
                episode_actions.append(info["applied_action"].copy())
            else:
                episode_actions.append(np.zeros_like(info["applied_action"]))

            total_delta_v += info["delta_v"]
            total_reward  += reward
            total_r_pos   += info["reward_pos"]
            total_r_fuel  += info["reward_fuel"]
            total_r_terminal  += info["reward_terminal"]
            total_r_milestone += info["reward_milestone"] 
            if info["docked"]:
                docked = True

            # ── Replay buffer ─────────────────────────────────────────────
            # Skip budget-exhausted transitions: the action is always zero
            # regardless of what the actor commanded, so these transitions
            # carry no policy-relevant information and only dilute the buffer.
            if USE_REPLAY_BUFFER:
                if not info["budget_exhausted"]:
                    buffer.append((
                        state.numpy().copy(),
                        applied_action.numpy().copy(),
                        reward,
                        next_state.numpy().copy(),
                        terminated or truncated,
                    ))
                ready = len(buffer) >= MIN_BUFFER
            else:
                ready = True

            # ── Sample a batch ────────────────────────────────────────────
            if ready:
                if USE_REPLAY_BUFFER:
                    batch = random.sample(buffer, BATCH_SIZE)
                    s, a, r, s2, done = zip(*batch)
                    s    = torch.tensor(np.array(s),    dtype=torch.float32)
                    a    = torch.tensor(np.array(a),    dtype=torch.float32)
                    r    = torch.tensor(np.array(r),    dtype=torch.float32).unsqueeze(1)
                    s2   = torch.tensor(np.array(s2),   dtype=torch.float32)
                    done = torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1)
                    s_nn  = normalize_state(s)
                    a_nn  = normalize_action(a)
                    s2_nn = normalize_state(s2)
                else:
                    s    = state.unsqueeze(0)
                    a    = applied_action.unsqueeze(0)
                    r    = reward_t.unsqueeze(0)
                    s2   = next_state.unsqueeze(0)
                    done = torch.tensor([[terminated or truncated]], dtype=torch.float32)
                    s_nn  = state_nn.unsqueeze(0)
                    a_nn  = normalize_action(a)
                    s2_nn = next_state_nn.unsqueeze(0)

                # ── Critic update ─────────────────────────────────────────
                with torch.no_grad():
                    a2_next = actor_target(s2_nn) if USE_ACTOR_TARGET else actor(s2_nn)
                    q_next  = (
                        critic_target(s2_nn, normalize_action(a2_next))
                        if USE_CRITIC_TARGET
                        else critic(s2_nn, normalize_action(a2_next))
                    )
                    target = r + GAMMA * (1 - done) * q_next

                critic_loss = torch.mean((critic(s_nn, a_nn) - target) ** 2)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP_NORM)
                critic_opt.step()

                # ── Soft-update targets ───────────────────────────────────
                with torch.no_grad():
                    if USE_CRITIC_TARGET:
                        for p, p_tgt in zip(critic.parameters(), critic_target.parameters()):
                            p_tgt.data.mul_(1 - TAU)
                            p_tgt.data.add_(TAU * p.data)
                    if USE_ACTOR_TARGET:
                        for p, p_tgt in zip(actor.parameters(), actor_target.parameters()):
                            p_tgt.data.mul_(1 - TAU)
                            p_tgt.data.add_(TAU * p.data)

                # ── Actor update ──────────────────────────────────────────
                set_requires_grad(critic, False)
                a_pred     = actor(s_nn)
                actor_loss = -critic(s_nn, normalize_action(a_pred)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP_NORM)
                actor_opt.step()
                set_requires_grad(critic, True)

                total_critic_loss += critic_loss.item()
                total_actor_loss  += actor_loss.item()
                update_count      += 1

            current_J = critic(next_state_nn, normalize_action(applied_action.detach()))
            total_J  += current_J.item()

            state    = next_state
            state_nn = next_state_nn

            if terminated or truncated:
                break

        n = step + 1
        episode_seconds = time.perf_counter() - episode_start
        episode_rewards.append(total_reward)
        episode_steps.append(n)
        episode_delta_vs.append(total_delta_v)
        episode_critic_losses.append(total_critic_loss / update_count if update_count else 0.0)
        episode_actor_losses.append(total_actor_loss   / update_count if update_count else 0.0)
        episode_docked.append(docked)
        episode_r_pos_totals.append(total_r_pos)
        episode_r_fuel_totals.append(total_r_fuel)

        if episode % LOG_EVERY == 0:
            dock_rate        = np.mean(episode_docked[-DOCK_RATE_WINDOW:]) * 100
            avg_critic_loss  = total_critic_loss / update_count if update_count else 0.0
            avg_actor_loss   = total_actor_loss  / update_count if update_count else 0.0
            avg_r_pos_total  = np.mean(episode_r_pos_totals[-LOG_EVERY:])
            avg_r_fuel_total = np.mean(episode_r_fuel_totals[-LOG_EVERY:])
            print(f"{episode:>6} | {n:>5} | {total_reward:>9.2f} | {avg_r_pos_total:>8.2f} | "
                f"{avg_r_fuel_total:>8.2f} | {total_r_terminal:>8.2f} | {total_r_milestone:>8.2f} | {total_delta_v:>7.3f} | "
                f"{avg_critic_loss:>9.4f} | {avg_actor_loss:>9.4f} | "
                f"{total_J/n:>8.3f} | {dock_rate:>5.1f}% | "
                f"{1000 * episode_seconds / n:>8.2f}")
        if (episode + 1) % LOG_EVERY == 0:
            episode_tag = f"ep_{episode + 1:04d}"
            traj_path   = tmp_dir / f"{episode_tag}.png"
            # min_dv_display: only draw burns >= 1% of max_dv so post-budget
            # zero-action steps don't clutter the plot.
            plot_trajectory(
                episode_states,
                episode_actions,
                str(traj_path),
                min_dv_display=ENV_MAX_DV * 0.01,
            )
            shutil.copy(traj_path, tmp_dir / "latest_trajectory.png")
            np.savez(
                tmp_dir / f"{episode_tag}.npz",
                states=np.array(episode_states),
                rewards=np.array([total_reward]),
                steps=np.array([n]),
                delta_v=np.array([total_delta_v]),
                docked=np.array([docked]),
            )

    torch.save(actor.state_dict(),  TRAINED_ACTOR_PATH)
    torch.save(critic.state_dict(), TRAINED_CRITIC_PATH)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("HDP Training Diagnostics", fontsize=14)

    plot_with_smooth(axes[0, 0], episode_rewards,       "reward",      "tab:blue",   "Total reward",      "reward")
    plot_with_smooth(axes[0, 1], episode_delta_vs,      "delta-v",     "tab:orange", "Total delta-V",     "m/s")
    plot_with_smooth(axes[0, 2], episode_steps,         "steps",       "tab:green",  "Episode length",    "steps")
    plot_with_smooth(axes[1, 0], episode_critic_losses, "critic loss", "tab:red",    "Critic loss (avg)", "MSE")
    plot_with_smooth(axes[1, 1], episode_actor_losses,  "actor loss",  "tab:purple", "Actor loss (avg)",  "loss")

    dock_rate = [
        np.mean(episode_docked[max(0, i - DOCK_RATE_WINDOW):i + 1]) * 100
        for i in range(len(episode_docked))
    ]
    axes[1, 2].plot(dock_rate, color="tab:cyan", linewidth=2)
    axes[1, 2].set_title(f"Dock rate ({DOCK_RATE_WINDOW}-ep rolling)")
    axes[1, 2].set_ylabel("%")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylim(0, 100)
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(DIAGNOSTICS_PLOT_PATH, dpi=150)
    plt.show()

    np.savez(
        TRAINING_HISTORY_PATH,
        rewards=np.array(episode_rewards),
        steps=np.array(episode_steps),
        delta_v=np.array(episode_delta_vs),
        critic_loss=np.array(episode_critic_losses),
        actor_loss=np.array(episode_actor_losses),
        docked=np.array(episode_docked),
    )


if __name__ == "__main__":
    main()