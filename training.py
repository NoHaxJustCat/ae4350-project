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
from libs.trajectory import plot_trajectory

# ── Feature flags ────────────────────────────────────────────────────────────
USE_REPLAY_BUFFER  = True   # experience replay (random mini-batch updates)
USE_ACTOR_TARGET   = True   # target actor network (soft-updated with TAU)
USE_CRITIC_TARGET  = True   # target critic network (soft-updated with TAU)
# Note: USE_ACTOR_TARGET / USE_CRITIC_TARGET only affect target *usage*;
#       when False the online network is used in its place.
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

    # Critic target — always built, but only soft-updated / queried when the
    # flag is set; otherwise the online critic doubles as its own target.
    critic_target = deepcopy(critic)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    # Actor target — same pattern.
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

    episode_rewards      = []
    episode_steps        = []
    episode_delta_vs     = []
    episode_critic_losses = []
    episode_actor_losses  = []
    episode_docked       = []

    print(f"{'ep':>6} | {'steps':>5} | {'reward':>9} | {'r_pos':>8} | {'r_fuel':>8} | "
          f"{'dv':>7} | {'c_loss':>9} | {'a_loss':>9} | {'J_mean':>8} | {'docked':>6} | "
          f"{'ms/step':>8}")
    print("-" * 110)

    for episode in range(NUM_EPISODES):
        episode_start = time.perf_counter()

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            prev_J = critic(state, actor(state).detach())
        episode_states = [state.detach().numpy().copy()]

        total_reward      = 0.0
        total_delta_v     = 0.0
        total_r_pos       = 0.0
        total_r_fuel      = 0.0
        total_critic_loss = 0.0
        total_actor_loss  = 0.0
        total_J           = 0.0
        update_count      = 0
        docked            = False

        for step in range(MAX_STEPS):
            action = actor(state).detach()
            total_delta_v += torch.linalg.norm(action).item()

            next_state, reward, terminated, truncated, info = env.step(action.numpy())
            next_state = torch.tensor(next_state, dtype=torch.float32)
            episode_states.append(next_state.detach().numpy().copy())
            reward_t = torch.tensor([reward], dtype=torch.float32)

            total_reward += reward
            total_r_pos  += info["reward_pos"]
            total_r_fuel += info["reward_fuel"]
            if info["docked"]:
                docked = True

            # ── Replay buffer ─────────────────────────────────────────────
            if USE_REPLAY_BUFFER:
                buffer.append((
                    state.numpy().copy(),
                    action.numpy().copy(),
                    reward,
                    next_state.numpy().copy(),
                    terminated or truncated,
                ))
                ready = len(buffer) >= MIN_BUFFER
            else:
                ready = True  # always update on-policy

            # ── Sample a batch (or use the current transition on-policy) ──
            if ready:
                if USE_REPLAY_BUFFER:
                    batch      = random.sample(buffer, BATCH_SIZE)
                    s, a, r, s2, done = zip(*batch)
                    s    = torch.tensor(np.array(s),    dtype=torch.float32)
                    a    = torch.tensor(np.array(a),    dtype=torch.float32)
                    r    = torch.tensor(np.array(r),    dtype=torch.float32).unsqueeze(1)
                    s2   = torch.tensor(np.array(s2),   dtype=torch.float32)
                    done = torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1)
                else:
                    # On-policy: single transition, shapes match batch API
                    s    = state.unsqueeze(0)
                    a    = action.unsqueeze(0)
                    r    = reward_t.unsqueeze(0)
                    s2   = next_state.unsqueeze(0)
                    done = torch.tensor([[terminated or truncated]], dtype=torch.float32)

                # ── Critic update ─────────────────────────────────────────
                with torch.no_grad():
                    a2_next  = actor_target(s2) if USE_ACTOR_TARGET  else actor(s2)
                    q_next   = critic_target(s2, a2_next) if USE_CRITIC_TARGET else critic(s2, a2_next)
                    target   = r + GAMMA * (1 - done) * q_next

                critic_loss = torch.mean((critic(s, a) - target) ** 2)
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

                # ── Actor update (model-based, gradient through propagator) ─
                set_requires_grad(critic, False)
                a_pred  = actor(s)
                s2_pred = env.propagate_torch(s, a_pred)
                a2_pred = actor(s2_pred)           # grad flows through both
                actor_loss = -critic(s2_pred, a2_pred).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP_NORM)
                actor_opt.step()
                set_requires_grad(critic, True)

                total_critic_loss += critic_loss.item()
                total_actor_loss  += actor_loss.item()
                update_count      += 1

            current_J = critic(next_state, action.detach())
            total_J  += current_J.item()

            state = next_state

            if terminated or truncated:
                break

        n = step + 1
        episode_seconds = time.perf_counter() - episode_start
        episode_rewards.append(total_reward)
        episode_steps.append(n)
        episode_delta_vs.append(total_delta_v)
        episode_critic_losses.append(total_critic_loss / update_count if update_count else 0.0)
        episode_actor_losses.append(total_actor_loss  / update_count if update_count else 0.0)
        episode_docked.append(docked)

        if episode % LOG_EVERY == 0:
            dock_rate       = np.mean(episode_docked[-DOCK_RATE_WINDOW:]) * 100
            avg_critic_loss = total_critic_loss / update_count if update_count else 0.0
            avg_actor_loss  = total_actor_loss  / update_count if update_count else 0.0
            print(f"{episode:>6} | {n:>5} | {total_reward:>9.2f} | {total_r_pos/n:>8.2f} | "
                  f"{total_r_fuel/n:>8.2f} | {total_delta_v:>7.3f} | "
                  f"{avg_critic_loss:>9.4f} | {avg_actor_loss:>9.4f} | "
                  f"{total_J/n:>8.3f} | {dock_rate:>5.1f}% | "
                  f"{1000 * episode_seconds / n:>8.2f}")

        if (episode + 1) % 50 == 0:
            episode_tag = f"ep_{episode + 1:04d}"
            plot_trajectory(episode_states, str(tmp_dir / f"{episode_tag}.png"))
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

    plot_with_smooth(axes[0, 0], episode_rewards,       "reward",      "tab:blue",   "Total reward",        "reward")
    plot_with_smooth(axes[0, 1], episode_delta_vs,      "delta-v",     "tab:orange", "Total delta-V",       "m/s")
    plot_with_smooth(axes[0, 2], episode_steps,         "steps",       "tab:green",  "Episode length",      "steps")
    plot_with_smooth(axes[1, 0], episode_critic_losses, "critic loss", "tab:red",    "Critic loss (avg)",   "MSE")
    plot_with_smooth(axes[1, 1], episode_actor_losses,  "actor loss",  "tab:purple", "Actor loss (avg)",    "loss")

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