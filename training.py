from pathlib import Path
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
    LOG_EVERY,
    MAX_STEPS,
    NUM_EPISODES,
    OMEGA,
    SMOOTHING_WINDOW,
    TRAINED_ACTOR_PATH,
    TRAINED_CRITIC_PATH,
    TRAINING_HISTORY_PATH,
)
from libs.critic import Critic
from libs.env import CWRendezvousEnv
from libs.trajectory import plot_trajectory


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

    env = build_env()

    actor = Actor(max_action=env.max_dv)
    critic = Critic()
    actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    Path("trained").mkdir(parents=True, exist_ok=True)
    Path("out").mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards = []
    episode_steps = []
    episode_delta_vs = []
    episode_critic_losses = []
    episode_actor_losses = []
    episode_docked = []

    print(f"{'ep':>6} | {'steps':>5} | {'reward':>9} | {'r_pos':>8} | {'r_fuel':>8} | "
          f"{'dv':>7} | {'c_loss':>9} | {'a_loss':>9} | {'J_mean':>8} | {'docked':>6} | "
          f"{'ms/step':>8} | {'env':>7} | {'critic':>7} | {'actor':>7}")
    print("-" * 141)

    for episode in range(NUM_EPISODES):
        episode_start = time.perf_counter()
        env_step_seconds = 0.0
        critic_seconds = 0.0
        actor_seconds = 0.0

        state, _ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32)
        prev_J = critic(state)
        episode_states = [state.detach().numpy().copy()]

        total_reward = 0.0
        total_delta_v = 0.0
        total_r_pos = 0.0
        total_r_fuel = 0.0
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_J = 0.0
        docked = False

        for step in range(MAX_STEPS):
            with torch.no_grad():
                action = actor(state)
            total_delta_v += torch.linalg.norm(action).item()

            step_start = time.perf_counter()
            next_state, reward, terminated, truncated, info = env.step(action.numpy())
            env_step_seconds += time.perf_counter() - step_start

            next_state = torch.as_tensor(next_state, dtype=torch.float32)
            episode_states.append(next_state.detach().numpy().copy())
            reward_t = torch.tensor(reward, dtype=torch.float32)

            total_reward += reward
            total_r_pos += info["reward_pos"]
            total_r_fuel += info["reward_fuel"]
            if info["docked"]:
                docked = True

            step_start = time.perf_counter()
            current_J = critic(next_state)
            target = reward_t + GAMMA * current_J.detach()
            critic_loss = torch.mean((prev_J - target) ** 2)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=GRAD_CLIP_NORM)
            critic_opt.step()
            critic_seconds += time.perf_counter() - step_start

            step_start = time.perf_counter()
            set_requires_grad(critic, False)
            current_action = actor(state)
            predicted_next_state = env.propagate_torch(state, current_action)
            action_penalty = env.fuel_coeff * torch.linalg.norm(current_action)
            actor_loss = -critic(predicted_next_state).mean() + action_penalty
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=GRAD_CLIP_NORM)
            actor_opt.step()
            set_requires_grad(critic, True)
            actor_seconds += time.perf_counter() - step_start

            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            total_J += current_J.item()

            state = next_state
            prev_J = critic(state)

            if terminated or truncated:
                break

        n = step + 1
        episode_seconds = time.perf_counter() - episode_start
        episode_rewards.append(total_reward)
        episode_steps.append(n)
        episode_delta_vs.append(total_delta_v)
        episode_critic_losses.append(total_critic_loss / n)
        episode_actor_losses.append(total_actor_loss / n)
        episode_docked.append(docked)

        if episode % LOG_EVERY == 0:
            dock_rate = np.mean(episode_docked[-DOCK_RATE_WINDOW:]) * 100
            print(f"{episode:>6} | {n:>5} | {total_reward:>9.2f} | {total_r_pos/n:>8.2f} | "
                  f"{total_r_fuel/n:>8.2f} | {total_delta_v:>7.3f} | "
                  f"{total_critic_loss/n:>9.4f} | {total_actor_loss/n:>9.4f} | "
                  f"{total_J/n:>8.3f} | {dock_rate:>5.1f}% | "
                  f"{1000 * episode_seconds / n:>8.2f} | {1000 * env_step_seconds / n:>7.2f} | "
                  f"{1000 * critic_seconds / n:>7.2f} | {1000 * actor_seconds / n:>7.2f}")

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

    torch.save(actor.state_dict(), TRAINED_ACTOR_PATH)
    torch.save(critic.state_dict(), TRAINED_CRITIC_PATH)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("HDP Training Diagnostics", fontsize=14)

    plot_with_smooth(axes[0, 0], episode_rewards, "reward", "tab:blue", "Total reward", "reward")
    plot_with_smooth(axes[0, 1], episode_delta_vs, "delta-v", "tab:orange", "Total delta-V", "m/s")
    plot_with_smooth(axes[0, 2], episode_steps, "steps", "tab:green", "Episode length", "steps")
    plot_with_smooth(axes[1, 0], episode_critic_losses, "critic loss", "tab:red", "Critic loss (avg)", "MSE")
    plot_with_smooth(axes[1, 1], episode_actor_losses, "actor loss", "tab:purple", "Actor loss (avg)", "loss")

    dock_rate = [np.mean(episode_docked[max(0, i - DOCK_RATE_WINDOW):i + 1]) * 100 for i in range(len(episode_docked))]
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