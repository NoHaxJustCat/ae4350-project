import numpy as np
from libs.env import CWRendezvousEnv
from libs.constants import ENV_INITIAL_STATE_VBAR, ENV_INITIAL_STATE_XPLUS, MAX_STEPS, OMEGA, TRAINED_ACTOR_PATH, TRAINED_CRITIC_PATH
from libs.actor import Actor
from libs.critic import Critic
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from libs.trajectory import plot_trajectory

env = CWRendezvousEnv(omega=OMEGA)

actor = Actor(max_action=env.max_dv)
actor.load_state_dict(torch.load(TRAINED_ACTOR_PATH))

critic = Critic()
critic.load_state_dict(torch.load(TRAINED_CRITIC_PATH))

# Random run
actor.eval()
total_dv = 0.0
with torch.no_grad():
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    trajectory = [state.numpy()]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        action = actor(state)
        action_np = action.numpy()
        clipped_action = np.clip(action_np, -env.max_dv, env.max_dv)  
        total_dv += np.linalg.norm(clipped_action)

        next_state, reward, terminated, truncated, info = env.step(clipped_action)
        state = torch.tensor(next_state, dtype=torch.float32)
        trajectory.append(state.numpy())
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Steps: {step+1}, Total reward: {total_reward:.2f}, Docked: {info['docked']}, "
        f"Final distance: {info['distance']:.3f} m, Total delta-V: {total_dv:.4f} m/s")
plot_trajectory(trajectory, "out/rand_trajectory.png")

# Shift in posx 
actor.eval()
total_dv = 0.0
with torch.no_grad():
    state, _ = env.reset()
    env.state = np.array(ENV_INITIAL_STATE_XPLUS, dtype=np.float64)  # sync the env
    env.elapsed_time = 0.0  # also reset the clock, otherwise timeout is wrong
    state = torch.tensor(env.state, dtype=torch.float32)  # derive tensor from env.state
    trajectory = [state.numpy()]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        action = actor(state)
        action_np = action.numpy()
        clipped_action = np.clip(action_np, -env.max_dv, env.max_dv)  
        total_dv += np.linalg.norm(clipped_action)

        next_state, reward, terminated, truncated, info = env.step(action.numpy())
        state = torch.tensor(next_state, dtype=torch.float32)
        trajectory.append(state.numpy())
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Steps: {step+1}, Total reward: {total_reward:.2f}, Docked: {info['docked']}, "
        f"Final distance: {info['distance']:.3f} m, Total delta-V: {total_dv:.4f} m/s")

plot_trajectory(trajectory, "out/xplus_trajectory.png")
