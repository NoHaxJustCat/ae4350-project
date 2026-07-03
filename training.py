import numpy as np
from gymnasium.utils.env_checker import check_env
from libs.env import CWRendezvousEnv
from libs.constants import *
from libs.actor import Actor
import matplotlib.pyplot as plt
from libs.critic import Critic
import torch
import torch.optim as optim



#########################################
# Environment definition
#########################################

check_env(CWRendezvousEnv(omega=omega))
env = CWRendezvousEnv(omega=omega)

#########################################

actor = Actor(max_action=env.max_dv)
critic = Critic()
actor_opt = optim.Adam(actor.parameters(), lr=1e-5)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

# --- Reward tracking for plotting ---
episode_rewards = []
episode_steps = []
episode_delta_vs = []
 
for episode in range(num_episodes):
    
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
 
    # Track the previous value for the TD update
    prev_J = critic(state)
 
    total_reward = 0.0  # accumulate reward for this episode
    total_delta_v = 0.0
 
    for step in range(max_steps):
        # 1. Actor selects action u = pi(s)
        action = actor(state).detach()  # Detach to only update critic first
        total_delta_v += float(torch.linalg.norm(action).item())
 
        # 2. Interaction with environment
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward_t = torch.tensor([reward], dtype=torch.float32)
 
        total_reward += reward  # accumulate raw scalar reward
 
        # 3. Critic Evaluation
        current_J = critic(next_state)
 
        # 4. Critic Training: Minimize TD Error
        # target = r + gamma * J(next_state)
        target = reward_t + gamma * current_J.detach()
        critic_loss = torch.mean((prev_J - target) ** 2)
 
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
 
        # 5. Actor Training: Maximize Value (Minimize J error)
        current_action = actor(state)
        predicted_next_state = env.propagate_torch(state, current_action)
        action_penalty = env.fuel_coeff * torch.linalg.norm(current_action)
        actor_loss = -critic(predicted_next_state).mean() + action_penalty
 
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
 
        # Preparation for next step
        state = next_state
        prev_J = critic(state)
 
        if terminated or truncated:
            break
 
    episode_rewards.append(total_reward)
    episode_steps.append(step + 1)
    episode_delta_vs.append(total_delta_v)
 
    if episode % 10 == 0:
        print(
            f"Episode {episode:4d} | steps {step+1:3d} | reward {total_reward:8.3f} | "
            f"delta-v {total_delta_v:7.3f}"
        )
 
torch.save(actor.state_dict(), "out/actor.pt")
torch.save(critic.state_dict(), "out/critic.pt")
 
#########################################
# Plot reward over training
#########################################
 
def moving_average(x, window=10):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")
 
episode_rewards = np.array(episode_rewards)
smoothed = moving_average(episode_rewards, window=10)
 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(episode_rewards, alpha=0.35, color="tab:blue", label="Episode reward")
if len(smoothed) > 0:
    offset = len(episode_rewards) - len(smoothed)
    ax.plot(
        np.arange(offset, len(episode_rewards)),
        smoothed,
        color="tab:blue",
        linewidth=2,
        label="10-episode moving average",
    )
ax.set_xlabel("Episode")
ax.set_ylabel("Total reward")
ax.set_title("HDP Training Reward over Episodes")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("out/reward_curve.png", dpi=150)
plt.show()
 
# Optional: also save raw data in case you want to re-plot later
np.savez(
    "out/training_history.npz",
    rewards=episode_rewards,
    steps=np.array(episode_steps),
    delta_v=np.array(episode_delta_vs),
)
 
