import numpy as np
from libs.env import CWRendezvousEnv
from libs.constants import (
    CLOSE_NOTE_THRESHOLD,
    ENV_INITIAL_STATE_VBAR,
    MAX_STEPS,
    OMEGA,
    ORBIT_PERIOD,
    TRAINED_ACTOR_PATH,
    TRAINED_CRITIC_PATH,
)
from libs.actor import Actor
from libs.critic import Critic
import torch
import matplotlib.pyplot as plt
from libs.trajectory import plot_trajectory

env = CWRendezvousEnv(omega=OMEGA)
actor = Actor(max_action=env.max_dv)
actor.load_state_dict(torch.load(TRAINED_ACTOR_PATH))
critic = Critic()
critic.load_state_dict(torch.load(TRAINED_CRITIC_PATH))

actor.eval()
total_dv = 0.0

print(f"{'step':>4} | {'pos_err':>8} | {'vel_err':>8} | {'action_norm':>11} | {'reward':>8} | {'docked':>6} | {'done':>6} | note")
print("-" * 80)

with torch.no_grad():
    state, _ = env.reset()
    env.state = np.array(ENV_INITIAL_STATE_VBAR, dtype=np.float64)
    env.elapsed_time = 0.0
    state = torch.tensor(env.state, dtype=torch.float32)
    trajectory = [env.state.copy()]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        action = actor(state)
        action_np = np.clip(action.numpy(), -env.max_dv, env.max_dv)
        total_dv += np.linalg.norm(action_np)

        next_state, reward, terminated, truncated, info = env.step(action_np)

        pos_err = info['distance']
        vel_err = np.linalg.norm(next_state[3:6])
        action_norm = np.linalg.norm(action_np)
        note = ""
        if pos_err < CLOSE_NOTE_THRESHOLD:
            note = "<<< CLOSE"
        if info['docked']:
            note = "<<< DOCKED"
        if pos_err > env.boundary * 0.8:
            note = "<<< NEAR BOUNDARY"
        if truncated:
            note = "<<< TIMEOUT"

        print(f"{step+1:>4} | {pos_err:>8.3f} | {vel_err:>8.4f} | {action_norm:>11.5f} | {reward:>8.3f} | {str(info['docked']):>6} | {str(terminated or truncated):>6} | {note}")

        state = torch.tensor(next_state, dtype=torch.float32)
        trajectory.append(next_state.copy())
        total_reward += reward

        if terminated or truncated:
            break

# summary
print("-" * 80)
print(f"Steps: {step+1} / {MAX_STEPS}  (timeout at {env.timeout/env.dt:.0f} steps)")
print(f"Total reward:   {total_reward:.2f}")
print(f"Docked:         {info['docked']}")
print(f"Final distance: {info['distance']:.4f} m")
print(f"Total delta-V:  {total_dv:.4f} m/s")
print(f"")
print(f"Env params: dt={env.dt}, boundary={env.boundary}, timeout={env.timeout}, pos_tol={env.pos_tolerance}")
print(f"Constants:  T={ORBIT_PERIOD}, max_steps={MAX_STEPS}")
print(f"")
dv_vbar = OMEGA / (3 * np.pi) * 100
dv_rbar = OMEGA / 2 * 100
print(f"Reference V-bar 2-impulse delta-V: {dv_vbar:.4f} m/s")
print(f"Reference R-bar 2-impulse delta-V: {dv_rbar:.4f} m/s")

plot_trajectory(trajectory, "out/yplus_trajectory.png")