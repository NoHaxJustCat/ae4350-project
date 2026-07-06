import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(trajectory, path: str):
    traj = np.array(trajectory)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(traj[:,1], traj[:,0], marker='o', markersize=2)  # along-track vs radial, adjust to your LVLH axes
    ax.scatter([0], [0], c='red', marker='*', s=150, label='Target')
    ax.scatter([traj[0,1]], [traj[0,0]], c='green', marker='s', label='Start')
    ax.set_xlabel('y (along-track) [m]')
    ax.set_ylabel('x (radial) [m]')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)
    fig.savefig(path)
    plt.close(fig)
    