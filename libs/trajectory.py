import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(trajectory, path: str):
    traj = np.array(trajectory)
    plt.figure(figsize=(6,6))
    plt.plot(traj[:,1], traj[:,0], marker='o', markersize=2)  # along-track vs radial, adjust to your LVLH axes
    plt.scatter([0], [0], c='red', marker='*', s=150, label='Target')
    plt.scatter([traj[0,1]], [traj[0,0]], c='green', marker='s', label='Start')
    plt.xlabel('y (along-track) [m]')
    plt.ylabel('x (radial) [m]')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    