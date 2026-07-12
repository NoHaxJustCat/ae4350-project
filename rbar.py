"""
Goal 2 evaluation: R-bar docking with coupled x/z displacement, both mirrored
sign combinations (+x,-z) and (-x,+z).

Usage:
    python rbar.py                              # evaluates the most recently trained model
    python rbar.py trained/<session>/<run_tag>/rbar_td3.zip   # evaluates a specific one
"""

import sys

from stable_baselines3 import TD3

from libs.evaluate import run_episode, print_summary, find_latest_model
from libs.trajectory import plot_trajectory

model_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_model("rbar")
print(f"Loading model: {model_path}")
model = TD3.load(model_path)

# sign=+1 -> direction (+ratio,-1) i.e. (+x,-z); sign=-1 -> (-x,+z)
for sign, label in [(+1, "(+x, -z)"), (-1, "(-x, +z)")]:
    result = run_episode(model, scenario="rbar", sign=sign)
    print_summary(label, result, scenario="rbar")
    tag = "pxmz" if sign > 0 else "mxpz"
    plot_trajectory(result["states"], result["actions"], f"out/rbar_{tag}_trajectory.png")

print(f"\nTrajectory plots saved to out/rbar_pxmz_trajectory.png, out/rbar_mxpz_trajectory.png")
