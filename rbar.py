"""
Goal 2 evaluation: R-bar docking with coupled x/z displacement, both mirrored
sign combinations (+x,-z) and (-x,+z).

Usage:
    python rbar.py
"""

from stable_baselines3 import TD3

from libs.constants import TRAINED_MODEL_DIR
from libs.evaluate import run_episode, print_summary
from libs.trajectory import plot_trajectory

model = TD3.load(f"{TRAINED_MODEL_DIR}/rbar_td3")

# sign=+1 -> direction (+ratio,-1) i.e. (+x,-z); sign=-1 -> (-x,+z)
for sign, label in [(+1, "(+x, -z)"), (-1, "(-x, +z)")]:
    result = run_episode(model, scenario="rbar", sign=sign)
    print_summary(label, result, scenario="rbar")
    tag = "pxmz" if sign > 0 else "mxpz"
    plot_trajectory(result["states"], result["actions"], f"out/rbar_{tag}_trajectory.png")

print(f"\nTrajectory plots saved to out/rbar_pxmz_trajectory.png, out/rbar_mxpz_trajectory.png")
