"""
Goal 1 evaluation: V-bar (pure Δx) docking, both +x and -x displacements.

Usage:
    python vbar.py                              # evaluates the most recently trained model
    python vbar.py trained/<session>/<run_tag>/vbar_td3.zip   # evaluates a specific one
"""

import sys

from stable_baselines3 import TD3

from libs.evaluate import run_episode, print_summary, find_latest_model
from libs.trajectory import plot_trajectory

model_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_model("vbar")
print(f"Loading model: {model_path}")
model = TD3.load(model_path)

for sign, label in [(+1, "x+ (positive V-bar displacement)"), (-1, "x- (negative V-bar displacement)")]:
    result = run_episode(model, scenario="vbar", sign=sign)
    print_summary(label, result, scenario="vbar")
    tag = "xplus" if sign > 0 else "xminus"
    plot_trajectory(result["states"], result["actions"], f"out/vbar_{tag}_trajectory.png")

print(f"\nTrajectory plots saved to out/vbar_xplus_trajectory.png, out/vbar_xminus_trajectory.png")
