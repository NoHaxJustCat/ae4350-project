"""
Goal 1 evaluation: V-bar (pure Δx) docking, both +x and -x displacements.

Usage:
    python vbar.py
"""

from stable_baselines3 import TD3

from libs.constants import TRAINED_MODEL_DIR
from libs.evaluate import run_episode, print_summary
from libs.trajectory import plot_trajectory

model = TD3.load(f"{TRAINED_MODEL_DIR}/vbar_td3")

for sign, label in [(+1, "x+ (positive V-bar displacement)"), (-1, "x- (negative V-bar displacement)")]:
    result = run_episode(model, scenario="vbar", sign=sign)
    print_summary(label, result, scenario="vbar")
    tag = "xplus" if sign > 0 else "xminus"
    plot_trajectory(result["states"], result["actions"], f"out/vbar_{tag}_trajectory.png")

print(f"\nTrajectory plots saved to out/vbar_xplus_trajectory.png, out/vbar_xminus_trajectory.png")
