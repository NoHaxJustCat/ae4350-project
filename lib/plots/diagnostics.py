"""Six-panel training-diagnostics figure, from a history dict or a saved npz."""

import numpy as np
import matplotlib.pyplot as plt

from config import (
    DOCK_RATE_WINDOW, ENV_ANGLE_CURRICULUM_MAX_DEG, ENV_BURN_DEADZONE_FRAC, SMOOTHING_WINDOW,
)
from lib.plots.style import COLOR_1, COLOR_2, COLOR_4, COLOR_5, save, style_axes, use_style


def moving_average(values, window=SMOOTHING_WINDOW):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def trend(ax, data, label, color, linewidth=1.5):
    """Raw series faint behind its moving average."""
    data = np.asarray(data, dtype=float)
    ax.plot(data, color=color, linewidth=0.6, alpha=0.22)
    smooth = moving_average(data)
    ax.plot(np.arange(len(data) - len(smooth), len(data)), smooth,
            color=color, linewidth=linewidth, label=label)


def rolling_dock_rate(docked, window=DOCK_RATE_WINDOW):
    docked = np.asarray(docked, dtype=float)
    return np.array([100 * docked[max(0, i - window):i + 1].mean() for i in range(len(docked))])


def panel_reward(ax, h):
    trend(ax, h["rewards"], "reward", COLOR_1)
    ax.set_title("Total reward")
    ax.set_ylabel("reward")


def panel_fuel(ax, h):
    trend(ax, h["dv_ratio"], r"$\Delta v/\Delta v_{opt}$", COLOR_4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="optimum")
    ax.set_title("Fuel vs. achievable optimum")
    ax.set_ylabel(r"$\Delta v / \Delta v_{opt}$")


def panel_dock(ax, h):
    ax.plot(rolling_dock_rate(h["docked"]), color=COLOR_2, linewidth=1.5)
    ax.set_title(f"Dock rate ({DOCK_RATE_WINDOW}-episode rolling)")
    ax.set_ylabel(r"\%" if plt.rcParams["text.usetex"] else "%")
    ax.set_ylim(0, 100)


def panel_reward_split(ax, h):
    trend(ax, h["r_pos"], r"$r_{pos}$ (shaping)", COLOR_5)
    trend(ax, h["r_fuel"], r"$r_{fuel}$", COLOR_1)
    trend(ax, h["r_term"], r"$r_{term}$ (dock + stop)", COLOR_4)
    ax.set_title("Reward breakdown")
    ax.set_ylabel("reward")


def panel_noise(ax, h):
    ax.plot(h["noise_std"], color=COLOR_5, linewidth=1.5, label="noise std")
    ax.axhline(ENV_BURN_DEADZONE_FRAC, color="black", linestyle="--", linewidth=1.0,
               label=f"deadzone ({ENV_BURN_DEADZONE_FRAC})")
    ax.set_title("Exploration noise")
    ax.set_ylabel("std [action units]")
    ax.set_ylim(bottom=0)


def panel_curriculum(ax, h):
    """Distance on the left axis; the angle sector on a right axis when active."""
    ax.plot(h["curriculum_distance"], color=COLOR_1, linewidth=1.5, label="distance")
    ax.set_ylabel("distance [m]", color=COLOR_1)
    ax.tick_params(axis="y", colors=COLOR_1)
    title = "Curriculum"

    angle = np.asarray(h["angle_half_width_deg"], dtype=float) \
        if "angle_half_width_deg" in h else np.array([])
    if angle.size and np.isfinite(angle).any():
        twin = ax.twinx()
        twin.plot(angle, color=COLOR_4, linewidth=1.5, label="angle sector")
        twin.set_ylabel("sector half-width [deg]", color=COLOR_4)
        twin.tick_params(axis="y", colors=COLOR_4)
        # Fixed 0..max: the reading that matters is how far toward the full
        # circle the sector has opened, not its local range.
        twin.set_ylim(0.0, ENV_ANGLE_CURRICULUM_MAX_DEG)
        title += " (distance + angle sector)"
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc="upper left",
                  frameon=True, edgecolor="black", framealpha=1.0)
    ax.set_title(title)


PANELS = [panel_reward, panel_fuel, panel_dock,
          panel_reward_split, panel_noise, panel_curriculum]


def build_diagnostics_figure(history, scenario, path=None):
    """Returns the figure; saves and closes it when `path` is given."""
    use_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"TD3 training diagnostics -- scenario = {scenario}", fontsize=18)
    for panel, ax in zip(PANELS, axes.ravel()):
        panel(ax, history)
        ax.set_xlabel("Episode")
        style_axes(ax, legend=panel is not panel_curriculum)
    if path is not None:
        save(fig, path)
        plt.close(fig)
    return fig
