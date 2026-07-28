"""Training diagnostics: one figure per metric, written to <run>/diagnostics/plots."""

import numpy as np
import matplotlib.pyplot as plt

from config import (
    DOCK_RATE_WINDOW, ENV_ANGLE_CURRICULUM_MAX_DEG, ENV_BURN_DEADZONE_FRAC, SMOOTHING_WINDOW,
)
from lib.plots.style import COLOR_1, COLOR_2, COLOR_4, COLOR_5, new_figure, save, style_axes


# A 6.5in figure at 300 dpi is ~1950 px wide, so past a few thousand points
# every extra sample is invisible -- but matplotlib still allocates and renders
# it, twice, because savefig uses bbox_inches="tight". Measured cost of ONE
# write_panels() call, which runs every DIAG_UPDATE_SECONDS:
#     25k episodes -> 30 s     100k -> 83 s     400k -> 141 s
# A 2M-step run reaches ~1M episodes, so this outran its own 60 s cadence and
# grew without bound until the machine ran out of memory.
MAX_PLOT_POINTS = 4000


def decimate(y, offset=0):
    """(x, y) thinned to at most MAX_PLOT_POINTS for plotting."""
    y = np.asarray(y, dtype=float)
    if y.size <= MAX_PLOT_POINTS:
        return np.arange(y.size) + offset, y
    idx = np.arange(0, y.size, int(np.ceil(y.size / MAX_PLOT_POINTS)))
    return idx + offset, y[idx]


def moving_average(values, window=SMOOTHING_WINDOW):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def trend(ax, data, label, color, linewidth=1.5):
    """Raw series faint behind its moving average."""
    data = np.asarray(data, dtype=float)
    if not data.size:
        return
    ax.plot(*decimate(data), color=color, linewidth=0.6, alpha=0.22)
    # Smooth on the FULL series, then thin -- decimating first would alias.
    smooth = moving_average(data)
    ax.plot(*decimate(smooth, offset=len(data) - len(smooth)),
            color=color, linewidth=linewidth, label=label)


def rolling_dock_rate(docked, window=DOCK_RATE_WINDOW):
    """Vectorized via a cumulative sum. The previous version built one numpy
    slice per episode in a Python loop -- O(n) temporaries every redraw."""
    docked = np.asarray(docked, dtype=float)
    if not docked.size:
        return docked
    cumulative = np.concatenate([[0.0], np.cumsum(docked)])
    hi = np.arange(1, docked.size + 1)
    lo = np.maximum(0, np.arange(docked.size) - window)
    return 100.0 * (cumulative[hi] - cumulative[lo]) / (hi - lo)


# -- panels ------------------------------------------------------------------
# Each takes (ax, history) and draws one metric. `PANELS` maps the output file
# stem to (title, draw function).

def _reward(ax, h):
    trend(ax, h["rewards"], "reward", COLOR_1)
    ax.set_ylabel("reward")


def _dv_ratio(ax, h):
    trend(ax, h["dv_ratio"], r"$\Delta v/\Delta v_{opt}$", COLOR_4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="optimum")
    ax.set_ylabel(r"$\Delta v / \Delta v_{opt}$")
    # Log: early episodes reach several hundred x optimal, which on a linear
    # axis squashes the 1-10x band that actually matters into a sliver.
    ax.set_yscale("log")


def _dock_rate(ax, h):
    ax.plot(*decimate(rolling_dock_rate(h["docked"])), color=COLOR_2,
            linewidth=1.5, label="dock rate")
    ax.set_ylabel(r"\%" if plt.rcParams["text.usetex"] else "%")
    ax.set_ylim(0, 100)


def _steps(ax, h):
    trend(ax, h["steps"], "steps", COLOR_2)
    ax.set_ylabel("steps")


def _r_pos(ax, h):
    trend(ax, h["r_pos"], r"$r_{pos}$", COLOR_5)
    ax.set_ylabel("reward")


def _r_fuel(ax, h):
    trend(ax, h["r_fuel"], r"$r_{fuel}$", COLOR_1)
    ax.set_ylabel("reward")


def _r_term(ax, h):
    trend(ax, h["r_term"], r"$r_{term}$", COLOR_4)
    ax.set_ylabel("reward")


def _noise(ax, h):
    ax.plot(*decimate(h["noise_std"]), color=COLOR_5, linewidth=1.5, label="noise std")
    ax.axhline(ENV_BURN_DEADZONE_FRAC, color="black", linestyle="--", linewidth=1.0,
               label=f"deadzone ({ENV_BURN_DEADZONE_FRAC})")
    ax.set_ylabel("std [action units]")
    ax.set_ylim(bottom=0)


def _actor_loss(ax, h):
    trend(ax, h["actor_loss"], "actor loss", COLOR_1)
    ax.set_ylabel("loss")


def _critic_loss(ax, h):
    trend(ax, h["critic_loss"], "critic loss", COLOR_4)
    ax.set_ylabel("loss")
    ax.set_yscale("symlog")


def _curriculum(ax, h):
    """Distance on the left axis; the angle sector on a right axis when active."""
    ax.plot(*decimate(h["curriculum_distance"]), color=COLOR_1,
            linewidth=1.5, label="distance")
    ax.set_ylabel("distance [m]", color=COLOR_1)
    ax.tick_params(axis="y", colors=COLOR_1)

    angle = np.asarray(h["angle_half_width_deg"], dtype=float) \
        if "angle_half_width_deg" in h else np.array([])
    if angle.size and np.isfinite(angle).any():
        twin = ax.twinx()
        twin.plot(*decimate(angle), color=COLOR_4, linewidth=1.5, label="angle sector")
        twin.set_ylabel("sector half-width [deg]", color=COLOR_4)
        twin.tick_params(axis="y", colors=COLOR_4)
        # Fixed 0..max: the reading that matters is how far toward the full
        # circle the sector has opened, not its local range.
        twin.set_ylim(0.0, ENV_ANGLE_CURRICULUM_MAX_DEG)
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc="upper left",
                  frameon=True, edgecolor="black", framealpha=1.0)


PANELS = {
    "reward":       ("Total reward", _reward),
    "dv_ratio":     ("Fuel vs. achievable optimum", _dv_ratio),
    "dock_rate":    (f"Dock rate ({DOCK_RATE_WINDOW}-episode rolling)", _dock_rate),
    "episode_length": ("Episode length", _steps),
    "reward_pos":   ("Shaping reward", _r_pos),
    "reward_fuel":  ("Fuel reward", _r_fuel),
    "reward_term":  ("Terminal reward (dock + stop)", _r_term),
    "noise":        ("Exploration noise", _noise),
    "actor_loss":   ("Actor loss", _actor_loss),
    "critic_loss":  ("Critic loss", _critic_loss),
    "curriculum":   ("Curriculum", _curriculum),
}


def write_panels(history, scenario, out_dir, only=None):
    """Writes one PNG per metric into `out_dir`. Returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, (title, draw) in PANELS.items():
        if only is not None and stem not in only:
            continue
        fig, ax = new_figure(figsize=(6.5, 4.2))
        draw(ax, history)
        ax.set_title(f"{title} -- {scenario}")
        ax.set_xlabel("Episode")
        style_axes(ax, legend=draw is not _curriculum)
        written.append(save(fig, out_dir / f"{stem}.png"))
        plt.close(fig)
    return written
