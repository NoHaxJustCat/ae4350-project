"""Relative-motion trajectory plot in the LVLH x-z plane."""

import numpy as np
import matplotlib.pyplot as plt

from config import ENV_POS_TOLERANCE, MODE_2D
from lib.plots.style import COLOR_1, COLOR_2, COLOR_3, COLOR_4, save, style_axes, use_style


def plot_trajectory(states, actions=None, path="trajectory.png",
                    min_dv_display=0.0, title=None, margin=0.10):
    """`states` are raw physical states, `actions` the impulse applied at each
    (zero where coasting). Burns are drawn as arrows anchored at the position
    they were applied from. Axes are square and centred on the target."""
    use_style()
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or len(states) < 2:
        return None
    xi, zi = (0, 1) if MODE_2D else (0, 2)
    x, z = states[:, xi], states[:, zi]

    # Square axes centred on the target, sized to contain the whole trajectory
    # plus a margin. Deriving the figure shape from the data instead collapsed
    # the axes to zero height on short early episodes ("constrained_layout not
    # applied"), and left the target off-centre.
    reach = float(np.nanmax(np.abs(np.stack([x, z])))) if len(x) else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(x, z, color=COLOR_1, linewidth=1.5, label="Trajectory")
    ax.plot(x[0], z[0], "o", color=COLOR_2, markersize=8, markeredgecolor="black",
            markeredgewidth=0.8, zorder=5, label="Start")
    ax.plot(0.0, 0.0, "o", color=COLOR_3, markersize=9, markeredgecolor="black",
            markeredgewidth=0.8, zorder=5, label="Target")

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ENV_POS_TOLERANCE * np.cos(theta), ENV_POS_TOLERANCE * np.sin(theta),
            linestyle="--", linewidth=1.0, color=COLOR_3, zorder=4,
            label=f"Dock tolerance ({ENV_POS_TOLERANCE:.0f} m)")

    if actions is not None:
        # Arrow tips widen the plot: the first impulse of a V-bar transfer
        # points outward from the start, i.e. past the trajectory's own reach,
        # and was being clipped away by limits sized from the states alone.
        reach = max(reach, _draw_burns(ax, states, actions, xi, zi, min_dv_display))

    limit = max(reach, 2 * ENV_POS_TOLERANCE) * (1.0 + margin)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.set_xlabel(r"$x$ [m] (V-bar)")
    ax.set_ylabel(r"$z$ [m] (R-bar)")
    ax.set_title(title or "Relative trajectory")
    style_axes(ax, legend=False, equal=True)
    # Legend below the axes: a rendezvous spiral fills its own bounding box, so
    # any in-axes placement covers the trajectory.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=12,
              frameon=True, edgecolor="black", framealpha=1.0, columnspacing=0.9,
              handletextpad=0.4, borderpad=0.3)
    save(fig, path)
    plt.close(fig)
    return path


def _draw_burns(ax, states, actions, xi, zi, min_dv_display):
    """Draws one arrow per impulse and returns how far the tips reach from the
    target, so the caller can size the axes to keep every arrow on the figure."""
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or not len(actions):
        return 0.0
    mags = np.linalg.norm(actions, axis=1)
    burns = np.flatnonzero(mags > max(min_dv_display, 0.0))
    burns = burns[burns < len(states)]
    if not burns.size:
        return 0.0
    reach = max(float(np.nanmax(np.abs(states[:, [xi, zi]]))), 1.0)
    scale = 0.18 * reach / mags[burns].max()
    tip_reach = 0.0
    for i in burns:
        tip = (states[i, xi] + actions[i, 0] * scale,
               states[i, zi] + actions[i, -1] * scale)
        tip_reach = max(tip_reach, abs(tip[0]), abs(tip[1]))
        ax.annotate("", xy=tip, xytext=(states[i, xi], states[i, zi]),
                    arrowprops=dict(arrowstyle="->", color=COLOR_4, lw=1.4), zorder=6)
    ax.plot([], [], color=COLOR_4, linewidth=1.4,
            label=rf"$\Delta v$ impulses ({len(burns)})")
    return tip_reach
