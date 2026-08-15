"""Relative-motion trajectory plot in the LVLH x-z plane."""

import numpy as np
import matplotlib.pyplot as plt

from config import ENV_POS_TOLERANCE, MODE_2D
from lib.plots.style import COLOR_1, COLOR_2, COLOR_3, COLOR_4, save, style_axes, use_style

# Figure geometry. AXES_HEIGHT_IN sizes the plotting box; the pads are the
# fixed room the labels, title and legend need around it. ASPECT_LIMITS caps
# how elongated the box may get -- an early episode can be a near-flat sliver,
# and deriving the figure shape from that alone once collapsed the axes to zero
# height ("constrained_layout not applied").
AXES_HEIGHT_IN = 4.0
PAD_W_IN = 1.0
PAD_H_IN = 1.8
ASPECT_LIMITS = (0.75, 2.4)


def _frame(bounds, margin):
    """(xlim, zlim, aspect) from data bounds [x_lo, x_hi, z_lo, z_hi].

    x follows the data. z is symmetric about zero, so the target sits on the
    vertical centreline whatever the trajectory does above or below it. The
    shorter axis is then padded until the box matches the clamped aspect, which
    keeps the scale equal on both axes without leaving a letterboxed strip.
    """
    x_lo, x_hi, z_lo, z_hi = bounds
    floor = 2 * ENV_POS_TOLERANCE
    x_mid = 0.5 * (x_lo + x_hi)
    half_w = max(0.5 * (x_hi - x_lo), floor) * (1.0 + margin)
    half_h = max(abs(z_lo), abs(z_hi), floor) * (1.0 + margin)

    aspect = min(max(half_w / half_h, ASPECT_LIMITS[0]), ASPECT_LIMITS[1])
    if half_w / half_h > aspect:
        half_h = half_w / aspect
    else:
        half_w = half_h * aspect
    return (x_mid - half_w, x_mid + half_w), (-half_h, half_h), aspect


def plot_trajectory(states, actions=None, path="trajectory.png",
                    min_dv_display=0.0, title=None, margin=0.10):
    """`states` are raw physical states, `actions` the impulse applied at each
    (zero where coasting). Burns are drawn as arrows anchored at the position
    they were applied from. The scale is equal on both axes; the target sits at
    z = 0 in the middle of the vertical range but is NOT centred horizontally,
    so the trajectory fills the frame instead of half of it."""
    use_style()
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or len(states) < 2:
        return None
    xi, zi = (0, 1) if MODE_2D else (0, 2)
    x, z = states[:, xi], states[:, zi]

    bounds = [float(np.nanmin(x)), float(np.nanmax(x)),
              float(np.nanmin(z)), float(np.nanmax(z))]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(x, z, color=COLOR_1, linewidth=1.5, label="Trajectory")
    ax.plot(x[0], z[0], "o", color=COLOR_2, markersize=8, markeredgecolor="black",
            markeredgewidth=0.8, zorder=5, label="Start")
    ax.plot(0.0, 0.0, "o", color=COLOR_3, markersize=9, markeredgecolor="black",
            markeredgewidth=0.8, zorder=5, label="Target")

    theta = np.linspace(0, 2 * np.pi, 200)

    if actions is not None:
        # Arrow tips widen the plot: the first impulse of a V-bar transfer
        # points outward from the start, i.e. past the trajectory's own reach,
        # and was being clipped away by limits sized from the states alone.
        tips = _draw_burns(ax, states, actions, xi, zi, min_dv_display)
        if tips is not None:
            bounds = [min(bounds[0], tips[0]), max(bounds[1], tips[1]),
                      min(bounds[2], tips[2]), max(bounds[3], tips[3])]

    (x_lo, x_hi), (z_lo, z_hi), aspect = _frame(bounds, margin)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    # Equal scale on both axes shrinks the axes box to the data's own shape and
    # letterboxes whatever is left over, so the FIGURE has to be given that
    # shape too or the leftover is what the reader sees. The pads are the room
    # taken by the labels, title and legend, which do not scale with the data.
    fig.set_size_inches(AXES_HEIGHT_IN * aspect + PAD_W_IN, AXES_HEIGHT_IN + PAD_H_IN)

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
    """Draws one arrow per impulse and returns the bounding box of the arrow
    TIPS as [x_lo, x_hi, z_lo, z_hi], so the caller can size the axes to keep
    every arrow on the figure. None when there is nothing to draw."""
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or not len(actions):
        return None
    mags = np.linalg.norm(actions, axis=1)
    burns = np.flatnonzero(mags > max(min_dv_display, 0.0))
    burns = burns[burns < len(states)]
    if not burns.size:
        return None
    reach = max(float(np.nanmax(np.abs(states[:, [xi, zi]]))), 1.0)
    scale = 0.18 * reach / mags[burns].max()
    tips = []
    for i in burns:
        tip = (states[i, xi] + actions[i, 0] * scale,
               states[i, zi] + actions[i, -1] * scale)
        tips.append(tip)
        ax.annotate("", xy=tip, xytext=(states[i, xi], states[i, zi]),
                    arrowprops=dict(arrowstyle="->", color=COLOR_4, lw=1.4), zorder=6)
    tips = np.asarray(tips)
    ax.plot([], [], color=COLOR_4, linewidth=1.4,
            label=rf"$\Delta v$ impulses")
    return [tips[:, 0].min(), tips[:, 0].max(), tips[:, 1].min(), tips[:, 1].max()]
