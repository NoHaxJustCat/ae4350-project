"""
Trajectory visualisation for the CW rendezvous environment.

Dispatches to a 2-D (x-z plane) or 3-D plot depending on MODE_2D.

2-D state layout : [x, z, ẋ, ż]           → plot x vs z
3-D state layout : [x, y, z, ẋ, ẏ, ż]    → 3-D axes
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from libs.constants import MODE_2D

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.linewidth":   0.8,
    "mathtext.fontset": "cm",
    "legend.fontsize":  9,
})


def plot_trajectory(
    trajectory,
    dv_vectors=None,
    path: str = "trajectory.png",
    dv_scale: float = None,       # None → auto-scale to plot extent
    min_dv_display: float = 1e-6, # burns smaller than this are not drawn
):
    if MODE_2D:
        _plot_2d(trajectory, dv_vectors, path, dv_scale, min_dv_display)
    else:
        _plot_3d(trajectory, dv_vectors, path, dv_scale, min_dv_display)


# ── helpers ──────────────────────────────────────────────────────────────────

def _auto_scale(pos, dv, fraction=0.08):
    """Scale dv arrows to `fraction` of the trajectory's bounding-box diagonal."""
    span = pos.max(axis=0) - pos.min(axis=0)
    diag = np.linalg.norm(span)
    if diag < 1e-9:
        diag = 1.0
    dv_mag = np.linalg.norm(dv, axis=1).max()
    if dv_mag < 1e-12:
        return 1.0
    return (fraction * diag) / dv_mag


# ── 2-D ──────────────────────────────────────────────────────────────────────

def _plot_2d(trajectory, dv_vectors, path, dv_scale, min_dv_display):
    traj = np.asarray(trajectory, dtype=np.float64)
    # 2-D physical state: [x (V-bar), z (R-bar), ẋ, ż]
    x = traj[:, 0]
    z = traj[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(x, z,
            color="#1f5fa8", linewidth=1.2,
            marker="o", markersize=2.5,
            markerfacecolor="#1f5fa8", markeredgewidth=0,
            label="Trajectory", zorder=2)

    ax.scatter(x[0], z[0], c="#2c8f3d", marker="s", s=60, zorder=5, label="Start")
    ax.scatter(0.0,  0.0,  c="#c0392b", marker="*", s=200, zorder=5, label="Target")

    if dv_vectors is not None and len(dv_vectors) > 0:
        dv = np.asarray(dv_vectors, dtype=np.float64)          # (N, 2)
        # Filter out near-zero burns so the legend entry is still shown
        # even when most burns are zero (budget exhausted episodes).
        norms = np.linalg.norm(dv, axis=1)
        mask  = norms > min_dv_display

        if mask.any():
            origins = np.column_stack([x[:len(dv)], z[:len(dv)]])

            # Auto-scale: arrows cover ~8 % of the bounding-box diagonal
            scale = dv_scale if dv_scale is not None else _auto_scale(
                np.column_stack([x, z]), dv[mask]
            )

            ax.quiver(
                origins[mask, 0], origins[mask, 1],
                dv[mask, 0] * scale, dv[mask, 1] * scale,
                angles="xy", scale_units="xy", scale=1,
                color="#e67e22", linewidth=0.8, width=0.004,
                headwidth=4, headlength=5,
                label=rf"$\Delta v$ (×{scale:.0f})",
                zorder=3,
            )
        else:
            # All burns were zero — note it in the legend
            ax.plot([], [], color="#e67e22",
                    label=r"$\Delta v$ (budget exhausted)")

    ax.set_xlabel(r"$x$ — V-bar [m]")
    ax.set_ylabel(r"$z$ — R-bar [m]")
    ax.set_title("In-plane rendezvous trajectory (2-D CW)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── 3-D ──────────────────────────────────────────────────────────────────────

def _plot_3d(trajectory, dv_vectors, path, dv_scale, min_dv_display):
    traj = np.asarray(trajectory, dtype=np.float64)
    pos  = traj[:, 0:3]   # [x, y, z]

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            color="#1f5fa8", linewidth=1.2,
            marker="o", markersize=2.5,
            markerfacecolor="#1f5fa8", markeredgewidth=0,
            label="Trajectory")

    ax.scatter(*pos[0], c="#2c8f3d", marker="s",  s=50,  depthshade=False, label="Start")
    ax.scatter(0, 0, 0, c="#c0392b", marker="*",  s=200, depthshade=False, label="Target")

    dv = None
    if dv_vectors is not None and len(dv_vectors) > 0:
        dv    = np.asarray(dv_vectors, dtype=np.float64)
        norms = np.linalg.norm(dv, axis=1)
        mask  = norms > min_dv_display

        if mask.any():
            scale   = dv_scale if dv_scale is not None else _auto_scale(pos, dv[mask])
            origins = pos[:len(dv)]
            ax.quiver(
                origins[mask, 0], origins[mask, 1], origins[mask, 2],
                dv[mask, 0] * scale, dv[mask, 1] * scale, dv[mask, 2] * scale,
                color="#e67e22", linewidth=0.8, arrow_length_ratio=0.3,
                label=rf"$\Delta v$ (×{scale:.0f})",
            )

    # Equal aspect ratio
    all_pts = pos if dv is None else np.vstack(
        [pos, pos[:len(dv)] + dv * (dv_scale or 1.0)]
    )
    span  = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_r = max(span.max() / 2.0, 1.0)
    mid   = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_r, mid[0] + max_r)
    ax.set_ylim(mid[1] - max_r, mid[1] + max_r)
    ax.set_zlim(mid[2] - max_r, mid[2] + max_r)
    ax.view_init(elev=22, azim=-60)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.03)
        pane.set_edgecolor("gray")

    ax.set_xlabel(r"$x$ — V-bar [m]",  labelpad=10)
    ax.set_ylabel(r"$y$ — H-bar [m]",  labelpad=10)
    ax.set_zlabel(r"$z$ — R-bar [m]",  labelpad=10)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)