import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from libs.constants import MODE_2D

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.linewidth":  0.8,
    "mathtext.fontset": "cm",
    "legend.fontsize": 9,
})


def plot_trajectory(
    trajectory,
    dv_vectors=None,
    path: str   = "trajectory.png",
    dv_scale: float = 10.0,
):
    """
    Plot a rendezvous trajectory and (optionally) Δv vectors.

    In 2-D mode (MODE_2D=True) the state layout is [x, z, ẋ, ż], so the
    plot axes are x (V-bar) and z (R-bar).
    In 3-D mode the layout is [x, y, z, ẋ, ẏ, ż].

    Parameters
    ----------
    trajectory  : sequence of states (each at least phys_dim long)
    dv_vectors  : applied burns, one per step; shape (N-1, action_dim)
    path        : output file path
    dv_scale    : visual scaling factor for Δv arrows
    """
    if MODE_2D:
        _plot_2d(trajectory, dv_vectors, path, dv_scale)
    else:
        _plot_3d(trajectory, dv_vectors, path, dv_scale)


# ── 2-D plot ─────────────────────────────────────────────────────────────────

def _plot_2d(trajectory, dv_vectors, path, dv_scale):
    traj = np.asarray(trajectory, dtype=np.float64)
    # state = [x, z, ẋ, ż]  →  position columns 0 (x=V-bar) and 1 (z=R-bar)
    x = traj[:, 0]
    z = traj[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(x, z,
            color="#1f5fa8", linewidth=1.2,
            marker="o", markersize=2.5,
            markerfacecolor="#1f5fa8", markeredgewidth=0,
            label="Trajectory")

    ax.scatter(x[0], z[0],
               c="#2c8f3d", marker="s", s=60, zorder=5, label="Start")
    ax.scatter(0.0, 0.0,
               c="#c0392b", marker="*", s=200, zorder=5, label="Target")

    if dv_vectors is not None and len(dv_vectors) > 0:
        dv = np.asarray(dv_vectors, dtype=np.float64)
        # dv columns: [dvx, dvz]
        ax.quiver(
            x[:len(dv)], z[:len(dv)],
            dv[:, 0] * dv_scale, dv[:, 1] * dv_scale,
            angles="xy", scale_units="xy", scale=1,
            color="#e67e22", linewidth=0.8,
            label=r"$\Delta v$ (scaled ${:.0f}\times$)".format(dv_scale),
        )

    ax.set_xlabel(r"$x$ — V-bar [m]")
    ax.set_ylabel(r"$z$ — R-bar [m]")
    ax.set_title("In-plane rendezvous trajectory (2-D CW)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── 3-D plot ─────────────────────────────────────────────────────────────────

def _plot_3d(trajectory, dv_vectors, path, dv_scale):
    traj = np.asarray(trajectory, dtype=np.float64)
    # state = [x, y, z, ẋ, ẏ, ż]
    pos = traj[:, 0:3]

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            color="#1f5fa8", linewidth=1.2,
            marker="o", markersize=2.5,
            markerfacecolor="#1f5fa8", markeredgewidth=0,
            label="Trajectory")

    ax.scatter(*pos[0],  c="#2c8f3d", marker="s",  s=50,  depthshade=False, label="Start")
    ax.scatter(0, 0, 0,  c="#c0392b", marker="*",  s=200, depthshade=False, label="Target")

    dv = None
    if dv_vectors is not None and len(dv_vectors) > 0:
        dv = np.asarray(dv_vectors, dtype=np.float64)
        origins = pos[:len(dv)]
        ax.quiver(
            origins[:, 0], origins[:, 1], origins[:, 2],
            dv[:, 0] * dv_scale, dv[:, 1] * dv_scale, dv[:, 2] * dv_scale,
            color="#e67e22", linewidth=0.8, arrow_length_ratio=0.3,
            label=r"$\Delta v$ (scaled ${:.0f}\times$)".format(dv_scale),
        )

    ax.set_xlabel(r"$x$ — V-bar [m]",  labelpad=10)
    ax.set_ylabel(r"$y$ — H-bar [m]",  labelpad=10)
    ax.set_zlabel(r"$z$ — R-bar [m]",  labelpad=10)

    all_pts = pos if dv is None else np.vstack([pos, pos[:len(dv)] + dv * dv_scale])
    span    = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_r   = max(span.max() / 2.0, 1.0)
    mid     = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_r, mid[0] + max_r)
    ax.set_ylim(mid[1] - max_r, mid[1] + max_r)
    ax.set_zlim(mid[2] - max_r, mid[2] + max_r)
    ax.view_init(elev=22, azim=-60)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.03)
        pane.set_edgecolor("gray")

    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)