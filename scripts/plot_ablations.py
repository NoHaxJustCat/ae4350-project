"""
Ablation study figures: one 2x2 panel per hyperparameter group.

    python scripts/plot_ablations.py                 # all groups
    python scripts/plot_ablations.py --group gamma   # just one

Reads the self-contained out/ablations/<name>.npz written by
collect_ablations.py, so plots can be redrawn without retraining. Figures go to
out/ablations/figures/.

Each figure holds one group (all its variants plus the nominal) and four
panels: reward, fuel, dock rate, critic loss. The nominal keeps the same colour
in every figure so it reads as the fixed reference.

x axis is TIMESTEPS, not episode index. Episode counts differ ~4x between
configs (12.6k for no_curriculum, 47.7k for net_32) purely because a failing
policy ends episodes sooner, so plotting against episode index would compare
different amounts of training. Cumulative episode length recovers the timestep
each episode ended at, and every run then spans the same 0..100k.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUT_DIR
from lib.plots.style import COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5, save, style_axes, use_style

REPO = Path(__file__).resolve().parents[1]
# out/ is tracked, tmp/ is gitignored, so the study lives in out/ once it is
# worth keeping. Fall back to tmp/ for a study that has not been promoted yet.
IN_DIR = REPO / "out" / "ablations"
if not IN_DIR.exists():
    IN_DIR = Path(OUT_DIR) / "ablations"
OUT_SUBDIR = "figures"

# Fixed across every figure so the fuel panels are comparable between groups,
# and so the optimum line does not land on the x axis. The smoothed series span
# 1.03 (net_128) to 206.6 (no_curriculum); the lower bound sits below 1.0 to
# leave visible clearance under the optimum.
FUEL_YLIM = (0.5, 300.0)

# The nominal is pinned to COLOR_1 in every figure; variants draw from the rest
# in order. Gold is last because it is the weakest on white.
NOMINAL_COLOR = COLOR_1
VARIANT_COLORS = [COLOR_4, COLOR_2, COLOR_5, COLOR_3]

# group -> (figure title, [(npz name, legend label), ...]) with the nominal
# first so it is drawn underneath the variants.
GROUPS = {
    "gamma": ("Discount factor", [
        ("nominal", "nominal"), ("gamma_095", r"$\gamma$ 0.95"),
        ("gamma_090", r"$\gamma$ 0.90"), ("gamma_080", r"$\gamma$ 0.80"),
    ]),
    "lr": ("Learning rate", [
        ("nominal", "nominal"), ("lr_low", r"$10^{-5}$"), ("lr_high", r"$10^{-3}$"),
    ]),
    "net": ("Network width", [
        ("nominal", "nominal"), ("net_32", "32"), ("net_64", "64"), ("net_128", "128"),
    ]),
    "algo": ("Algorithm", [
        ("nominal", "nominal"), ("sac", "SAC"), ("ddpg", "DDPG"),
    ]),
    "components": ("Components", [
        ("nominal", "nominal"), ("no_replay", "no replay"),
        ("no_curriculum", "no curriculum"), ("no_encoder", "no encoder"),
    ]),
}

MAX_POINTS = 3000
BOX_ASPECT = 0.72      # identical for all four panels, per the spec
# Font sizes are fixed in rcParams, so a larger figure buys panel area rather
# than bigger text -- the labels get relatively smaller and the curves clearer.
FIGSIZE = (14.5, 10.0)


def rolling_nanmean(y, window):
    """Centred-trailing rolling mean that ignores NaN.

    Critic and actor loss are NaN until learning_starts, and np.convolve would
    propagate a single NaN across a whole window, blanking the early curve.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    filled = np.where(ok, y, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    ccnt = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    hi = np.arange(1, y.size + 1)
    lo = np.maximum(0, hi - window)
    total, count = csum[hi] - csum[lo], ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def decimate(x, y):
    """Thin to MAX_POINTS; a 300 dpi panel cannot resolve 40k points anyway."""
    if x.size <= MAX_POINTS:
        return x, y
    idx = np.arange(0, x.size, int(np.ceil(x.size / MAX_POINTS)))
    return x[idx], y[idx]


def load(name):
    path = IN_DIR / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        h = {k: data[k] for k in data.files if k != "params_json"}
        h["params"] = json.loads(str(data["params_json"]))
    # Episode end times: episode lengths accumulate to the timestep count.
    h["t"] = np.cumsum(np.asarray(h["steps"], dtype=float))
    return h


# "seaborn-dark-palette" was renamed in matplotlib 3.6; the old name raises on
# 3.11. Try the requested name first so this keeps working on either version.
SEABORN_STYLES = ("seaborn-dark-palette", "seaborn-v0_8-dark-palette")


def apply_style():
    """Seaborn palette underneath, the project's LaTeX serif on top.

    Order matters and this must run exactly once: plt.style.use resets every
    rcParam, so calling it after use_style() would throw the Computer Modern
    font settings away. use_style() is itself guarded against re-entry, so it
    could not put them back.
    """
    for name in SEABORN_STYLES:
        if name in plt.style.available:
            plt.style.use(name)
            break
    else:
        print(f"WARNING: none of {SEABORN_STYLES} available; using defaults")
    use_style()


def pct_label():
    return r"\%" if plt.rcParams["text.usetex"] else "%"


def draw(ax, h, color, label, key, window, scale=None, percent=False):
    y = np.asarray(h[key], dtype=float)
    if percent:
        y = 100.0 * y
    x, ys = decimate(h["t"], rolling_nanmean(y, window))
    ax.plot(x, ys, color=color, linewidth=1.6, label=label)
    if scale:
        ax.set_yscale(scale)


def panel_fuel_reference(ax):
    """The 1x line is the whole point of the fuel panel: it marks the
    achievable optimum, so a curve sitting on it has nothing left to gain."""
    return ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
                      label="optimum")


def thousands(x, _pos):
    """40000 -> 40k. Four panels of six-digit ticks crowd the axes."""
    if x == 0:
        return "0"
    return f"{x / 1000:g}k"


def make_figure(group, window, out_dir):
    title, members = GROUPS[group]
    loaded = [(name, lab, load(name)) for name, lab in members]
    missing = [name for name, _, h in loaded if h is None]
    loaded = [(name, lab, h) for name, lab, h in loaded if h is not None]
    if len(loaded) < 2:
        print(f"  {group:<12} SKIPPED (need 2 runs, have {len(loaded)})")
        return None

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    (ax_r, ax_dv), (ax_dk, ax_cl) = axes

    variant = iter(VARIANT_COLORS)
    for name, lab, h in loaded:
        color = NOMINAL_COLOR if name == "nominal" else next(variant)
        draw(ax_r, h, color, lab, "rewards", window)
        draw(ax_dv, h, color, lab, "dv_ratio", window, scale="log")
        draw(ax_dk, h, color, lab, "docked", window, percent=True)
        draw(ax_cl, h, color, lab, "critic_loss", window, scale="log")

    panel_fuel_reference(ax_dv)

    ax_r.set_title("Reward")
    ax_r.set_ylabel("Reward")
    ax_dv.set_title("Fuel")
    ax_dv.set_ylabel(r"$\Delta v / \Delta v_{opt}$")
    ax_dv.set_ylim(*FUEL_YLIM)
    ax_dk.set_title("Dock rate")
    ax_dk.set_ylabel(f"Docked [{pct_label()}]")
    ax_dk.set_ylim(0, 100)
    ax_cl.set_title("Critic loss")
    ax_cl.set_ylabel("Loss")

    # Every panel spans the full run, so the four are read on one x scale. The
    # critic-loss curve simply starts where gradient updates do.
    t_end = max(float(h["t"][-1]) for _, _, h in loaded)
    for ax in axes.flat:
        ax.set_xlabel("Timesteps")
        ax.set_xlim(0, t_end)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))
        ax.set_box_aspect(BOX_ASPECT)
        style_axes(ax)          # legend upper left, per style_axes
    fig.suptitle(title)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = save(fig, out_dir / f"{group}.png")
    plt.close(fig)
    names = ", ".join(lab for _, lab, _ in loaded)
    note = f"  (missing: {', '.join(missing)})" if missing else ""
    print(f"  {group:<12} {len(loaded)} runs [{names}]{note}")
    return path


def main():
    global IN_DIR          # must precede the f-string default below
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=sorted(GROUPS), default=None,
                   help="Only this group (default: all).")
    p.add_argument("--window", type=int, default=400,
                   help="Moving-average window in episodes.")
    p.add_argument("--in-dir", type=Path, default=None,
                   help=f"Where the collected npz live (default {IN_DIR}).")
    p.add_argument("--out", type=Path, default=None,
                   help="Figure directory (default <in-dir>/figures).")
    args = p.parse_args()
    if args.in_dir is not None:
        IN_DIR = args.in_dir
    if args.out is None:
        args.out = IN_DIR / OUT_SUBDIR

    if not IN_DIR.exists():
        sys.exit(f"No {IN_DIR}. Run scripts/collect_ablations.py first.")

    apply_style()

    groups = [args.group] if args.group else list(GROUPS)
    print(f"Writing to {args.out} (window {args.window} episodes)")
    written = [make_figure(g, args.window, args.out) for g in groups]
    written = [w for w in written if w]
    print(f"\n{len(written)} figures written")


if __name__ == "__main__":
    main()
