"""
Plot a saved training history.npz.

    python scripts/plot.py tmp/<run-tag>
    python scripts/plot.py tmp/<run-tag> --panels     # individual panels too
    python scripts/plot.py tmp/a tmp/b --compare      # overlay runs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lib.plots.diagnostics import build_diagnostics_figure, moving_average, rolling_dock_rate
from lib.plots.style import PALETTE, save, style_axes, use_style

PANELS = [
    ("rewards", "Total reward", "reward"),
    ("dv_ratio", r"Fuel vs. optimum", r"$\Delta v / \Delta v_{opt}$"),
    ("steps", "Episode length", "steps"),
    ("r_fuel", "Fuel reward", "reward"),
    ("r_term", "Terminal reward", "reward"),
    ("noise_std", "Exploration noise", "std"),
]


def resolve(path: Path) -> Path:
    if path.is_dir():
        found = sorted(path.rglob("history.npz"))
        if not found:
            sys.exit(f"No history.npz under {path}")
        return found[0]
    return path


def run_root(npz: Path) -> Path:
    """history.npz lives at <run>/model/, so the run root is two levels up."""
    return npz.parent.parent if npz.parent.name == "model" else npz.parent


def load(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def scenario_of(npz: Path) -> str:
    """From status.json beside the history, falling back to the run name."""
    status = npz.parent / "status.json"
    if status.exists():
        try:
            return json.loads(status.read_text())["scenario"]
        except (ValueError, KeyError):
            pass
    for name in ("vbar", "rbar", "random"):
        if any(name in part for part in npz.parts):
            return name
    return "unknown"


def plot_panels(history, out_dir: Path, stem: str):
    use_style()
    written = []
    for key, title, ylabel in PANELS:
        if key not in history or not len(history[key]):
            continue
        data = np.asarray(history[key], dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data, color=PALETTE[0], linewidth=0.6, alpha=0.25)
        smooth = moving_average(data)
        ax.plot(np.arange(len(data) - len(smooth), len(data)), smooth,
                color=PALETTE[0], linewidth=1.5, label=title)
        if key == "dv_ratio":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="optimum")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        style_axes(ax)
        written.append(save(fig, out_dir / f"{stem}_{key}.png"))
        plt.close(fig)
    return written


def plot_compare(histories, labels, out_path: Path):
    """Smoothed reward and dock rate for several runs on shared axes."""
    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for h, label, color in zip(histories, labels, PALETTE):
        rewards = np.asarray(h["rewards"], dtype=float)
        smooth = moving_average(rewards)
        ax1.plot(np.arange(len(rewards) - len(smooth), len(rewards)), smooth,
                 color=color, linewidth=1.5, label=label)
        ax2.plot(rolling_dock_rate(h["docked"]), color=color, linewidth=1.5, label=label)
    ax1.set_title("Total reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("reward")
    ax2.set_title("Dock rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel(r"\%" if plt.rcParams["text.usetex"] else "%")
    ax2.set_ylim(0, 100)
    style_axes(ax1)
    style_axes(ax2)
    save(fig, out_path)
    plt.close(fig)
    return out_path


def summarize(history, label):
    docked = np.asarray(history["docked"], dtype=float)
    ratio = np.asarray(history["dv_ratio"], dtype=float)
    tail = slice(-max(1, len(docked) // 10), None)      # last 10% of episodes
    finite = ratio[tail][np.isfinite(ratio[tail])]
    print(f"{label}")
    print(f"  episodes        : {len(docked)}")
    print(f"  dock rate (all) : {100 * docked.mean():.1f}%")
    print(f"  dock rate (last): {100 * docked[tail].mean():.1f}%")
    if finite.size:
        print(f"  dv/opt   (last) : median {np.median(finite):.2f}x")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="history.npz files or run dirs")
    p.add_argument("--out", type=Path, default=None, help="Output dir (default: alongside)")
    p.add_argument("--panels", action="store_true", help="Also write individual panels")
    p.add_argument("--compare", action="store_true", help="Overlay all inputs on one figure")
    args = p.parse_args()

    paths = [resolve(q) for q in args.paths]
    histories = [load(q) for q in paths]
    for q, h in zip(paths, histories):
        summarize(h, str(q))

    out_dir = args.out or (run_root(paths[0]) / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare and len(paths) > 1:
        labels = [q.parent.name for q in paths]
        print("wrote", plot_compare(histories, labels, out_dir / "comparison.png"))
        return

    for q, h in zip(paths, histories):
        diag = args.out or (run_root(q) / "diagnostics")
        (diag / "plots").mkdir(parents=True, exist_ok=True)
        target = diag / "diagnostics.png"
        build_diagnostics_figure(h, scenario_of(q), target)
        print("wrote", target)
        if args.panels:
            for w in plot_panels(h, diag / "plots", run_root(q).name):
                print("wrote", w)


if __name__ == "__main__":
    main()
