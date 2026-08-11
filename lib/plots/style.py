"""Shared matplotlib style: LaTeX/Computer Modern serif with a safe fallback."""

import matplotlib as mpl
import matplotlib.pyplot as plt

COLOR_1 = "#0B4A8A"   # blue
COLOR_2 = "#4CAF50"   # green
COLOR_3 = "#FFD700"   # gold
COLOR_4 = "#E74C3C"   # red
COLOR_5 = "#C70039"   # crimson
PALETTE = [COLOR_1, COLOR_4, COLOR_2, COLOR_5, COLOR_3]

_CONFIGURED = False


def use_style():
    """Idempotent. Tries real LaTeX, falls back to mathtext Computer Modern."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    try:
        from matplotlib import texmanager
        texmanager.TexManager().make_dvi("test", 10)
        mpl.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Computer Modern Roman"]})
    except Exception:
        mpl.rcParams.update({
            "text.usetex": False, "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        })
    mpl.rcParams.update({
        "font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
        "legend.fontsize": 14, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "figure.constrained_layout.use": True, "savefig.dpi": 300,
    })
    _CONFIGURED = True


def style_axes(ax, legend=True, equal=False):
    """Inward ticks on all four sides, dashed grid, boxed legend."""
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, bottom=True, left=True, right=True)
    ax.tick_params(which="major", length=8, width=0.8)
    ax.tick_params(which="minor", length=4, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if legend and (ax.get_legend_handles_labels()[0]):
        ax.legend(loc="upper left", frameon=True, edgecolor="black", framealpha=1.0,
                  borderaxespad=0.2, labelspacing=0.2, handletextpad=0.4)
    return ax


def new_figure(nrows=1, ncols=1, figsize=(6, 4), **kw):
    use_style()
    return plt.subplots(nrows, ncols, figsize=figsize, **kw)


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
