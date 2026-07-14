"""
Reference two-impulse Δv formulas from the project objective (CLAUDE.md).

Reported alongside a trained policy's actual Δv (see libs/evaluate.py) so
the learned solution can be judged against the classical analytic transfer.

All formulas take Δx / Δz as positive magnitudes.
"""

import numpy as np

# These output the TOTAL Delta V


def dv_vbar_two_impulse_vv(dx: float, omega: float) -> float:
    """Two V-bar impulses, Δx displacement along V-bar (goal 1)."""
    return omega / (3.0 * np.pi) * abs(dx)


def dv_vbar_two_impulse_rr(dx: float, omega: float) -> float:
    """Two R-bar impulses, Δx displacement along V-bar (goal 1 comparison)."""
    return omega / 2.0 * abs(dx)


def dv_rbar_strategy_rv(dz: float, omega: float) -> float:
    """R-bar impulse + V-bar impulse (goal 2, strategy 1). Δv_tot = 3·ω·Δz."""
    return 3.0 * omega * abs(dz)


def dv_rbar_strategy_vv(dz: float, omega: float) -> float:
    """Two V-bar impulses (goal 2, strategy 2). Δv_tot = (ω/2)·Δz."""
    return omega / 2.0 * abs(dz)
