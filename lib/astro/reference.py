"""
Reference two-impulse Δv formulas for the CW rendezvous problem (see README).

Reported alongside a trained policy's actual Δv (see libs/evaluate.py) so
the learned solution can be judged against the classical analytic transfer.

All formulas take Δx / Δz as positive magnitudes.
"""

import numpy as np
from lib.astro.dynamics import stm_inplane

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
    """Two V-bar impulses (goal 2, strategy 2). Δv_tot = (ω/2)·Δz.

    CAUTION: only realisable at Δx = (3π/4)·Δz ≈ 2.356·Δz. The "rbar" scenario
    uses Δx = 2·Δz (strategy 1's geometry), so this figure is NOT achievable
    from that start — comparison only, never a reward target (grading against
    an unreachable Δv collapses training). Use
    optimal_two_impulse_stop_dv_per_m() for the real optimum.
    """
    return omega / 2.0 * abs(dz)


# --- True achievable optimum (numeric) --------------------------------------
# The formulas above each assume a fixed maneuver geometry. For grading a
# learned policy we want the best Δv physically achievable from the actual
# start: a fixed-endpoint two-impulse rendezvous is a 1-D family in the coast
# time T (the first impulse is whatever reaches the origin at T from rest, the
# second cancels the arrival velocity), so the optimum is a 1-D minimisation.


def optimal_two_impulse_stop_dv_per_m(
    direction: np.ndarray,
    omega: float,
    t_max_orbits: float = 1.5,
    n_coarse: int = 600,
    n_refine: int = 200,
) -> float:
    """Minimum total Δv of a two-impulse rendezvous that starts AT REST one
    metre from the target along `direction` (unit vector, in-plane [x, z]) and
    ends AT REST at the origin — i.e. the true achievable optimum for that
    geometry — expressed PER METRE of initial distance.

    CW dynamics are linear, so both impulses scale exactly with the initial
    distance: dv_opt(d) = d * optimal_two_impulse_stop_dv_per_m(...). That means
    this only has to be evaluated once per scenario, not per episode.

    Verified against the analytic references: for a pure V-bar displacement it
    reproduces ω/(3π)·Δx to within 0.1% (the two-V-bar transfer really is the
    optimum there, at a ~1 orbit coast). For the "rbar" geometry (Δx = 2·Δz) it
    returns ~0.668x strategy 1, i.e. the optimum genuinely beats the analytic
    R-bar+V-bar maneuver — while strategy 2's figure is unreachable from that
    start (see dv_rbar_strategy_vv).
    """
    direction = np.asarray(direction, dtype=np.float64)
    r0 = direction[:2] / np.linalg.norm(direction[:2])  # unit displacement
    period = 2.0 * np.pi / omega

    def total_dv(t: float) -> float:
        if t <= 0.0:
            return np.inf
        phi = stm_inplane(omega, t)
        m_rr, m_rv = phi[:2, :2], phi[:2, 2:]
        try:
            # Impulse (from rest) that drives the position to zero at time t.
            v0 = -np.linalg.solve(m_rv, m_rr @ r0)
        except np.linalg.LinAlgError:
            return np.inf  # m_rv singular at this coast time
        v_arrival = (phi @ np.concatenate([r0, v0]))[2:]
        return float(np.linalg.norm(v0) + np.linalg.norm(v_arrival))

    # Coarse sweep for the basin, then a local refine around the best sample.
    grid = np.linspace(1e-3 * period, t_max_orbits * period, n_coarse)
    values = np.array([total_dv(t) for t in grid])
    i = int(np.argmin(values))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, n_coarse - 1)]
    fine = np.linspace(lo, hi, n_refine)
    return float(min(values[i], min(total_dv(t) for t in fine)))


# --- Per-direction optimum table (for the "random" scenario) --------------
# In the "random" scenario the initial displacement can point ANY direction on
# the circle, and the achievable optimum varies ~23x with that angle (cheapest
# along-track / V-bar, ~23x costlier near radial / R-bar). So dv_opt has to be
# looked up per episode from the sampled angle rather than precomputed once.
# Evaluating optimal_two_impulse_stop_dv_per_m() every reset would be far too
# slow, so we build a fine table over the angle once and interpolate. By the
# point-reflection symmetry of the CW dynamics (negate position+velocity) the
# optimum has period pi, so the table only spans [0, pi). Memoized per (omega,
# resolution) so the parallel sub-envs of a DummyVecEnv share one build.
_DV_OPT_TABLE_CACHE: dict = {}


def dv_opt_per_m_table(omega: float, n_angles: int = 180):
    """(angles, values) table of optimal_two_impulse_stop_dv_per_m over the
    in-plane direction angle in [0, pi). Built once per (omega, n_angles)."""
    key = (round(float(omega), 12), int(n_angles))
    if key not in _DV_OPT_TABLE_CACHE:
        angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
        values = np.array([
            optimal_two_impulse_stop_dv_per_m(
                np.array([np.cos(a), np.sin(a)]), omega
            )
            for a in angles
        ])
        _DV_OPT_TABLE_CACHE[key] = (angles, values)
    return _DV_OPT_TABLE_CACHE[key]


def dv_opt_per_m_lookup(direction: np.ndarray, table) -> float:
    """Interpolate the per-metre optimum for an in-plane unit `direction`
    ([x, z]) from a table built by dv_opt_per_m_table(). The angle is folded
    into [0, pi) (period-pi by the point-reflection symmetry)."""
    angles, values = table
    a = np.arctan2(float(direction[1]), float(direction[0])) % np.pi
    return float(np.interp(a, angles, values, period=np.pi))
