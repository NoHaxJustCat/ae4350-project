"""
Batch, training-free, noise-free evaluation of a saved model.

Rolls a model out deterministically (model.predict(deterministic=True), no
exploration noise, no gradient updates, no curriculum) for N episodes and
reports the distribution of outcomes rather than the single sample vbar.py /
rbar.py print. This is the tool to answer "is this checkpoint actually any
good", which the training log cannot: that log reports the NOISY behaviour
policy, and for a solution as narrow as the V-bar transfer the two differ a
lot (see the --angle-sweep output for how fast quality falls off-axis).

Usage:
    python evaluate_model.py MODEL.zip --scenario vbar
    python evaluate_model.py MODEL.zip --scenario random -n 200
    python evaluate_model.py MODEL.zip --scenario random --angle 0
    python evaluate_model.py MODEL.zip --scenario random --angle-sweep
    python evaluate_model.py MODEL.zip --scenario random --half-width 2
    python evaluate_model.py MODEL.zip --scenario vbar --distance 500

Notes:
  * dv/opt is the headline number: 1.00 == matched the best two-impulse
    maneuver that physically exists for that start (libs/reference.py).
  * ALWAYS read it next to arrival speed. A policy that skips its terminal
    brake scores below 1.00 while performing a fly-through, not a rendezvous
    -- the analytic references all assume a full stop, so only a near-zero
    arrival speed is a like-for-like comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import sys

import numpy as np
from stable_baselines3 import TD3

from config import (
    ENV_ANGLE_CURRICULUM_MAX_DEG, ENV_CURRICULUM_MAX_DISTANCE, MAX_STEPS, OMEGA, OUT_DIR,
)
from lib.rl.env import CWRendezvousEnv
from lib.rl.obs import NormalizedObsEnv
from lib.rl.symmetry import CanonicalizeDirectionEnv
from lib.astro.reference import (
    dv_vbar_two_impulse_vv, dv_vbar_two_impulse_rr,
    dv_rbar_strategy_rv, dv_rbar_strategy_vv,
)


def find_latest_model(scenario: str) -> str:
    """Most recent final model for `scenario` under tmp/."""
    found = sorted(Path(OUT_DIR).rglob(f"{scenario}_td3.zip"),
                   key=lambda p: p.stat().st_mtime)
    if not found:
        sys.exit(f"No {scenario}_td3.zip under {OUT_DIR}/ - train one, or pass a path.")
    return str(found[-1])


def build_env(scenario: str, distance: float, half_width_deg: float):
    """Must match training.py's wrapper stack exactly: the policy only ever
    saw the canonicalized x>=0 view, then normalized."""
    raw = CWRendezvousEnv(
        omega=OMEGA,
        scenario=scenario,
        curriculum_enabled=False,                    # no ramp: start at `distance`
        curriculum_max_distance=distance,
        angle_half_width_deg=half_width_deg,
    )
    return raw, NormalizedObsEnv(CanonicalizeDirectionEnv(raw))


def rollout(model, raw, env, options=None, seed=None):
    obs, info = env.reset(seed=seed, options=options or {})
    start = info["state"][:2].copy()
    dv_opt = raw.dv_opt
    coasts, burns = [], 0
    truncated = False
    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        coasts.append(info["coast_units"])
        if info["delta_v"] > 0:
            burns += 1
        if terminated or truncated:
            break
    docked = bool(info.get("docked", False))
    return {
        "docked": docked,
        "cause": "dock" if docked else ("timeout" if truncated else "out-of-bounds"),
        "ratio": info["dv_used"] / dv_opt if dv_opt else float("nan"),
        "dv_used": info["dv_used"],
        "dv_opt": dv_opt,
        "arrival_speed": info["vel_error"],
        "final_distance": info["distance"],
        "steps": step + 1,
        "burns": burns,
        "coasts": coasts,
        "angle_deg": float(np.degrees(np.arctan2(start[1], start[0]))),
        "start": start,
    }


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def summarize(results, label=""):
    n = len(results)
    docked = [r for r in results if r["docked"]]
    print(f"\n=== {label} ===" if label else "")
    print(f"episodes            : {n}")
    print(f"dock rate           : {100 * len(docked) / n:.1f}%  ({len(docked)}/{n})")
    for cause in ("out-of-bounds", "timeout"):
        k = sum(1 for r in results if r["cause"] == cause)
        if k:
            print(f"  failed by {cause:<14}: {k} ({100 * k / n:.1f}%)")
    if not docked:
        # Still informative: how close did the failures get, and how much did
        # they burn getting there?
        d = [r["final_distance"] for r in results]
        print(f"final distance      : median {np.median(d):.1f} m  "
              f"[min {np.min(d):.1f}, max {np.max(d):.1f}]")
        print(f"dv/opt (all eps)    : median {np.median([r['ratio'] for r in results]):.4f}x")
        return

    ratios = [r["ratio"] for r in docked]
    vels = [r["arrival_speed"] for r in docked]
    steps = [r["steps"] for r in docked]
    print(f"--- docked episodes only ---")
    print(f"dv/opt              : median {np.median(ratios):8.4f}x   "
          f"p10 {_pct(ratios, 10):.4f}x  p90 {_pct(ratios, 90):.4f}x   "
          f"[min {np.min(ratios):.4f}, max {np.max(ratios):.4f}]")
    dvs = [r["dv_used"] for r in docked]
    print(f"dv used [m/s]       : median {np.median(dvs):.4f}   "
          f"p10 {_pct(dvs, 10):.4f}  p90 {_pct(dvs, 90):.4f}   "
          f"[min {np.min(dvs):.4f}, max {np.max(dvs):.4f}]")
    print(f"arrival speed [m/s] : median {np.median(vels):.5f}   "
          f"p90 {_pct(vels, 90):.5f}   max {np.max(vels):.5f}")
    print(f"arrival speed / dv_opt : median {np.median([r['arrival_speed'] / r['dv_opt'] for r in docked]):.3f}"
          f"   (>0.05 means the brake is being skipped -- ratio is NOT comparable"
          f" to the analytic references)")
    print(f"steps / burns       : {np.median(steps):.1f} / {np.median([r['burns'] for r in docked]):.1f}")
    allc = [c for r in docked for c in r["coasts"] if c]
    if allc:
        print(f"coast units         : median {np.median(allc):.0f}  "
              f"[min {np.min(allc)}, max {np.max(allc)}]")


def print_references(results, scenario):
    docked = [r for r in results if r["docked"]]
    if not docked or scenario == "random":
        return
    dx = np.median([abs(r["start"][0]) for r in docked])
    dz = np.median([abs(r["start"][1]) for r in docked])
    dv = np.median([r["dv_used"] for r in docked])
    print(f"--- vs. the analytic references (CLAUDE.md), median dv_used = {dv:.5f} m/s ---")
    if scenario == "vbar":
        for name, val in [("two V-bar impulses", dv_vbar_two_impulse_vv(dx, OMEGA)),
                          ("two R-bar impulses", dv_vbar_two_impulse_rr(dx, OMEGA))]:
            print(f"  {name:<22}: {val:.5f} m/s  ({dv / val:.4f}x)")
    else:
        for name, val in [("R-bar + V-bar impulse", dv_rbar_strategy_rv(dz, OMEGA)),
                          ("two V-bar impulses*", dv_rbar_strategy_vv(dz, OMEGA))]:
            print(f"  {name:<22}: {val:.5f} m/s  ({dv / val:.4f}x)")
        print("  * not reachable from this scenario's geometry -- comparison only")


def angle_sweep(model, args, step_deg=10):
    """Per-angle deterministic sweep -- the most useful view for `random`,
    since quality varies enormously with direction."""
    raw, env = build_env("random", args.distance, ENV_ANGLE_CURRICULUM_MAX_DEG)
    print(f"\n=== per-angle sweep (deterministic, distance {args.distance:.0f} m) ===")
    print(f"{'angle':>6} {'result':>14} {'dv/opt':>10} {'arr.speed':>10} {'steps':>6} {'coasts'}")
    docked = 0
    angles = np.arange(0, 360, step_deg)
    for deg in angles:
        r = rollout(model, raw, env, options={"angle": np.deg2rad(float(deg))}, seed=args.seed)
        docked += r["docked"]
        print(f"{deg:6.0f} {r['cause']:>14} {r['ratio']:10.4f} {r['arrival_speed']:10.5f} "
              f"{r['steps']:6d} {r['coasts'][:6]}")
    print(f"\ndock rate over the circle: {docked}/{len(angles)} = {100 * docked / len(angles):.0f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", default=None,
                   help="Path to a saved .zip. Omit to use the most recent model "
                        "under trained/ for --scenario.")
    p.add_argument("--scenario", choices=["vbar", "rbar", "random"], default="vbar")
    p.add_argument("-n", "--episodes", type=int, default=100)
    p.add_argument("--distance", type=float, default=ENV_CURRICULUM_MAX_DISTANCE,
                   help="Fixed start distance [m] (curriculum is disabled).")
    p.add_argument("--angle", type=float, default=None,
                   help="(random) Force this start angle in DEGREES for every episode.")
    p.add_argument("--half-width", type=float, default=ENV_ANGLE_CURRICULUM_MAX_DEG,
                   help="(random) Sample angles from +-this many degrees around the "
                        "V-bar axis. Default 90 = the full circle. Set to what a "
                        "training run's angle curriculum had reached to reproduce it.")
    p.add_argument("--angle-sweep", action="store_true",
                   help="(random) Also print a deterministic per-angle table.")
    p.add_argument("--sweep-step", type=float, default=10.0,
                   help="(random, --angle-sweep) Angle step in degrees.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model_path = args.model or find_latest_model(args.scenario)
    print(f"model    : {model_path}")
    model = TD3.load(model_path, device=args.device)
    print(f"scenario : {args.scenario}   distance: {args.distance:.0f} m   "
          f"steps trained: {model.num_timesteps}")
    print("mode     : deterministic, NO exploration noise, NO training")

    if args.scenario != "random" and (args.angle is not None or args.angle_sweep):
        sys.exit("--angle / --angle-sweep only apply to --scenario random "
                 "(vbar/rbar have a fixed displacement direction).")

    half_width = args.half_width
    options = {}
    if args.scenario == "random":
        if args.angle is not None:
            options = {"angle": np.deg2rad(args.angle)}
            print(f"angle    : forced to {args.angle:.2f} deg")
        else:
            print(f"angle    : sampled from +-{half_width:.1f} deg around V-bar")

    raw, env = build_env(args.scenario, args.distance, half_width)
    results = []
    for i in range(args.episodes):
        opts = dict(options)
        if args.scenario in ("vbar", "rbar"):
            opts["sign"] = 1.0 if i % 2 == 0 else -1.0   # both signs, evenly
        results.append(rollout(model, raw, env, options=opts,
                               seed=args.seed + i if i == 0 else None))

    summarize(results, label=f"{args.scenario} — {args.episodes} deterministic episodes")
    print_references(results, args.scenario)

    if args.scenario == "random" and args.angle is None:
        docked = [r for r in results if r["docked"]]
        if docked or results:
            print("\n--- by |angle from V-bar| ---")
            print(f"{'bin':>12} {'n':>5} {'dock%':>7} {'median dv/opt':>15}")
            off = np.array([abs(((r["angle_deg"] + 90) % 180) - 90) for r in results])
            edges = [0, 1, 2, 5, 10, 20, 45, 90.001]
            for a, b in zip(edges[:-1], edges[1:]):
                m = (off >= a) & (off < b)
                if not m.any():
                    continue
                sub = [r for r, k in zip(results, m) if k]
                dk = [r for r in sub if r["docked"]]
                med = np.median([r["ratio"] for r in dk]) if dk else float("nan")
                print(f"{a:5.0f}-{b:<6.0f} {len(sub):5d} {100 * len(dk) / len(sub):7.1f} {med:15.4f}")

    if args.angle_sweep:
        angle_sweep(model, args, step_deg=args.sweep_step)


if __name__ == "__main__":
    main()
