"""
Plot the trajectory of a saved model on a single deterministic episode.

The training run drops ep_*.png for the NOISY behaviour policy; this rolls the
checkpoint out the way evaluate does (deterministic, no noise, no curriculum)
and draws the same figure, so the picture in the report matches the dv/opt that
scripts/eval.py reports.

    python scripts/plot_run.py out/vbar_specialist_2/model/vbar_td3.zip --scenario vbar
    python scripts/plot_run.py out/vbar_specialist_2/model/best_model.zip --sign -1
    python scripts/plot_run.py MODEL.zip --scenario random --angle 30 --distance 500

Every impulse is drawn, including the first one applied at the start point and
the terminal brake: the start state is seeded before the loop and each burn is
anchored to the state it was applied FROM (--min-dv 0 by default, so nothing is
filtered out of the arrow set).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse

import numpy as np
from stable_baselines3 import TD3

from config import (
    ACTION_IMPULSE_DIM, ENV_ANGLE_CURRICULUM_MAX_DEG, ENV_CURRICULUM_MAX_DISTANCE,
    MAX_STEPS, OMEGA,
)
from lib.rl.env import CWRendezvousEnv
from lib.rl.obs import NormalizedObsEnv
from lib.rl.symmetry import CanonicalizeDirectionEnv
from lib.plots.trajectory import plot_trajectory


def build_env(scenario, distance, half_width_deg):
    """Identical wrapper stack to training/eval: the policy only ever saw the
    canonicalized x>=0 view, then normalized."""
    raw = CWRendezvousEnv(
        omega=OMEGA,
        scenario=scenario,
        curriculum_enabled=False,
        curriculum_max_distance=distance,
        angle_half_width_deg=half_width_deg,
    )
    return raw, NormalizedObsEnv(CanonicalizeDirectionEnv(raw))


def rollout(model, env, options, seed):
    """Returns (states, actions, info). Mirrors EpisodeLogger._on_step: the
    reset state is seeded first so the impulse commanded on step 0 has a
    position to be anchored at, then each step's substates are appended and the
    applied impulse is written back onto the state it was fired from."""
    obs, info = env.reset(seed=seed, options=options)
    states = [np.asarray(info["state"], dtype=float).copy()]
    actions = [np.zeros(ACTION_IMPULSE_DIM)]

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        substates = info.get("substates") or [info["state"]]
        for s in substates:
            states.append(np.asarray(s, dtype=float).copy())
            actions.append(np.zeros(ACTION_IMPULSE_DIM))
        applied = np.asarray(info.get("applied_action", np.zeros(ACTION_IMPULSE_DIM)))
        actions[-len(substates) - 1] = applied.copy()   # anchored at the pre-burn state

        if terminated or truncated:
            break
    return np.array(states), np.array(actions), info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path, help="Path to a saved .zip")
    p.add_argument("--scenario", choices=["vbar", "rbar", "random"], default="vbar")
    p.add_argument("--distance", type=float, default=ENV_CURRICULUM_MAX_DISTANCE,
                   help="Fixed start distance [m] (curriculum disabled).")
    p.add_argument("--sign", type=float, default=1.0,
                   help="(vbar/rbar) Which side of the target to start on.")
    p.add_argument("--angle", type=float, default=None,
                   help="(random) Start angle in DEGREES.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-dv", type=float, default=0.0,
                   help="Hide impulses smaller than this [m/s]. 0 draws them all.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG (default: <run>/episodes/deterministic_<scenario>.png)")
    args = p.parse_args()

    model = TD3.load(str(args.model), device=args.device)
    raw, env = build_env(args.scenario, args.distance,
                         ENV_ANGLE_CURRICULUM_MAX_DEG)

    options = {}
    if args.scenario == "random":
        if args.angle is not None:
            options["angle"] = np.deg2rad(args.angle)
    else:
        options["sign"] = args.sign

    states, actions, info = rollout(model, env, options, args.seed)
    dv_opt = raw.dv_opt
    ratio = info["dv_used"] / dv_opt if dv_opt else float("nan")
    docked = bool(info.get("docked", False))
    burns = int(np.count_nonzero(np.linalg.norm(actions, axis=1) > args.min_dv))

    out = args.out
    if out is None:
        run_root = args.model.parent.parent if args.model.parent.name == "model" \
            else args.model.parent
        out = run_root / "episodes" / f"deterministic_{args.scenario}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    title = (f"{args.scenario} deterministic — "
             rf"$\Delta v$/opt {ratio:.2f}x, {burns} impulses"
             + ("" if docked else ", NOT docked"))
    plot_trajectory(states, actions, str(out),
                    min_dv_display=args.min_dv, title=title)

    print(f"model         : {args.model}")
    print(f"docked        : {docked}   final distance {info['distance']:.2f} m   "
          f"arrival speed {info['vel_error']:.5f} m/s")
    print(f"dv used / opt : {info['dv_used']:.5f} / {dv_opt:.5f} = {ratio:.2f}x")
    print(f"impulses drawn: {burns}  (first at "
          f"[{states[0][0]:.1f}, {states[0][1]:.1f}] m, "
          f"|dv| = {np.linalg.norm(actions[0]):.5f} m/s)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
