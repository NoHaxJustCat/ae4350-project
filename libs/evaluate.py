"""
Deterministic evaluation rollouts for a trained TD3 model, plus comparison
against the classical reference Δv formulas from CLAUDE.md.
"""

from pathlib import Path

import numpy as np

from libs.constants import OMEGA, MAX_STEPS
from libs.env import CWRendezvousEnv
from libs.normalization import NormalizedObsEnv
from libs.symmetry import CanonicalizeDirectionEnv
from libs.reference import (
    dv_vbar_two_impulse_vv,
    dv_vbar_two_impulse_rr,
    dv_rbar_strategy_rv,
    dv_rbar_strategy_vv,
)


def find_latest_model(scenario: str, trained_dir: str = "trained") -> str:
    """Most recently modified '{scenario}_td3.zip' anywhere under
    trained_dir/ — every training.py run now lives in its own
    trained/<session_id>/<run_tag>/ folder (see training.py) rather than a
    shared flat path, so vbar.py/rbar.py find "whatever finished most
    recently" instead of a hardcoded location."""
    candidates = list(Path(trained_dir).rglob(f"{scenario}_td3.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No {scenario}_td3.zip found under {trained_dir}/ — train one first, "
            f"or pass an explicit model path as the first argument."
        )
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def run_episode(model, scenario: str, sign: float, max_steps: int = MAX_STEPS) -> dict:
    """Roll out `model` deterministically on `scenario`, forcing the initial
    displacement to the given sign (+1 or -1) instead of the random draw
    used during training."""
    raw_env = CWRendezvousEnv(omega=OMEGA, scenario=scenario, curriculum_enabled=False)
    # Must match the training wrapper stack exactly — the policy only ever
    # learned the x>=0 canonical view (see libs/symmetry.py).
    env = NormalizedObsEnv(CanonicalizeDirectionEnv(raw_env))
    obs, _ = env.reset(options={"sign": sign})

    states, actions, rewards = [], [], []
    info, step = {}, -1
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(info["state"].copy())
        actions.append(info["applied_action"].copy())
        rewards.append(reward)
        if terminated or truncated:
            break

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "info": info,
        "steps": step + 1,
        "docked": info.get("docked", False),
        "final_distance": info.get("distance", np.nan),
        "total_dv": info.get("dv_used", 0.0),
        "start_state": states[0] if states else None,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
    }


def print_summary(label: str, result: dict, scenario: str):
    r = result
    print(f"\n=== {label} ===")
    print(f"Steps: {r['steps']}  docked: {r['docked']}  "
          f"final distance: {r['final_distance']:.4f} m  total reward: {r['total_reward']:.2f}")
    print(f"Total dv used: {r['total_dv']:.5f} m/s")

    start = r["start_state"]
    if start is None:
        return
    dx, dz = abs(start[0]), abs(start[1])

    if scenario == "vbar":
        # Goal 1: compare against BOTH reference two-impulse strategies.
        ref_vbar = dv_vbar_two_impulse_vv(dx, OMEGA)
        ref_rbar = dv_vbar_two_impulse_rr(dx, OMEGA)
        print(f"Reference two V-bar impulses (dx={dx:.1f} m): {ref_vbar:.5f} m/s "
              f"({r['total_dv'] / ref_vbar:.2f}x)" if ref_vbar > 0 else "")
        print(f"Reference two R-bar impulses (dx={dx:.1f} m): {ref_rbar:.5f} m/s "
              f"({r['total_dv'] / ref_rbar:.2f}x)" if ref_rbar > 0 else "")
    else:
        # Goal 2: compare against both R-bar strategies.
        ref_rv = dv_rbar_strategy_rv(dz, OMEGA)
        ref_vv = dv_rbar_strategy_vv(dz, OMEGA)
        print(f"Reference R-bar + V-bar impulse strategy (dz={dz:.1f} m): {ref_rv:.5f} m/s "
              f"({r['total_dv'] / ref_rv:.2f}x)" if ref_rv > 0 else "")
        print(f"Reference two V-bar impulses strategy    (dz={dz:.1f} m): {ref_vv:.5f} m/s "
              f"({r['total_dv'] / ref_vv:.2f}x)" if ref_vv > 0 else "")
