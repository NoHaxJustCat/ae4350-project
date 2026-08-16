"""
Pre-fill a replay buffer with uniform-random experience on a chosen sector.

The "random" policy converges with a hole within ~10 deg of the V-bar axis: it
docks 83% over the circle but flies out of bounds on exactly that band. Noise
is at its floor by then, so it cannot explore its way out -- the buffer holds
almost nothing but its own failures there.

This collects targeted experience instead, writes it as an SB3 replay buffer,
and `train.py --load-replay-buffer` restores it so a resumed run trains against
data that actually contains successes in the failing region.

WHERE to collect matters more than how much. Uniform-random actions dock at
these rates (measured, 4000 episodes per cell):

        angle    1000 m    100 m     30 m
        0 deg     0.47%    4.60%   13.70%
       10 deg     0.10%    1.27%    3.45%
       90 deg     0.05%    1.05%    3.88%

At 1000 m a full buffer would hold a few hundred successes against 150k
failures, which teaches a critic that the region is hopeless. At 30 m it is
~29x richer. CW is linear, so the optimal normalized burn and coast time depend
only on DIRECTION, not distance -- short-range experience carries the same
lesson the policy needs at long range.

Usage:
    python scripts/seed_buffer.py --out tmp/seed.pkl --half-width 15 \
        --transitions 60000 --min-distance 30 --max-distance 300
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse

import numpy as np
from config import MAX_STEPS, OMEGA
from lib.rl.env import CWRendezvousEnv
from lib.rl.obs import NormalizedObsEnv
from lib.rl.symmetry import CanonicalizeDirectionEnv


def build_env(distance):
    """The training wrapper stack exactly: transitions have to land in the same
    observation and action space the policy is trained in, or the buffer is
    poison rather than data."""
    raw = CWRendezvousEnv(omega=OMEGA, scenario="random", curriculum_enabled=False,
                          curriculum_max_distance=distance)
    return raw, NormalizedObsEnv(CanonicalizeDirectionEnv(raw))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="Output .npz")
    p.add_argument("--transitions", type=int, default=60000)
    p.add_argument("--half-width", type=float, default=15.0,
                   help="Collect within +-this many degrees of the V-bar axis.")
    p.add_argument("--min-distance", type=float, default=30.0)
    p.add_argument("--max-distance", type=float, default=300.0,
                   help="Log-uniform between the two. Keep the upper end low: "
                        "random actions almost never dock at 1000 m.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    # One env per distance would rebuild the dv_opt table every episode, so
    # build once at the max and push the sampled distance in per reset.
    raw, env = build_env(args.max_distance)
    # Raw arrays, not a pickled ReplayBuffer: a buffer is shaped for a fixed
    # n_envs and training runs NUM_ENVS of them, so a pickled single-env buffer
    # cannot be dropped into a six-env run. train.py inserts these in chunks.
    obs_l, next_l, act_l, rew_l, done_l = [], [], [], [], []

    stored, episodes, docks = 0, 0, 0
    while stored < args.transitions:
        # Log-uniform: the basin scales as 5/d, so a linear draw would spend
        # most episodes at distances where random actions never succeed.
        dist = float(np.exp(rng.uniform(np.log(args.min_distance),
                                        np.log(args.max_distance))))
        raw.set_curriculum_distance(dist)
        hw = np.radians(args.half_width)
        base = 0.0 if rng.random() < 0.5 else np.pi
        angle = base + float(rng.uniform(-hw, hw))

        obs, _ = env.reset(seed=int(rng.integers(1 << 30)), options={"angle": angle})
        for _ in range(MAX_STEPS):
            action = rng.uniform(-1.0, 1.0, env.action_space.shape[0])
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs_l.append(obs); next_l.append(next_obs); act_l.append(action)
            rew_l.append(reward); done_l.append(terminated)
            stored += 1
            obs = next_obs
            if terminated or truncated:
                break
        episodes += 1
        docks += bool(info.get("docked", False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=np.asarray(obs_l, dtype=np.float32),
        next_obs=np.asarray(next_l, dtype=np.float32),
        actions=np.asarray(act_l, dtype=np.float32),
        rewards=np.asarray(rew_l, dtype=np.float32),
        dones=np.asarray(done_l, dtype=np.float32),
    )

    print(f"sector      : +-{args.half_width:.1f} deg around V-bar")
    print(f"distances   : {args.min_distance:.0f} - {args.max_distance:.0f} m (log-uniform)")
    print(f"transitions : {stored}  over {episodes} episodes")
    print(f"docked      : {docks} ({100 * docks / episodes:.2f}% of episodes)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
