# Autonomous orbital rendezvous with deep reinforcement learning

A TD3 agent that flies a chaser spacecraft to a target in Clohessy–Wiltshire
relative motion, using impulsive burns and ballistic coasts, and is graded
against the analytically optimal two-impulse transfer.

Two trained policies ship with the repository:

| model | scenario | dock rate | Δv / Δv_opt | arrival speed |
|---|---|---|---|---|
| [`out/vbar_specialist`](out/vbar_specialist) | approach along V-bar from 1000 m | 100 % | **1.114x** | 0.0018 m/s |
| [`out/random_specialist`](out/random_specialist) | approach from any direction | 91.5 % | 1.59x | 3.21 m/s |

`Δv_opt` is the true minimum for that geometry, found numerically — not a
textbook formula. For a pure V-bar displacement it agrees with the classical
ω/(3π)·Δx two-impulse result to 0.14 %.

Read the two numbers together. The V-bar policy arrives essentially at rest, so
its 1.114x is a like-for-like comparison. The random policy arrives at 3.2 m/s,
so it performs a fly-through rather than a rendezvous and its 1.59x is **not**
comparable to references that assume a full stop.

## The problem

The chaser starts at rest a fixed distance from the target and must reach it
with a closest approach under 5 m. Relative motion follows the linearised
Clohessy–Wiltshire equations at a 600 km circular orbit.

Each action is `[Δv_x, Δv_z, coast]`: one impulsive burn, then a ballistic
coast of agent-chosen length (1–72 decision steps, one orbit ≈ 58). Because a
whole coast is one decision, a complete rendezvous is about **two decisions**
rather than hundreds of thrust commands. Reaching the target opens a one-step
braking phase for a dedicated final impulse to null the arrival velocity.

Reward combines dense distance shaping, a docking bonus, a fuel term graded
against `Δv_opt`, and a stopping bonus graded on arrival speed — the last three
paid only on a completed dock.

Scenarios: `vbar` (displaced along the velocity vector), `rbar` (along the
radius vector) and `random` (uniform over the circle).

## Setup

Python 3.11+, then:

```bash
pip install "stable-baselines3[extra]" gymnasium torch numpy matplotlib
```

**A GPU is not required.** `DEVICE` in `config.py` resolves to CUDA when a GPU
is visible and CPU otherwise, so the commands below work unchanged either way.
The networks are small (256×256), so CPU training is roughly 2–3x slower, not
orders of magnitude. To force it:

```bash
AE4350_DEVICE=cpu python scripts/eval.py ...
```

`NUM_ENVS` defaults to 6 parallel environments, tuned for a 6-core machine.
On fewer cores, lower it — oversubscribing makes the workers fight for cores
and is slower than using fewer:

```bash
AE4350_NUM_ENVS=4 python scripts/train.py ...
```

LaTeX is optional. Plots use it for typesetting when available and fall back to
matplotlib's mathtext otherwise.

## Evaluating the trained models

`scripts/eval.py` rolls a checkpoint out deterministically — no exploration
noise, no gradient updates, no curriculum — and reports the distribution of
outcomes over many episodes. This is the tool that answers "is this checkpoint
any good"; the training log cannot, because it reports the *noisy* behaviour
policy.

```bash
# V-bar specialist, 100 episodes from 1000 m
python scripts/eval.py out/vbar_specialist/model/best_model.zip --scenario vbar

# random specialist over the full circle of start directions
python scripts/eval.py out/random_specialist/model/best_model.zip --scenario random -n 200

# how performance varies with start direction — the key view for `random`
python scripts/eval.py out/random_specialist/model/best_model.zip --scenario random --angle-sweep

# a single forced direction, or a different start distance
python scripts/eval.py out/random_specialist/model/best_model.zip --scenario random --angle 45
python scripts/eval.py out/vbar_specialist/model/best_model.zip --scenario vbar --distance 500
```

Reading the output:

- **`dv/opt`** is the headline. 1.00 means the policy matched the best
  two-impulse maneuver that physically exists for that start.
- **Always read it next to `arrival speed`.** A policy that skips its terminal
  brake scores *below* 1.00 while flying through rather than rendezvousing. The
  printed `arrival speed / dv_opt` flags this: above ~0.05 the fuel ratio is
  not comparable to the analytic references.
- **`steps / burns`** should be about 2 for a good policy. Much higher means it
  is thrusting its way in instead of coasting.

Each shipped model has a `README.txt` recording its provenance, training chain
and known limitations.

### Plotting a trajectory

```bash
python scripts/plot_run.py out/vbar_specialist/model/best_model.zip --scenario vbar
python scripts/plot_run.py out/random_specialist/model/best_model.zip --scenario random --angle 45
```

Writes a figure of the relative-motion path with every Δv impulse drawn as an
arrow, next to `<run>/episodes/`.

## Training your own

Launchers live in `runs/` and write to `tmp/<run-tag>/`. They are PowerShell
scripts; the underlying command is plain Python, shown below each.

```bash
.\runs\vbar.ps1           # V-bar specialist, 100k steps
.\runs\rbar.ps1           # R-bar specialist
.\runs\random.ps1         # random directions, 1M steps
```

Equivalent direct invocation:

```bash
python scripts/train.py --scenario vbar --total-timesteps 100000 --learning-starts 25000 --run-tag my_run
```

**`--learning-starts` matters more than anything else here.** It sets how many
uniform-random steps fill the replay buffer before the actor takes over and
gradient updates begin. At the library default of 5000 the actor starts driving
an untrained critic, saturates at the edge of the action box, and settles for
brute-force thrusting — measured at 7.2x optimal on V-bar against 1.08x at
25000, and 5–20x against ~1.6x on `random`. The warmup is nearly free: it runs
at ~1200 steps/s against ~80 once gradients start.

Useful flags:

| flag | effect |
|---|---|
| `--total-timesteps N` | absolute target, **including** a resumed model's own steps |
| `--learning-starts N` | uniform-random warmup before updates begin |
| `--noise-std-start/end`, `--noise-decay-frac` | exploration schedule |
| `--resume-from CKPT.zip` | continue from a checkpoint |
| `--no-curriculum` | train at the final distance from step 0 |
| `--net-width N` | width of both hidden layers and the encoder |
| `--eval-freq N` | deterministic evaluation cadence (0 disables) |

### Watching a run

Each run writes live, refreshed every 60 s:

```
tmp/<run-tag>/diagnostics/plots/   reward, dock rate, dv_ratio, curriculum, losses
tmp/<run-tag>/episodes/            trajectory snapshots
tmp/<run-tag>/model/status.json    machine-readable progress
tmp/<run-tag>/model/best_model.zip best checkpoint by deterministic evaluation
```

`best_model.zip` is the artifact to keep. It is written only when the
deterministic evaluation improves *at the hardest curriculum stage reached*, so
it survives the late-training collapses that a terminal checkpoint does not.

### Resuming

```bash
.\runs\random_resume.ps1 -Run <run-tag> -TotalTimesteps 1000000
```

The replay buffer is **not** saved with a checkpoint, so a resume starts with
an empty one. `--learning-starts` is therefore re-anchored on resume to mean
"collect this many fresh transitions before any update runs" — without that,
updates would restart against a nearly empty buffer and can destroy a good
policy in a few thousand steps. Expect the first stretch of a resumed run to be
exploration only, with a dock rate that is not comparable to what came before.

### Curriculum

Training ramps the start distance from 30 m to 1000 m, advancing when a window
of episodes clears a dock-rate threshold and regressing after a sustained
stall. This matters because the docking basin scales as 5/d — the tolerance is
a fixed 5 m while everything else scales with distance — so the task is roughly
12x more forgiving at 30 m than at 1000 m.

### Seeding the buffer for a stuck region

The `random` policy converges with a hole around the V-bar axis: noise reaches
its floor and it can no longer explore out. `scripts/seed_buffer.py` collects
uniform-random experience inside a chosen sector and `--load-replay-buffer`
injects it:

```bash
python scripts/seed_buffer.py --out tmp/seed.npz --half-width 15 \
    --transitions 60000 --min-distance 30 --max-distance 300
python scripts/train.py --scenario random --resume-from CKPT.zip \
    --total-timesteps N --load-replay-buffer tmp/seed.npz
```

Collect at **short range**. Uniform-random actions dock 3–14 % of the time at
30 m against 0.1–0.5 % at 1000 m, and Clohessy–Wiltshire is linear, so the
optimal normalized burn and coast depend only on direction, not distance — what
is learned at 30 m is what is needed at 1000 m. This took circle coverage from
83 % to 94 %.

## Analytic references

`lib/astro/reference.py` provides the closed-form transfers the agent is graded
against, for a displacement Δx along V-bar:

| strategy | Δv | Δv / Δv_opt | time |
|---|---|---|---|
| numerical optimum | 0.11476 m/s | 1.000 | 0.992 orbits |
| two V-bar impulses, ω/(3π)·Δx | 0.11492 m/s | 1.001 | 1.000 orbits |
| two R-bar impulses, (ω/2)·Δx | 0.54155 m/s | 4.719 | 0.500 orbits |

at Δx = 1000 m. The numerical optimum searches over coast duration and burn
direction; the two-impulse V-bar transfer fixes the coast at exactly one orbit
with both burns along-track, and is optimal to 0.14 %. The R-bar figure is 4.7x
more expensive but takes half the time — a fuel-for-time trade, not simply a
worse maneuver.

## Repository layout

```
config.py              every hyperparameter and physical constant
lib/astro/             CW dynamics and the analytic Δv references
lib/rl/env.py          the Gymnasium environment
lib/rl/obs.py          observation normalisation
lib/rl/symmetry.py     mirror-symmetry wrapper (the policy only solves x >= 0)
lib/rl/callbacks/      curriculum, checkpointing, best-model evaluation, logging
lib/plots/             trajectory and diagnostics figures
scripts/train.py       training entry point
scripts/eval.py        deterministic evaluation
scripts/plot_run.py    trajectory figure for a saved model
scripts/plot.py        diagnostics from a saved history.npz
runs/                  launchers, one per scenario
out/                   the shipped models and the ablation study
```

The environment is wrapped as `NormalizedObsEnv(CanonicalizeDirectionEnv(env))`
during training, and evaluation must use the same stack — the policy has only
ever seen the canonicalised x ≥ 0 view.

## Ablation study

`out/ablations/` holds a one-knob-at-a-time study (learning rate, gamma,
network width, replay, curriculum, actuator scale) generated by
`runs/ablations.ps1` and collected with `scripts/collect_ablations.py`.
