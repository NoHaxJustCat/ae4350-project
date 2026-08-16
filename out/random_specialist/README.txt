Random-direction rendezvous specialist.

Final policy: model/best_model.zip (929,640 steps), selected by deterministic
evaluation at the 1000 m curriculum stage, not by where training stopped.

Performance (200 deterministic episodes from 1000 m, no exploration noise):
  dock rate      91.5%          full circle of start directions
  circle sweep   34/36 = 94%    deterministic, every 10 deg
  dv/opt         1.59x median   [p10 1.46, p90 1.85]
  arrival speed  3.21 m/s       NOT braking -- see the caveat below
  burns          2 per episode

Dock rate by angle from the V-bar axis:
     0-5 deg      0%      the residual failure: a +-5 deg cone
     5-10 deg    23%
    10-20 deg    87%
    20-45 deg   100%
    45-90 deg   100%

CAVEAT: arrival speed is ~1.4x dv_opt, so the agent reaches the target without
nulling its relative velocity. The analytic references assume a full stop, so
the 1.59x figure is NOT a like-for-like fuel comparison -- it is a fly-through.

Training chain (each stage resumed from the previous best_model):
  1. 0      -> 500k   runs/random.ps1, warmup 25000, noise 0.10 -> 0.005/0.6
  2. 500k   -> 1M     resumed; noise continued at 0.0209 -> 0.005 over 0.2
  3. 1M     -> 1.19M  resumed with a seeded replay buffer (see below)

Stage 3 is what fixed the direction coverage. The policy had converged with a
+-10 deg hole around the V-bar axis and noise at its floor, so it could not
explore out of it. scripts/seed_buffer.py collected 60,000 uniform-random
transitions inside that band at 30-300 m, where random actions dock 3-14% of
the time against 0.1-0.5% at 1000 m, and train.py --load-replay-buffer seeded
them. Circle coverage went 83% -> 94%.

Two alternatives were measured and rejected: training only inside the band at
1000 m drove in-band dock from 15% to 5% (nothing to learn from), and simply
training longer left the failure count pinned at 6/36 while shuffling which
angles failed.
