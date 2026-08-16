V-bar approach specialist.

Final policy: model/best_model.zip (239,904 steps of a 250,000-step run),
selected by deterministic evaluation, not by where training stopped. The
terminal checkpoint (model/vbar_td3.zip) is slightly worse: 1.121x vs 1.114x.

Performance (100 deterministic episodes from 1000 m, no exploration noise):
  dock rate      100%
  dv used        0.1279 m/s
  dv/opt         1.114x
  arrival speed  0.00179 m/s    a genuine rendezvous, not a fly-through
  burns          2 per episode
  transfer time  0.964 orbits

The policy flies a 0.921-orbit transfer rather than the one-orbit two-impulse
optimum, and executes it to within 2.7% of the best possible for that arc -- so
most of the 11% excess is the choice of arc, not sloppy execution.

Trained with --learning-starts 25000. At the config default of 5000 the actor
starts driving an untrained critic, saturates at the action-box corner, and the
run settles for brute-force thrusting at ~7x optimal.
