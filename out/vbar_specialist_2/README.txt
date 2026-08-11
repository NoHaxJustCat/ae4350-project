V-bar specialist v2.

Origin: best_model of run random_20260727_152029 at 319998 steps (scenario=random, sector +-0.25 deg).

Refined from out/vbar_specialist and strictly better at V-bar:
  v1: |u_burn| 0.162 -> 1.15x optimal
  v2: |u_burn| 0.150 -> 1.04x optimal
Loads in both the vbar and random scenarios (ENV_RANDOM_DV_REF_MULT =
3*pi/4 makes their actuator scales identical at V-bar).
