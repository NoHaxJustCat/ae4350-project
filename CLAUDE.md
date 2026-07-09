# Objective

The objective of this project is to build a Reinforcement Learning model capable of:

- Learning how to dock in VBAR, which in the LHLV frame is the x axis, when a simple Delta X displacement is present (like the one in constant.py). It must be able to find find both a good transfer from positive and negative displacements. Compare the delta v used to standard formulation of omega / 2 _ delta x for two R bar impulses or of omega / (3 _ pi) \* delta x for two V bar impulses
- Learning how to dock in RBAR, which in the LHLV frame is the z axis, when a displacement is present both in the x and z directions. the displacements have opposite sign in both cases (+x and -z / -x and +z). Compare it to these two cases:
  1.  Rbar impulse + V bar impulse
      ∆vz = ω \* ∆z
      ∆vx = 2 \* ∆vz
      ∆v_tot = 3 \* ∆vz
      ∆x = w / ω \* ∆vz = 2 \* ∆z
  2.  Two Vbar impulses
      ∆vx = ω /4 \* ∆z
      ∆vtot = 2 \* ∆vx = ω / 2 \* ∆z
      ∆x = 3π/4 \* ∆z

Both a single model capable of doing all trasfer of 2D or multiple could be created. Note: the STM already present is correct! It is in the LHLV frame, so keep it that way without changing it to Hill frame.

Finally, if the first two goals are achieved for a considerable distance (let's say 100m from target), one could also insert the y, a decoupled dynamics that introduce oscillations. THis is the third optional goal.
