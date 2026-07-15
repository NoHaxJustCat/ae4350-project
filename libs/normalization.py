"""
State normalisation helpers.

Normalisation maps values symmetrically into [0, 1] via:
    norm(v) = clip(v, -scale, scale) / (2 * scale) + 0.5

This keeps network inputs centred and bounded regardless of sign
(positions/velocities can be negative).

Scale choices follow the physical problem:
  • position  → ENV_BOUNDARY                    (excursion limit; derived
                                                   from the curriculum range,
                                                   see constants.py)
  • velocity  → OMEGA * ENV_BOUNDARY             (characteristic CW velocity scale)
  • dv_used   → DV_USED_NORM_MULT * env.max_dv   (per-episode: max_dv already
                                                   varies ~12x between "vbar"
                                                   and "rbar" and linearly
                                                   with curriculum distance,
                                                   so this can't be a fixed
                                                   constant — see NormalizedObsEnv)

2-D state  : [x, z, ẋ, ż, dv_used]          (5-dim)
3-D state  : [x, y, z, ẋ, ẏ, ż, dv_used]    (7-dim)

NormalizedObsEnv wraps CWRendezvousEnv so training and evaluation always see
the same normalized observations — no need to call normalize_state by hand
at each call site. There is no analogous action normalisation: the env's
action_space is already the native [-1, 1] fraction the actor outputs, and
CWRendezvousEnv.step() does the (per-episode, distance-dependent) physical
scaling internally.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from libs.constants import MODE_2D, ENV_BOUNDARY, DV_USED_NORM_MULT, OMEGA

# --- Scale arrays (numpy; cast to tensor on demand) -------------------------
_POS_SCALE = ENV_BOUNDARY           # excursion-limit ceiling [m]
_VEL_SCALE = OMEGA * ENV_BOUNDARY   # characteristic CW velocity [m/s]

if MODE_2D:
    # phys state = [x, z, ẋ, ż]
    _PHYS_SCALE = np.array(
        [_POS_SCALE, _POS_SCALE,
         _VEL_SCALE, _VEL_SCALE],
        dtype=np.float64,
    )
else:
    # phys state = [x, y, z, ẋ, ẏ, ż]
    _PHYS_SCALE = np.array(
        [_POS_SCALE, _POS_SCALE, _POS_SCALE,
         _VEL_SCALE, _VEL_SCALE, _VEL_SCALE],
        dtype=np.float64,
    )


# --- Low-level helpers -------------------------------------------------------

def _normalize_array(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -scale, scale)
    return (clipped + scale) / (2.0 * scale)


def _normalize_tensor(values: torch.Tensor, scale: np.ndarray) -> torch.Tensor:
    scale_t = torch.as_tensor(scale, dtype=values.dtype, device=values.device)
    clipped = torch.clamp(values, -scale_t, scale_t)
    return (clipped + scale_t) / (2.0 * scale_t)


# --- Public API --------------------------------------------------------------

def normalize_state(state, scale: np.ndarray):
    """Normalise a state observation (numpy array or torch.Tensor) to [0, 1]
    using the given per-element scale (last element is the episode's
    dv_used scale; see NormalizedObsEnv)."""
    if isinstance(state, torch.Tensor):
        return _normalize_tensor(state, scale)
    return _normalize_array(np.asarray(state, dtype=np.float64), scale)


class NormalizedObsEnv(gym.ObservationWrapper):
    """Wraps CWRendezvousEnv so the policy always sees normalized obs in
    [0, 1], both at training and at evaluation time."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float64
        )

    def observation(self, observation):
        # dv_used's scale is per-episode (depends on that episode's
        # scenario/distance via max_dv, set in CWRendezvousEnv.reset()), so
        # it's computed fresh here rather than baked into a fixed array.
        # The trailing braking-phase flag is already a 0/1 indicator (see
        # OBS_DIM in constants.py) — pass it through unnormalized rather than
        # squashing it toward 0.5 like a signed physical quantity.
        observation = np.asarray(observation, dtype=np.float64)
        dv_scale = DV_USED_NORM_MULT * self.unwrapped.max_dv
        scale = np.concatenate([_PHYS_SCALE, [dv_scale]])
        core = normalize_state(observation[:-1], scale)  # phys state + dv_used
        flag = observation[-1:]                          # braking-phase flag
        return np.concatenate([core, flag])
