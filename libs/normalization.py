"""
State and action normalisation helpers.

Normalisation maps values symmetrically into [0, 1] via:
    norm(v) = clip(v, -scale, scale) / (2 * scale) + 0.5

This keeps network inputs centred and bounded regardless of sign
(positions/velocities can be negative).

Scale choices follow the physical problem:
  • position  → ENV_BOUNDARY              (excursion limit, ~200 m)
  • velocity  → OMEGA * ENV_BOUNDARY      (characteristic CW velocity scale)
  • dv_used   → DV_USED_NORM_SCALE        (typical cumulative Δv range)
  • action    → ENV_MAX_DV per axis       (per-step cap)

2-D state  : [x, z, ẋ, ż, dv_used]          (5-dim)
3-D state  : [x, y, z, ẋ, ẏ, ż, dv_used]    (7-dim)
Action     : [dvx, (dvy,) dvz]                     (2 or 3-dim)

NormalizedObsEnv wraps CWRendezvousEnv so training and evaluation always see
the same normalized observations — no need to call normalize_state by hand
at each call site.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from libs.constants import MODE_2D, ENV_BOUNDARY, ENV_MAX_DV_COEFF, DV_USED_NORM_SCALE, OMEGA

# --- Scale arrays (numpy; cast to tensor on demand) -------------------------
_POS_SCALE = ENV_BOUNDARY           # ~200 m
_VEL_SCALE = OMEGA * ENV_BOUNDARY   # characteristic CW velocity [m/s]
_DV_SCALE  = DV_USED_NORM_SCALE     # typical cumulative Δv range [m/s]

if MODE_2D:
    # state = [x, z, ẋ, ż, dv_used]
    STATE_SCALE = np.array(
        [_POS_SCALE, _POS_SCALE,
         _VEL_SCALE, _VEL_SCALE,
         _DV_SCALE],
        dtype=np.float64,
    )
    ACTION_SCALE = np.array([ENV_MAX_DV_COEFF, ENV_MAX_DV_COEFF], dtype=np.float64)
else:
    # state = [x, y, z, ẋ, ẏ, ż, dv_used]
    STATE_SCALE = np.array(
        [_POS_SCALE, _POS_SCALE, _POS_SCALE,
         _VEL_SCALE, _VEL_SCALE, _VEL_SCALE,
         _DV_SCALE],
        dtype=np.float64,
    )
    ACTION_SCALE = np.array(
        [ENV_MAX_DV_COEFF, ENV_MAX_DV_COEFF, ENV_MAX_DV_COEFF], dtype=np.float64
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

def normalize_state(state):
    """Normalise a state observation (numpy array or torch.Tensor) to [0, 1]."""
    if isinstance(state, torch.Tensor):
        return _normalize_tensor(state, STATE_SCALE)
    return _normalize_array(np.asarray(state, dtype=np.float64), STATE_SCALE)


def normalize_action(action):
    """Normalise an action (numpy array or torch.Tensor) to [0, 1]."""
    if isinstance(action, torch.Tensor):
        return _normalize_tensor(action, ACTION_SCALE)
    return _normalize_array(np.asarray(action, dtype=np.float64), ACTION_SCALE)


class NormalizedObsEnv(gym.ObservationWrapper):
    """Wraps CWRendezvousEnv so the policy always sees normalized obs in
    [0, 1], both at training and at evaluation time."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float64
        )

    def observation(self, observation):
        return normalize_state(observation)
