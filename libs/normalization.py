"""
State and action normalisation helpers.

Normalisation maps values symmetrically into [0, 1] via:
    norm(v) = clip(v, -scale, scale) / (2 * scale) + 0.5

This keeps network inputs centred and bounded regardless of sign
(positions/velocities can be negative).

Scale choices follow the physical problem:
  • position  → ENV_BOUNDARY              (excursion limit, ~500 m)
  • velocity  → OMEGA * ENV_BOUNDARY      (characteristic CW velocity scale)
  • dv_budget → ENV_DV_BUDGET             (total budget, ~50 m/s)
  • action    → ENV_MAX_DV per axis       (per-step cap)

2-D state  : [x, z, ẋ, ż, dv_remaining]          (5-dim)
3-D state  : [x, y, z, ẋ, ẏ, ż, dv_remaining]    (7-dim)
Action     : [dvx, (dvy,) dvz]                     (2 or 3-dim)
"""

import numpy as np
import torch

from libs.constants import MODE_2D, ENV_BOUNDARY, ENV_MAX_DV, ENV_DV_BUDGET, OMEGA

# --- Scale arrays (numpy; cast to tensor on demand) -------------------------
_POS_SCALE = ENV_BOUNDARY           # ~500 m
_VEL_SCALE = OMEGA * ENV_BOUNDARY   # characteristic CW velocity [m/s]
_DV_SCALE  = ENV_DV_BUDGET          # total Δv budget [m/s]

if MODE_2D:
    # state = [x, z, ẋ, ż, dv_remaining]
    STATE_SCALE = np.array(
        [_POS_SCALE, _POS_SCALE,
         _VEL_SCALE, _VEL_SCALE,
         _DV_SCALE],
        dtype=np.float64,
    )
    ACTION_SCALE = np.array([ENV_MAX_DV, ENV_MAX_DV], dtype=np.float64)
else:
    # state = [x, y, z, ẋ, ẏ, ż, dv_remaining]
    STATE_SCALE = np.array(
        [_POS_SCALE, _POS_SCALE, _POS_SCALE,
         _VEL_SCALE, _VEL_SCALE, _VEL_SCALE,
         _DV_SCALE],
        dtype=np.float64,
    )
    ACTION_SCALE = np.array(
        [ENV_MAX_DV, ENV_MAX_DV, ENV_MAX_DV], dtype=np.float64
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