"""Observation normalisation to [0, 1].

    norm(v) = clip(v, -scale, scale) / (2*scale) + 0.5

The position/velocity scales are deliberately INDEPENDENT of
ENV_CURRICULUM_BOUNDARY_MULT. They used to be derived from ENV_BOUNDARY, which
meant raising the out-of-bounds limit silently rescaled every observation and
invalidated every saved model -- measured: the V-bar specialist went from 100%
to 0% dock rate. With the scale pinned, the same change leaves it at 100%.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from config import DV_USED_NORM_MULT, MODE_2D, OBS_POS_SCALE, OMEGA

_POS_SCALE = OBS_POS_SCALE
_VEL_SCALE = OMEGA * OBS_POS_SCALE
_N = 2 if MODE_2D else 3
_PHYS_SCALE = np.array([_POS_SCALE] * _N + [_VEL_SCALE] * _N, dtype=np.float64)


def normalize_state(state, scale):
    """Accepts a numpy array or a torch.Tensor."""
    if isinstance(state, torch.Tensor):
        scale_t = torch.as_tensor(scale, dtype=state.dtype, device=state.device)
        return (torch.clamp(state, -scale_t, scale_t) + scale_t) / (2.0 * scale_t)
    state = np.asarray(state, dtype=np.float64)
    return (np.clip(state, -scale, scale) + scale) / (2.0 * scale)


class NormalizedObsEnv(gym.ObservationWrapper):
    """Wraps CWRendezvousEnv so training and evaluation see identical obs."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, env.observation_space.shape,
                                            dtype=np.float64)

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float64)
        # dv_used's scale is per-episode (max_dv varies with scenario and
        # distance), so it cannot be baked into a fixed array. The trailing
        # braking flag is already 0/1 and passes through unnormalized.
        scale = np.concatenate([_PHYS_SCALE, [DV_USED_NORM_MULT * self.unwrapped.max_dv]])
        return np.concatenate([normalize_state(observation[:-1], scale), observation[-1:]])
