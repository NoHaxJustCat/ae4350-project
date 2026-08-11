"""
Enforces the CW rendezvous MDP's exact mirror symmetry: dynamics are linear
and reward is built from norms, so the MDP is invariant under
(position, velocity, action) -> -(position, velocity, action).

Rather than hope a generic MLP learns this from data (a policy trained mostly
on x>0 would have to extrapolate to x<0), this wrapper enforces it
structurally: an x<0 observation is mirrored into the x>=0 canonical view
before the policy sees it, and the resulting action is mirrored back before
being applied. The policy only ever solves the x>=0 half.

Must wrap the RAW env (before NormalizedObsEnv) — needs the true signed x.
"""

import numpy as np
import gymnasium as gym

from config import ACTION_IMPULSE_DIM, PHYS_STATE_DIM


class CanonicalizeDirectionEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mirror = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        canon_obs, self._mirror = self._canonicalize(obs)
        return canon_obs, info

    def step(self, action):
        # Mirror ONLY the physical impulse components. The trailing
        # coast-duration scalar (action[ACTION_IMPULSE_DIM:]) is a magnitude,
        # not a direction — negating it would flip the requested coast length
        # under the x<0 mirror and desync the two halves of the symmetry.
        if self._mirror:
            real_action = np.asarray(action, dtype=np.float64).copy()
            real_action[:ACTION_IMPULSE_DIM] *= -1
        else:
            real_action = action
        obs, reward, terminated, truncated, info = self.env.step(real_action)
        canon_obs, self._mirror = self._canonicalize(obs)
        return canon_obs, reward, terminated, truncated, info

    @staticmethod
    def _canonicalize(obs: np.ndarray):
        mirror = obs[0] < 0
        if not mirror:
            return obs, False
        canon = obs.copy()
        # Negate only the physical state (position + velocity). The trailing
        # dv_used and braking-phase flag are sign-invariant magnitudes and must
        # be left untouched.
        canon[:PHYS_STATE_DIM] *= -1
        return canon, True
